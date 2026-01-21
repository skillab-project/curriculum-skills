from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from urllib.parse import unquote
import logging

from recommendation_system.backend.database import get_db
from recommendation_system.backend.models import University
from recommendation_system.backend.course_recommender_for_university import CourseRecommender as CourseRecommenderV2
from recommendation_system.backend.degree_recommender_for_university import UniversityRecommender
from recommendation_system.backend.student_recommender import CourseRecommender as CourseRecommenderV3
from recommendation_system.backend.models import Course

from recommendation_system.backend.schemas import (
    UserPreferences,
    CourseRecommendationsResponse,
    RecommendedCourse,
    UniversityProfileOut,
    SimilarUniversityOut,
    DegreeWithSkillsOut,
    DegreesWithSkillsResponse,
    ExistingDegreeCourseRequest,
    NewDegreeCourseRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# backend/student_recommender.py (μέσα στην κλάση CourseRecommender)

from sqlalchemy.orm import Session
from collections import Counter

class CourseRecommender:


    def __init__(self, db: Session):
        self.db = db

# ---------------------------
# Νέα μέθοδος: History-based recommendations
# ---------------------------
def recommend_based_on_history(self, user_id: int, top_n: int = 10):
    def normalize_name(n: str) -> str:
        if not n:
            return ""
        # collapse multiple whitespace, strip, and casefold for stable comparison
        return " ".join(n.split()).casefold()

    history = self.db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    if not history:
        return []

    # collect user skills (filter None, normalize skill names too)
    user_skills = set()
    for h in history:
        if h.skills:
            for sk in h.skills:
                if sk:
                    user_skills.add(" ".join(str(sk).split()))

    all_degrees = self.db.query(DegreeProgram).all()
    viewed_courses = {h.course_name for h in history if h.course_name}

    temp_recs = []
    for degree in all_degrees:
        # skip degree if user already viewed any course of that degree
        degree_course_names = {c.lesson_name for c in getattr(degree, "courses", []) if c.lesson_name}
        if viewed_courses & degree_course_names:
            continue

        degree_skills = []
        for course in getattr(degree, "courses", []) or []:
            for s in getattr(course, "skills", []) or []:
                name = getattr(s, "skill_name", None)
                if name:
                    degree_skills.append(" ".join(str(name).split()))

        common_skills = user_skills & set(degree_skills)
        score = len(common_skills)
        if score > 0:
            temp_recs.append({
                "degree_name": degree.name or "",
                "degree_name_norm": normalize_name(degree.name or ""),
                "score": float(score),
                "matching_skills": list(common_skills),
                "source_universities": [getattr(degree, "institution", None)]  # προαιρετικό
            })

    # Deduplicate by normalized name, keep entry with highest score and merge skills/sources
    unique = {}
    for r in temp_recs:
        key = r["degree_name_norm"]
        if not key:
            continue
        if key not in unique:
            unique[key] = {
                "degree_name": r["degree_name"],
                "score": r["score"],
                "matching_skills": set(r["matching_skills"]),
                "source_universities": list(filter(None, r.get("source_universities", [])))
            }
        else:
            # keep the one with higher score (or update score to max)
            unique[key]["score"] = max(unique[key]["score"], r["score"])
            unique[key]["matching_skills"].update(r["matching_skills"])
            unique[key]["source_universities"].extend(filter(None, r.get("source_universities", [])))

    final_results = []
    for v in unique.values():
        final_results.append({
            "degree_name": v["degree_name"],
            "score": v["score"],
            "matching_skills": sorted(list(v["matching_skills"])),
            "source_universities": sorted(list(set(v["source_universities"])))
        })

    return sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_n]



@router.get("/recommend/personalized/history/{user_id}")
def recommend_personalized_history(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    recommender = CourseRecommenderV3(db)
    results = recommender.recommend_based_on_history(user_id, top_n)
    return {"user_id": user_id, "recommendations": results}


@router.get(
    "/recommend/new_degree/{degree_name}",
    response_model=CourseRecommendationsResponse,
    summary="Recommend courses for a new degree across all universities."
)
async def recommend_courses_for_new_degree(
    degree_name: str = Path(..., description="URL-encoded name of the new degree."),
    top_n_courses: int = 10,
    db: Session = Depends(get_db)
):
    """
    Return recommended courses for a new degree based on similar degrees from other universities.
    Does not exclude existing courses since this is a completely new degree.
    """
    try:
        decoded_degree_name = unquote(degree_name).strip()
        recommender = CourseRecommenderV2(db)

        logger.info(f"Request for new degree_name='{decoded_degree_name}'")

        # 1️⃣ Gather all degree profiles from all universities
        all_univs = recommender.get_all_universities()
        all_profiles: List[Dict[str, Any]] = []
        for u in all_univs:
            profiles = recommender.build_degree_profiles(u.university_id)
            if profiles:
                all_profiles.extend(profiles)

        if not all_profiles:
            logger.warning("No degree profiles found in any university.")
            raise HTTPException(status_code=404, detail="No degree profiles found in any university.")

        # 2️⃣ Find similar degrees by matching normalized degree name
        similar_degrees = [
            p for p in all_profiles
            if recommender.normalize_name(p.get("degree_title")) == recommender.normalize_name(decoded_degree_name)
        ]

        # Fallback to all profiles if no exact match
        if not similar_degrees:
            logger.info("No exact degree name matches, using all profiles as similar degrees.")
            similar_degrees = all_profiles

        # 3️⃣ Aggregate all skills from similar degrees
        all_skills = set()
        for p in similar_degrees:
            all_skills.update(p.get("skills", []) or [])

        # 4️⃣ Suggest courses for the new degree
        try:
            result = recommender.suggest_courses_for_new_degree(
                similar_degrees=similar_degrees,
                target_skills=all_skills,
                top_n=top_n_courses
            )
        except Exception as e:
            logger.error(f"Error in suggest_courses_for_new_degree: {e}")
            raise HTTPException(status_code=500, detail="Course recommendation failed due to internal error.")

        # 5️⃣ Format final recommendations
        final_recommendations = [
            {
                "course_name": item.get("course_name", "Unknown"),
                "score": item.get("score", 0.0),
                "description": item.get("description", ""),
                "objectives": item.get("objectives", ""),
                "learning_outcomes": item.get("learning_outcomes", ""),
                "course_content": item.get("course_content", ""),
                "new_skills": sorted(item.get("new_skills", [])),
                "compatible_skills": sorted(item.get("compatible_skills", [])),
            }
            for item in result
            if isinstance(item, dict) and "course_name" in item
        ]

        # 6️⃣ Return final response
        return CourseRecommendationsResponse(
            university_id=-1,  # Not tied to a specific university
            program_id=-1,
            degree=decoded_degree_name,
            recommendations=final_recommendations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in recommend_courses_for_new_degree: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.get(
    "/recommend/courses/{university_id}/{degree_name}",
    response_model=CourseRecommendationsResponse,
    summary="Recommend courses for a specific degree at a specific university."
)
async def recommend_courses_by_name_safe(
    university_id: int = Path(..., description="University ID"),
    degree_name: str = Path(..., description="URL-encoded name of the degree"),
    top_n_courses: int = 10,
    db: Session = Depends(get_db)
):
    """
    Return recommended courses for a specific degree, based on similar degree programs.
    """
    try:
        decoded_degree_name = unquote(degree_name).strip()
        recommender = CourseRecommenderV2(db)

        logger.info(f"Request for university_id={university_id}, degree_name='{decoded_degree_name}'")

        # 1️⃣ Gather all degree profiles
        all_univs = recommender.get_all_universities()
        all_profiles: List[Dict[str, Any]] = []
        for u in all_univs:
            profiles = recommender.build_degree_profiles(u.university_id)
            if profiles:
                all_profiles.extend(profiles)

        if not all_profiles:
            logger.warning("No degree profiles found in any university.")
            raise HTTPException(status_code=404, detail="No degree profiles found in any university.")

        # 2️⃣ Find representative profiles matching the degree name
        representative_profiles = [
            p for p in all_profiles
            if recommender.normalize_name(p.get("degree_title")) == recommender.normalize_name(decoded_degree_name)
        ]

        if not representative_profiles:
            logger.warning(f"Degree '{decoded_degree_name}' not found in any university.")
            raise HTTPException(
                status_code=404,
                detail=f"Degree '{decoded_degree_name}' not found in any university."
            )

        # 3️⃣ Create a synthetic target degree combining skills and courses
        degree_type = representative_profiles[0].get("degree_type", "N/A")
        all_skills = set()
        all_courses = set()
        for p in representative_profiles:
            all_skills.update(p.get("skills", []) or [])
            all_courses.update(p.get("courses", []) or [])

        synthetic_target_degree = {
            "university_id": university_id,
            "program_id": -1,
            "degree_title": decoded_degree_name,
            "degree_type": degree_type,
            "skills": list(all_skills),
            "courses": list(all_courses),
        }

        # 4️⃣ Find similar degrees
        similar_degrees = recommender.find_similar_degrees(
            synthetic_target_degree,
            all_profiles,
            top_n=5
        )

        if not similar_degrees:
            logger.info("No similar degrees found. Returning empty recommendations.")
            return CourseRecommendationsResponse(
                university_id=university_id,
                program_id=-1,
                degree=decoded_degree_name,
                recommendations=[]
            )

        # 5️⃣ Suggest courses
        try:
            result = recommender.suggest_courses_for_degree(
                synthetic_target_degree,
                similar_degrees,
                top_n=top_n_courses
            )
        except Exception as e:
            logger.error(f"Error in suggest_courses_for_degree: {e}")
            raise HTTPException(status_code=500, detail="Course recommendation failed due to internal error.")

        # 6️⃣ Format final recommendations
        final_recommendations = [
            {
                "course_name": item.get("course_name", "Unknown"),
                "score": item.get("score", 0.0),
                "description": item.get("description", ""),
                "objectives": item.get("objectives", ""),
                "learning_outcomes": item.get("learning_outcomes", ""),
                "course_content": item.get("course_content", ""),
                "new_skills": sorted(item.get("new_skills", [])),
                "compatible_skills": sorted(item.get("compatible_skills", [])),
            }
            for item in result
            if isinstance(item, dict) and "course_name" in item
        ]

        # 7️⃣ Return response
        return CourseRecommendationsResponse(
            university_id=university_id,
            program_id=-1,
            degree=decoded_degree_name,
            recommendations=final_recommendations
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in recommend_courses_by_name_safe: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.get("/recommend/degrees/{university_id}", summary="Recommend degrees for a university.")
def recommend_degrees(university_id: int, top_n: int = 5, db: Session = Depends(get_db)):
    """
    Recommend top N degrees for a university based on recognized skills.
    """
    recommender = UniversityRecommender(db)
    results = recommender.suggest_degrees_with_skills(university_id, top_n=top_n)
    return {"university_id": university_id, "recommended_degrees": results}


@router.get("/recommendations/university/{univ_id}", summary="Suggest courses for a university.")
def suggest_courses_for_university(univ_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    """
    Suggest top N courses for a specific university.
    """
    recommender = CourseRecommenderV2(db)
    result = recommender.suggest_courses(univ_id, top_n)
    return {"university_id": univ_id, "recommendations": result}


@router.post("/recommend/personalized", summary="Recommend personalized courses based on user preferences.")
def recommend_personalized(preferences: UserPreferences, db: Session = Depends(get_db)):
    """
    Recommend courses tailored to user preferences including target skills, language, country, and degree type.
    """
    try:
        recommender = CourseRecommenderV3(db)
        results = recommender.recommend_personalized(
            target_skills=preferences.target_skills,
            language=preferences.language,
            country=preferences.country,
            degree_type=preferences.degree_type,
            top_n=preferences.top_n
        )
        return results
    except Exception as e:
        logger.exception(f"Error in recommend_personalized: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

        
from pydantic import BaseModel
from recommendation_system.backend.models import UserInteraction
from recommendation_system.backend.models import DegreeProgram, Course, UserInteraction

class InteractionIn(BaseModel):
    user_id: int
    course_name: str
    interest_score: float = 1.0  # default 1.0, αυξάνεται αν enrolled/confirmed


# ---------------------------
# Endpoint για αποθήκευση interaction
# ---------------------------
@router.post("/interactions/add")
def add_interaction(data: InteractionIn, db: Session = Depends(get_db)):

    course = db.query(Course).filter(Course.lesson_name == data.course_name).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    user_skills = [cs.skill.skill_name for cs in course.skills]

    interaction = UserInteraction(
        user_id=data.user_id,
        course_name=data.course_name,
        skills=user_skills,
        interest_score=data.interest_score
    )

    db.add(interaction)
    db.commit()
    db.refresh(interaction)

    return {"status": "ok", "interaction_id": interaction.id}


# ---------------------------
# Κλάση CourseRecommender
# ---------------------------
class DegreeRecommender:
    def __init__(self, db: Session):
        self.db = db

    def recommend_based_on_history(self, user_id: int, top_n: int = 10):

        # --- helper for perfect deduplication ---
        def normalize(name: str) -> str:
            if not name:
                return ""
            n = name.replace("\u00A0", " ")      # non-breaking spaces → normal spaces
            n = " ".join(n.split())              # collapse multiple spaces
            return n.strip().lower()             # trim + lowercase

        # --- load user history ---
        history = (
            self.db.query(UserInteraction)
            .filter(UserInteraction.user_id == user_id)
            .all()
        )

        if not history:
            return []

        # collect user skills
        user_skills = set()
        for h in history:
            if h.skills:
                for s in h.skills:
                    if s:
                        user_skills.add(" ".join(s.split()))

        # load all degrees
        all_degrees = self.db.query(DegreeProgram).all()

        # courses user already viewed
        viewed_courses = {h.course_name for h in history if h.course_name}

        recommendations = []

        # --- build raw recommendations ---
        for degree in all_degrees:
            degree_course_names = {
                c.lesson_name for c in degree.courses if c.lesson_name
            }

            # skip degrees that user already saw
            if viewed_courses & degree_course_names:
                continue

            degree_skills = []
            for course in degree.courses:
                for sk in course.skills:
                    if sk.skill_name:
                        degree_skills.append(" ".join(sk.skill_name.split()))

            common = user_skills & set(degree_skills)
            score = len(common)

            if score > 0:
                recommendations.append({
                    "degree_name": degree.name or "",
                    "score": score,
                    "matching_skills": list(common),
                    "reason": "Based on your past selections"
                })

        # --- REAL DUPLICATE FIX ---
        unique = {}
        for r in recommendations:
            key = normalize(r["degree_name"])
            if key not in unique:
                unique[key] = r
            else:
                # keep max score and merge skills
                unique[key]["score"] = max(unique[key]["score"], r["score"])
                unique[key]["matching_skills"] = sorted(
                    list(set(unique[key]["matching_skills"]) | set(r["matching_skills"]))
                )

        final_results = list(unique.values())

        return sorted(final_results, key=lambda x: x["score"], reverse=True)[:top_n]


# ---------------------------
# Endpoint για personalized history-based recommendations
# ---------------------------
# ---------------------------
# Endpoint για personalized history-based degree recommendations
# ---------------------------
@router.get("/recommend/degrees/history/{user_id}")
def recommend_personalized_degree_history(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    recommender = DegreeRecommender(db)   # <-- εδώ χρησιμοποιούμε τη νέα κλάση
    results = recommender.recommend_based_on_history(user_id, top_n)
    
    # Επιβεβαίωση ότι κάθε αντικείμενο έχει score
    for r in results:
        if "score" not in r:
            r["score"] = 0.0
    
    return {"user_id": user_id, "recommendations": results}



    # --------------------------------------------
# Unified Degree Recommendation Endpoint
# --------------------------------------------
@router.get("/recommend/degrees/combined/{user_id}/{university_id}")
def recommend_degrees_combined(
    user_id: int,
    university_id: int,
    top_n: int = 10,
    db: Session = Depends(get_db)
):
    """
    Returns both main recommended degrees AND history-based degree recommendations,
    while removing duplicates between the two lists.
    """

    # 1️⃣ Load MAIN recommended degrees (University-based)
    main_results = UniversityRecommender(db).suggest_degrees_with_skills(
        university_id,
        top_n=top_n
    )

    # Normalize structure → convert to simple list of names
    recommended_degrees = []
    for r in main_results:
        recommended_degrees.append({
            "name": r.get("degree_title") or r.get("degree_name") or "",
            "score": r.get("score", 0.0),
            "skills": r.get("skills", []),
            "source": "main"
        })

    # 2️⃣ Load HISTORY-based recommendations
    history_results = DegreeRecommender(db).recommend_based_on_history(
        user_id,
        top_n=top_n
    )

    # Standardize names
    history_degrees = []
    for r in history_results:
        history_degrees.append({
            "name": r.get("degree_name") or "",
            "score": r.get("score", 0.0),
            "skills": r.get("matching_skills", []),
            "source": "history"
        })

    # 3️⃣ Remove duplicates (History must NOT repeat Main)
    def normalize(n: str):
        return " ".join(str(n).split()).lower()

    main_names = {normalize(d["name"]) for d in recommended_degrees}

    filtered_history = [
        d for d in history_degrees
        if normalize(d["name"]) not in main_names
    ]

    # 4️⃣ FINAL RESPONSE
    return {
        "user_id": user_id,
        "university_id": university_id,
        "recommended_degrees": recommended_degrees,
        "history_degrees": filtered_history,
    }

@router.get(
    "/recommend/universities/{univ_id}/profile",
    summary="Get profile of a university (skills, degrees, courses)"
)
def get_university_profile(univ_id: int, db: Session = Depends(get_db)):
    recommender = UniversityRecommender(db)
    profile = recommender.build_university_profile(univ_id)
    if not profile:
        raise HTTPException(status_code=404, detail="No profile found.")
    return profile


@router.get(
    "/recommend/universities/{univ_id}/similar",
    summary="Find similar universities based on skills, degrees, courses"
)
def find_similar_universities(univ_id: int, top_n: int = 5, db: Session = Depends(get_db)):
    recommender = UniversityRecommender(db)
    return recommender.find_similar_universities(univ_id, top_n)


@router.post(
    "/recommend/universities/{univ_id}/programs/{program_id}/courses-existing-degree",
    summary="Recommend new courses for an existing degree."
)
def recommend_courses_for_existing_degree(
    univ_id: int,
    program_id: int,
    request: ExistingDegreeCourseRequest,
    db: Session = Depends(get_db)
):
    recommender = CourseRecommenderV2(db)
    profiles = recommender.build_degree_profiles(univ_id)

    target = None
    for p in profiles:
        if p["program_id"] == program_id:
            target = p
            break

    if not target:
        raise HTTPException(status_code=404, detail="Degree profile not found.")

    all_profiles = []
    for u in recommender.get_all_universities():
        all_profiles.extend(recommender.build_degree_profiles(u.university_id))

    similar = recommender.find_similar_degrees(target, all_profiles, top_n=10)

    result = recommender.suggest_courses_for_degree(
        target_degree=target,
        similar_degrees=similar,
        top_n=request.top_n
    )

    return {"university_id": univ_id, "program_id": program_id, "recommendations": result}

@router.post(
    "/recommend/universities/{univ_id}/courses-new-degree",
    summary="Recommend courses for a new custom degree."
)
def recommend_courses_new_degree(
    univ_id: int,
    request: NewDegreeCourseRequest,
    db: Session = Depends(get_db)
):
    recommender = CourseRecommenderV2(db)

    all_profiles = []
    for u in recommender.get_all_universities():
        all_profiles.extend(recommender.build_degree_profiles(u.university_id))

    synthetic_profile = {
        "university_id": univ_id,
        "program_id": None,
        "degree_title": request.degree_title,
        "degree_type": request.degree_type,
        "skills": request.target_skills,
        "courses": []
    }

    similar = recommender.find_similar_degrees(synthetic_profile, all_profiles, top_n=10)

    result = recommender.suggest_courses_for_new_degree(
        similar_degrees=similar,
        target_skills=request.target_skills,
        top_n=request.top_n
    )

    return {"university_id": univ_id, "degree": request.degree_title, "recommendations": result}
