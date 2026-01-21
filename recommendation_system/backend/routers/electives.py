from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from recommendation_system.backend.database import get_db
from recommendation_system.backend.models import DegreeProgram, Skill
from recommendation_system.backend.student_recommender import CourseRecommenderV4
from recommendation_system.backend.models import Course
from fastapi import APIRouter, Depends, HTTPException, Query

from recommendation_system.backend.schemas import (
    ElectiveRecommendationRequest,
    ElectiveRecommendationResponse,
    ElectiveEnhancedRequest,
    ElectiveEnhancedResponse,
    ElectiveEnhancedCourseOut,
)

router = APIRouter()


@router.post("/universities/{univ_id}/degrees/electives")
def recommend_electives(
    univ_id: int,
    payload: ElectiveRecommendationRequest,
    semester: str = Query(None, description="Semester number to filter elective courses"),
    min_overlap_ratio: float = 0.0,
    db: Session = Depends(get_db)
):
    """
    Recommend elective courses for a specific degree program at a given university.

    Parameters:
    - univ_id (int): University ID.
    - payload (ElectiveRecommendationRequest): Request payload containing program_id, target skills, and top_n courses.
    - semester (str, optional): Only recommend electives from this semester if provided.
    - min_overlap_ratio (float): Minimum ratio of matching skills required for recommendation.
    - db (Session): SQLAlchemy database session (injected by FastAPI dependency).

    Returns:
    - JSON with success status, recommended electives with scores and matching skills, and meta information.
    """
    try:
        recommender = CourseRecommenderV4(db)

        # Fetch recommended electives using the recommender system
        result = recommender.recommend_electives_for_degree_enhanced(
    univ_id=univ_id,
    program_id=payload.program_id,
    target_skills=payload.target_skills,
    top_n=payload.top_n,
    min_overlap_ratio=min_overlap_ratio,
    semester=semester
)


        # Handle empty or error response from recommender
        if not result or "message" in result:
            return {
                "success": False,
                "message": result.get("message", "No electives found for this program."),
                "recommended_electives": []
            }

        # Format recommended courses with scores and skills
        recommended_courses = [
{
"course_name": item.get("lesson_name", "Unknown"),
"score": float(item.get("final_score", 0.0)),
"skills": item.get("skills", []),
"matching_skills": item.get("matching_skills", []),
"missing_skills": item.get("missing_skills", []),
"reason": item.get("reason", ""),
"website": item.get("website", "")  # <-- Προσθήκη του website
}
for item in result.get("recommended_electives", [])
]


        return {
            "success": True,
            "recommended_electives": recommended_courses,
            "meta": result.get("meta", {})
        }

    except Exception as e:
        # Log the exception and return 500 Internal Server Error
        print(f"Error in recommend_electives endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")




@router.get(
    "/universities/{univ_id}/degrees/{program_id}/elective-skills",
    summary="Get skills from elective courses of a specific degree program for a specific semester"
)
def get_elective_skills_for_program(
    univ_id: int,
    program_id: int,
    semester: str = Query(..., description="Semester number as string (required)"),
    db: Session = Depends(get_db)
):
    try:
        program = db.query(DegreeProgram).filter(
            DegreeProgram.program_id == program_id,
            DegreeProgram.university_id == univ_id
        ).first()

        if not program:
            raise HTTPException(status_code=404, detail="Degree program not found for this university.")

        elective_courses = []
        for course in getattr(program, "courses", []) or []:
            if course.program_id != program.program_id:
                continue

            mand_opt_list = getattr(course, "mand_opt_list", [])
            if not isinstance(mand_opt_list, (list, tuple, set)):
                mand_opt_list = [str(mand_opt_list)]
            is_optional = any(str(v).lower() == "optional" for v in mand_opt_list)

            if is_optional and str(getattr(course, "semester_number", "")) == str(semester):
                elective_courses.append(course)

        skill_ids = {
            cs.skill_id
            for course in elective_courses
            for cs in getattr(course, "skills", [])
            if hasattr(cs, "skill_id") and cs.skill_id
        }

        if not skill_ids:
            return {"skills": []}

        skills = db.query(Skill).filter(Skill.skill_id.in_(skill_ids)).order_by(Skill.skill_name.asc()).all()
        skill_list = [{"skill_id": s.skill_id, "skill_name": s.skill_name} for s in skills]

        return {"skills": skill_list}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_elective_skills_for_program: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")





@router.get("/universities/{univ_id}/degrees/{program_id}/semesters")
def get_program_semesters(univ_id: int, program_id: int, db: Session = Depends(get_db)):
    """
    Επιστρέφει μια λίστα από 1 έως duration_semesters ενός προγράμματος,
    για χρήση σε dropdown.
    """
    program = (
        db.query(DegreeProgram)
        .filter(
            DegreeProgram.program_id == program_id,
            DegreeProgram.university_id == univ_id
        )
        .first()
    )

    if not program or not program.duration_semesters:
        return {"semesters": []}

    try:
        num_semesters = int(program.duration_semesters)
        if num_semesters < 1:
            return {"semesters": []}
    except ValueError:
        return {"semesters": []}

    semesters = list(range(1, num_semesters + 1))
    return {"semesters": semesters}


@router.post(
    "/universities/{univ_id}/degrees/electives/enhanced",
    summary="Enhanced elective recommendation with scoring, overlap, and explanations"
)
def recommend_electives_enhanced(
    univ_id: int,
    payload: ElectiveRecommendationRequest,
    semester: str = Query(None),
    min_overlap_ratio: float = 0.1,
    db: Session = Depends(get_db)
):
    recommender = CourseRecommenderV4(db)
    result = recommender.recommend_electives_for_degree_enhanced(
        univ_id=univ_id,
        program_id=payload.program_id,
        target_skills=payload.target_skills,
        top_n=payload.top_n,
        min_overlap_ratio=min_overlap_ratio,
        semester=semester
    )
    return result
