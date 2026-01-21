from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import distinct, func
from collections import defaultdict

from recommendation_system.backend.database import get_db
from recommendation_system.backend.models import DegreeProgram, University, Course, Skill, CourseSkill
from recommendation_system.backend.schemas import CourseRecommendationsResponse, DegreeProgramOut
from recommendation_system.backend.course_recommender_for_university import CourseRecommender as CourseRecommenderV2
from recommendation_system.backend.degree_recommender_for_university import UniversityRecommender

router = APIRouter()


@router.get("/filters/degree-types", response_model=List[str], summary="Return all unique degree types.")
def get_unique_degree_types(db: Session = Depends(get_db)):
    """
    Retrieve unique degree types (e.g., BSc, MSc, PhD, Other) for filtering purposes.
    """
    results = db.query(distinct(DegreeProgram.degree_type)).all()
    return [r[0] for r in results if r[0]]


@router.get("/filters/countries", response_model=List[str], summary="Return all unique countries of universities.")
def get_unique_countries(db: Session = Depends(get_db)):
    """
    Retrieve unique countries from the University table for filtering purposes.
    """
    results = db.query(distinct(University.country)).order_by(University.country).all()
    return [r[0] for r in results if r[0]]


@router.get("/filters/languages", response_model=List[str], summary="Return all unique languages used in programs/courses.")
def get_unique_languages(db: Session = Depends(get_db)):
    """
    Retrieve unique languages from both DegreeProgram and Course tables for filtering purposes.
    Handles multiple languages in a single field separated by commas.
    """
    # Languages from DegreeProgram
    program_languages = db.query(DegreeProgram.language).filter(DegreeProgram.language.isnot(None)).distinct()
   
    # Languages from Course
    course_languages = db.query(Course.language).filter(Course.language.isnot(None)).distinct()
   
    combined_languages = set()
    # Union of languages from both tables
    for lang_tuple in program_languages.union(course_languages).all():
        if lang_tuple[0]:
            # Split multiple languages in one field
            parts = [part.strip() for part in lang_tuple[0].split(',') if part.strip()]
            combined_languages.update(parts)

    return sorted(list(combined_languages))


@router.get("/universities", summary="Return all universities with basic information.")
def get_all_universities(db: Session = Depends(get_db)):
    """
    Retrieve all universities with their ID, name, and country.
    """
    universities = db.query(University).order_by(University.university_name).all()
    return [
        {"university_id": u.university_id, "university_name": u.university_name, "country": u.country}
        for u in universities
    ]


@router.get("/universities/{univ_id}/degrees", response_model=List[DegreeProgramOut], summary="Return all degree programs of a university.")
def get_degree_programs(univ_id: int, db: Session = Depends(get_db)):
    """
    Retrieve all degree programs offered by a specific university.
    """
    university = db.query(University).filter(University.university_id == univ_id).first()
    if not university:
        raise HTTPException(status_code=404, detail="University not found")
    return university.programs


@router.get("/metrics/{university_id}", summary="Return basic metrics for a university.")
def get_university_metrics(university_id: int, db: Session = Depends(get_db)):
    """
    Return metrics for a university including total degree programs and number of unique recognized skills.
    """
    university = db.query(University).filter_by(university_id=university_id).first()
    if not university:
        raise HTTPException(status_code=404, detail="University not found")

    total_programs = len(university.programs)
   
    # Count unique skills across all courses of the university
    unique_skills_count = (
        db.query(func.count(distinct(CourseSkill.skill_id)))
        .filter(CourseSkill.course_id.in_([c.course_id for c in university.courses]))
        .scalar()
    )

    recognized_skills_final = unique_skills_count if unique_skills_count is not None else 0
   
    return {
        "university_id": university_id,
        "total_programs": total_programs,
        "recognized_skills": recognized_skills_final
    }


@router.get("/skills/grouped-by-categories", response_model=Dict[str, List[Dict[str, Any]]], summary="Return skills grouped by categories.")
def get_grouped_skills_by_categories(db: Session = Depends(get_db)):
    """
    Retrieve all skills grouped by categories defined in CourseSkill.categories.
    - A skill belonging to multiple categories will appear in each category.
    - Duplicate skills within the same category (case-insensitive) are removed.
    - Skills without a category are grouped under 'Other/No Category'.
    """
    try:
        links = db.query(CourseSkill).all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    grouped_skills = defaultdict(list)

    for link in links:
        skill = db.query(Skill).filter(Skill.skill_id == link.skill_id).first()
        if not skill:
            continue

        skill_name = skill.skill_name.strip()

        # Place skill in each assigned category
        if link.categories and isinstance(link.categories, list) and len(link.categories) > 0:
            for cat in link.categories:
                if not any(s["name"].lower() == skill_name.lower() for s in grouped_skills[cat]):
                    grouped_skills[cat].append({
                        "id": skill.skill_id,
                        "name": skill_name
                    })
        else:
            cat = "Other/No Category"
            if not any(s["name"].lower() == skill_name.lower() for s in grouped_skills[cat]):
                grouped_skills[cat].append({
                    "id": skill.skill_id,
                    "name": skill_name
                })

    # Sort skills alphabetically within each category
    for cat in grouped_skills:
        grouped_skills[cat] = sorted(grouped_skills[cat], key=lambda x: x["name"].lower())

    # Sort categories alphabetically
    grouped_sorted = dict(sorted(grouped_skills.items(), key=lambda x: x[0].lower()))

    return grouped_sorted
