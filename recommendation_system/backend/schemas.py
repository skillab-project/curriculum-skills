# schemas.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# =======================================================
# Skills
# =======================================================
class SkillOut(BaseModel):
    skill_id: int
    skill_name: str

    class Config:
        orm_mode = True


# =======================================================
# Universities
# =======================================================
class UniversityBase(BaseModel):
    university_id: int
    university_name: str
    country: str

    class Config:
        orm_mode = True


# =======================================================
# Degree Programs
# =======================================================
class DegreeProgramSearch(BaseModel):
    program_id: int
    degree_type: str
    degree_titles: Optional[Dict[str, Any]]
    language: Optional[str]
    university: UniversityBase

    class Config:
        orm_mode = True


class DegreeProgramOut(BaseModel):
    program_id: int
    degree_type: str
    degree_titles: Optional[List[str]] = None
    language: Optional[str] = None
    duration_semesters: Optional[str] = None
    total_ects: Optional[str] = None

    class Config:
        orm_mode = True


# =======================================================
# Courses
# =======================================================
class CourseSearch(BaseModel):
    course_id: int
    lesson_name: str
    language: Optional[str]
    semester_label: Optional[str]
    description: Optional[str]
    university: UniversityBase

    class Config:
        orm_mode = True


class RecommendedCourse(BaseModel):
    course_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    description: str = ""
    objectives: str = ""
    learning_outcomes: str = ""
    course_content: str = ""
    new_skills: List[str] = []
    compatible_skills: List[str] = []


class CourseRecommendationsResponse(BaseModel):
    university_id: int
    program_id: int
    degree: str
    recommendations: List[RecommendedCourse]


# =======================================================
# Search Results
# =======================================================
class SearchResult(BaseModel):
    degree_programs: List[DegreeProgramSearch]
    courses: List[CourseSearch]


# =======================================================
# Filters
# =======================================================
class FiltersResponse(BaseModel):
    degree_types: List[str] = []
    countries: List[str] = []
    languages: List[str] = []


# =======================================================
# Personalized Recommendations
# =======================================================
class UserPreferences(BaseModel):
    target_skills: List[str]
    language: Optional[str] = None
    country: Optional[str] = None
    degree_type: Optional[str] = None
    top_n: int = 10


# =======================================================
# Electives
# =======================================================
class ElectiveRecommendationRequest(BaseModel):
    program_id: int
    target_skills: List[str]
    top_n: int = 10


class ElectiveCourseOut(BaseModel):
    course_name: str
    score: float
    skills: List[str] = []
    matching_skills: List[str] = []
    missing_skills: List[str] = []
    reason: Optional[str] = ""


class ElectiveRecommendationResponse(BaseModel):
    success: bool
    recommended_electives: List[ElectiveCourseOut] = []
    message: Optional[str] = ""
    meta: Optional[Dict[str, Any]] = {}

# =======================================================
# Enhanced Electives (CourseRecommenderV4)
# =======================================================
class ElectiveEnhancedRequest(BaseModel):
    univ_id: int
    program_id: int
    target_skills: List[str]
    top_n: int = 10
    min_overlap_ratio: float = 0.1
    semester: Optional[int] = None


class ElectiveEnhancedCourseOut(BaseModel):
    course_id: Optional[int] = None
    lesson_name: Optional[str] = None
    university: Optional[str] = None
    final_score: float = Field(..., ge=0.0, le=1.0)
    tfidf_score: float = Field(..., ge=0.0)
    overlap_ratio: float = Field(..., ge=0.0, le=1.0)
    matching_skills: List[str] = []
    missing_skills: List[str] = []
    skills: List[str] = []
    reason: Optional[str] = ""
    website: Optional[str] = None


class ElectiveEnhancedResponse(BaseModel):
    recommended_electives: List[ElectiveEnhancedCourseOut] = []
    meta: Dict[str, Any] = {}
    message: Optional[str] = None


# =======================================================
# University-level recommender (UniversityRecommender)
# =======================================================
class UniversityProfileOut(BaseModel):
    university_id: int
    skills: List[str]
    courses: List[str]
    degrees: List[str]


class SimilarUniversityOut(BaseModel):
    university_id: int
    name: str
    country: str
    similarity_score: float


class DegreeSkillOut(BaseModel):
    skill_name: str
    skill_score: float


class DegreeWithSkillsOut(BaseModel):
    degree: str
    degree_type: str
    score: float
    top_skills: List[DegreeSkillOut]
    metrics: Dict[str, Any]


class DegreesWithSkillsResponse(BaseModel):
    university_id: int
    recommendations: List[DegreeWithSkillsOut]


# =======================================================
# Course recommender for universities
#    (course_recommender_for_university.CourseRecommender)
# =======================================================
class ExistingDegreeCourseRequest(BaseModel):
    univ_id: int
    program_id: int
    degree_title: str
    top_n: int = 10


class NewDegreeCourseRequest(BaseModel):
    univ_id: int
    degree_type: str
    degree_title: str
    target_skills: List[str] = []
    top_n: int = 10