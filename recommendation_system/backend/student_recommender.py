"""
Two recommendation engines:
1) CourseRecommender  - personalized degree & standalone course recommendations
2) CourseRecommenderV4 - elective (course option) recommendations for a specific degree program

Both engines use TF-IDF + cosine similarity over skills/courses text and lightweight
heuristics for scoring and filtering. The original logic is preserved.
"""

from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging
import re

from recommendation_system.backend.models import DegreeProgram, University, Course, Skill, CourseSkill

# Configure logger for module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Helper: skill normalization
# -----------------------------------------------------------------------------
def _normalize_skill(s: str) -> str:
    """
    Normalize a skill token to a compact canonical form:
      - lowercase
      - strip whitespace
      - remove uncommon characters but preserve digits, Greek letters,
        and a few common symbols (+ - . #)
    Returns empty string for falsy input.
    """
    if not s:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\u0370-\u03FF\u1F00-\u1FFF\s\-\+\.#]", "", s)
    return s


# =============================================================================
# 1) CourseRecommender
#    - recommend_personalized: returns degree programs and unlinked courses
# =============================================================================
class CourseRecommender:
    """
    Recommender for degree programs and standalone (unlinked) courses.

    - recommend_personalized: given user target skills (and optional filters),
      returns a ranked list of degree programs and standalone courses that
      best match the skills using TF-IDF + small filter-based bonuses.
    """

    def __init__(self, db: Session):
        """
        Initialize with a SQLAlchemy session (db).
        """
        self.db = db

    def recommend_personalized(
        self,
        target_skills: List[str],
        language: Optional[str] = None,
        country: Optional[str] = None,
        degree_type: Optional[str] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Return personalized recommendations for degree programs and standalone courses.

        Parameters
        ----------
        target_skills : List[str]
            List of user's target skills (strings).
        language : Optional[str]
            Optional language filter (case-insensitive substring match).
        country : Optional[str]
            Optional university country filter (case-insensitive substring match).
        degree_type : Optional[str]
            Optional degree type filter (case-insensitive substring match).
        top_n : int
            Maximum number of items to return per list.

        Returns
        -------
        dict
            {
                "recommended_programs": [...],
                "recommended_unlinked_courses": [...],
                "skills_by_category": {...}
            }
            or {"message": "..."} when no data is available or input invalid.
        """

        # ---------------------------
        # Normalize and validate input
        # ---------------------------
        target_skills_norm = [s.lower().strip() for s in target_skills if s]
        user_text = " ".join(target_skills_norm)
        if not user_text:
            return {"message": "Please provide at least one skill."}

        # ---------------------------
        # Query degree programs with optional filters
        # ---------------------------
        query = self.db.query(DegreeProgram).join(University)
        if language:
            query = query.filter(DegreeProgram.language.ilike(f"%{language}%"))
        if country:
            query = query.filter(University.country.ilike(f"%{country}%"))
        if degree_type:
            query = query.filter(DegreeProgram.degree_type.ilike(f"%{degree_type}%"))

        programs = query.all()
        if not programs:
            return {"message": "No matching programs were found."}

        # ---------------------------
        # Build program skill texts for TF-IDF
        # ---------------------------
        program_texts: List[str] = []
        program_objs: List[DegreeProgram] = []
        for prog in programs:
            skills = self._get_program_skills(prog.program_id)
            if not skills:
                continue
            program_texts.append(" ".join([s.lower() for s in skills]))
            program_objs.append(prog)

        if not program_texts:
            return {"message": "No skills are recorded for any matching program."}

        # ---------------------------
        # Compute TF-IDF similarity between user skills and programs
        # ---------------------------
        vectorizer = TfidfVectorizer()
        try:
            vectors = vectorizer.fit_transform([user_text] + program_texts)
            sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
        except Exception as e:
            # If TF-IDF fails, fall back to zero similarities
            logger.exception("TF-IDF vectorization error for degree programs: %s", e)
            sims = [0.0] * len(program_objs)

        # ---------------------------

        # Score and sort programs
        # ---------------------------
        scored_programs: List[Dict[str, Any]] = []

        for i, prog in enumerate(program_objs):
            score = float(sims[i]) if i < len(sims) else 0.0

            # Small bonuses when filters align exactly
            if language and prog.language and prog.language.lower() == language.lower():
                score += 0.05
            if degree_type and prog.degree_type and prog.degree_type.lower() == degree_type.lower():
                score += 0.05
            if country and prog.university.country and prog.university.country.lower() == country.lower():
                score += 0.05

            degree_name = "N/A"
            if prog.degree_titles:
                if isinstance(prog.degree_titles, dict):
                    degree_name = (
                        prog.degree_titles.get("el")
                        or prog.degree_titles.get("en")
                        or str(prog.degree_titles)
                    )
                else:
                    degree_name = str(prog.degree_titles)

            prog_skills = self._get_program_skills(prog.program_id)

            scored_programs.append({
                "program_id": prog.program_id,
                "degree_name": degree_name,
                "university": getattr(prog.university, "university_name", None),
                "language": prog.language,
                "country": getattr(prog.university, "country", None),
                "degree_type": prog.degree_type,
                "score": round(score, 3),
                "skills": prog_skills
            })

        scored_programs = sorted(scored_programs, key=lambda x: x["score"], reverse=True)[:top_n]




        # ---------------------------
        # Recommend standalone (unlinked) courses
        # ---------------------------
        unlinked_courses = self.db.query(Course).filter(Course.program_id == None).all()
        course_texts: List[str] = []
        course_objs: List[Course] = []
        for c in unlinked_courses:
            skills = self._get_course_skills(c.course_id)
            if skills:
                course_texts.append(" ".join([s.lower() for s in skills]))
                course_objs.append(c)

        unlinked_recs: List[Dict[str, Any]] = []
        if course_texts:
            try:
                vectors = vectorizer.fit_transform([user_text] + course_texts)
                sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
                for i, c in enumerate(course_objs):
                    unlinked_recs.append({
                        "course_id": c.course_id,
                        "lesson_name": c.lesson_name,
                        "university": getattr(c.university, "university_name", None),
                        "score": round(float(sims[i]) if i < len(sims) else 0.0, 3)
                    })
                unlinked_recs = sorted(unlinked_recs, key=lambda x: x["score"], reverse=True)[:top_n]
            except Exception as e:
                logger.exception("TF-IDF error for unlinked courses: %s", e)
                unlinked_recs = []

        # ---------------------------
        # Group user skills by category for presentation
        # ---------------------------
        grouped_skills = self._group_skills_by_category(target_skills_norm)

        return {
            "recommended_programs": scored_programs,
            "recommended_unlinked_courses": unlinked_recs,
            "skills_by_category": grouped_skills
        }

    # -------------------------------------------------------------------------
    # Helper: extract all skill names from a program's courses
    # -------------------------------------------------------------------------
    def _get_program_skills(self, program_id: int) -> List[str]:
        """
        Return a list of unique skill names referenced by the courses of the given program.
        Returns an empty list when the program isn't found or there are no skills.
        """
        program = self.db.query(DegreeProgram).filter_by(program_id=program_id).first()
        if not program:
            return []

        skills_set: Set[str] = set()
        for course in getattr(program, "courses", []) or []:
            for cs in getattr(course, "skills", []) or []:
                try:
                    skill_name = getattr(cs.skill, "skill_name", None)
                    if skill_name:
                        skills_set.add(skill_name)
                except Exception:
                    # Defensive: skip malformed relationships
                    continue
        return list(skills_set)

    # -------------------------------------------------------------------------
    # Helper: extract skill names attached to a course (CourseSkill relation)
    # -------------------------------------------------------------------------
    def _get_course_skills(self, course_id: int) -> List[str]:
        """
        Return a list of skill names for a given course_id by inspecting CourseSkill rows.
        """
        skills: List[str] = []
        try:
            entries = self.db.query(CourseSkill).filter_by(course_id=course_id).all()
            for cs in entries:
                if hasattr(cs, "skill") and getattr(cs.skill, "skill_name", None):
                    skills.append(cs.skill.skill_name)
        except Exception:
            # Defensive: return empty on DB errors
            pass
        return skills

    # -------------------------------------------------------------------------
    # Helper: group skills by category (presentation)
    # -------------------------------------------------------------------------
    def _group_skills_by_category(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Group skill names by their stored category. If a skill's category is not present,
        group under 'Other'. Matching is attempted using case-insensitive contains.
        """
        grouped: Dict[str, List[str]] = {}
        for s in skills:
            try:
                # Find a skill row matching the token (case-insensitive contains)
                skill_obj = self.db.query(Skill).filter(Skill.skill_name.ilike(f"%{s}%")).first()
                category = "Other"
                if skill_obj:
                    # Skill.categories may be a dict or another structure; attempt sensible extraction
                    if isinstance(skill_obj.categories, dict):
                        category = skill_obj.categories.get("preferredLabel", "Other")
                    elif skill_obj.categories:
                        category = str(skill_obj.categories)
                grouped.setdefault(category, []).append(s)
            except Exception:
                grouped.setdefault("Other", []).append(s)

        # Deduplicate and sort each category list
        for k in list(grouped.keys()):
            grouped[k] = sorted(set(grouped[k]), key=lambda x: x.lower())

        # Return categories ordered alphabetically
        return dict(sorted(grouped.items(), key=lambda x: x[0].lower()))


# =============================================================================
# 2) CourseRecommenderV4
#    - recommend_electives_for_degree_enhanced: return elective course suggestions
# =============================================================================
class CourseRecommenderV4:
    """
    Elective recommender for a given degree program at a specific university.

    Scoring combines:
      - TF-IDF semantic similarity between user skills and course skill text
      - Overlap ratio between user skills and course skills

    Final score = tfidf_weight * tfidf_score + overlap_weight * overlap_ratio
    """

    def __init__(self, db: Session, tfidf_weight: float = 0.6, overlap_weight: float = 0.4):
        """
        Initialize with a SQLAlchemy session and optional weights for combining signals.
        Weights are normalized so that their sum equals 1.0 when possible.
        """
        self.db = db
        total = (tfidf_weight or 0.0) + (overlap_weight or 0.0)
        if total <= 0:
            # Defaults if invalid weights provided
            self.tfidf_weight = 0.6
            self.overlap_weight = 0.4
        else:
            self.tfidf_weight = tfidf_weight / total
            self.overlap_weight = overlap_weight / total

    def recommend_electives_for_degree_enhanced(
        self,
        univ_id: int,
        program_id: int,
        target_skills: List[str],
        top_n: int = 5,
        min_overlap_ratio: float = 0.1,
        semester: Optional[int] = None 
    ) -> Dict[str, Any]:
        """
        Recommend elective courses for a program.

        Parameters
        ----------
        univ_id : int
            University id where the program is offered.
        program_id : int
            Program id for which to recommend electives.
        target_skills : List[str]
            User-provided skills used as the target profile.
        top_n : int
            Maximum number of electives to return.
        min_overlap_ratio : float
            Minimum overlap ratio threshold to consider a course (0..1).

        Returns
        -------
        dict
            {
                "recommended_electives": [...],
                "meta": {...}
            }
            or {"message": "..."} on error or no results.
        """

        # ---------------------------
        # Normalize user skills
        # ---------------------------
        user_skills_norm = [_normalize_skill(s) for s in (target_skills or []) if _normalize_skill(s)]
        user_skills_set: Set[str] = set(user_skills_norm)
        if not user_skills_set:
            return {"message": "Please provide at least one valid skill."}
        user_text = " ".join(sorted(user_skills_set))

        # ---------------------------
        # Locate the degree program
        # ---------------------------
        program = self.db.query(DegreeProgram).filter_by(program_id=program_id, university_id=univ_id).first()
        if not program:
            return {"message": "Degree program not found for this university."}

        # ---------------------------
        # Identify elective courses from the program
        # ---------------------------
        elective_courses: List[Course] = []
        for c in getattr(program, "courses", []) or []:
            mand_opt = getattr(c, "mand_opt_list", None)
            is_optional = False
            if isinstance(mand_opt, str) and "optional" in mand_opt.lower():
                is_optional = True
            elif isinstance(mand_opt, (list, tuple, set)):
                is_optional = any("optional" in str(v).lower() for v in mand_opt)
            if not is_optional:
                continue
            if semester is not None:
                c_sem = getattr(c, "semester_number", None)
                if str(c_sem) != str(semester):
                    continue
        
            elective_courses.append(c)
        # ---------------------------
        # Extract and normalize skills for each elective
        # ---------------------------
        course_objs: List[Course] = []
        course_skill_sets: List[Set[str]] = []
        course_skill_texts: List[str] = []
        course_raw_skills: List[List[str]] = []

        for c in elective_courses:
            raw_skill_names: List[str] = []

            # support two possible relationship shapes: courseskills or skills
            if hasattr(c, "courseskills"):
                for cs in getattr(c, "courseskills") or []:
                    if hasattr(cs, "skill") and getattr(cs.skill, "skill_name", None):
                        raw_skill_names.append(str(cs.skill.skill_name))
            elif hasattr(c, "skills"):
                for cs in getattr(c, "skills") or []:
                    if hasattr(cs, "skill") and getattr(cs.skill, "skill_name", None):
                        raw_skill_names.append(str(cs.skill.skill_name))
                    elif getattr(cs, "skill_name", None):
                        # Some models store skill_name directly on the relation
                        raw_skill_names.append(str(cs.skill_name))

            if not raw_skill_names:
                continue

            normalized = [_normalize_skill(s) for s in raw_skill_names if _normalize_skill(s)]
            if not normalized:
                continue

            course_objs.append(c)
            course_raw_skills.append(sorted(set(raw_skill_names)))
            course_skill_sets.append(set(normalized))
            course_skill_texts.append(" ".join(normalized))

        if not course_objs:
            return {"message": "No skills found on the elective courses of this program."}

        # ---------------------------
        # TF-IDF similarity between user skills and course skill text
        # ---------------------------
        try:
            vectorizer = TfidfVectorizer()
            docs = [user_text] + course_skill_texts
            vectors = vectorizer.fit_transform(docs)
            tfidf_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten().tolist()
        except Exception as e:
            logger.exception("TF-IDF vectorization failed for electives: %s", e)
            tfidf_scores = [0.0] * len(course_objs)

        # ---------------------------
        # Compute final scores and reasons
        # ---------------------------
        scored: List[Dict[str, Any]] = []
        for i, c in enumerate(course_objs):
            course_set = course_skill_sets[i]
            matched = sorted(list(user_skills_set & course_set))
            missing = sorted(list(course_set - user_skills_set))
            union_set = course_set | user_skills_set
            overlap_ratio = (len(matched) / len(union_set)) if union_set else 0.0
            tfidf_score = float(tfidf_scores[i]) if i < len(tfidf_scores) else 0.0

            # filter out ultra-low relevance candidates
            if overlap_ratio < min_overlap_ratio and tfidf_score < 0.01:
                continue

            # combine signals
            final_score = (self.tfidf_weight * tfidf_score) + (self.overlap_weight * overlap_ratio)
            final_score = max(0.0, min(1.0, final_score))

            # human-friendly reason text describing why the course scored
            reason_parts: List[str] = []
            if matched:
                reason_parts.append(f"Common skills: {len(matched)} ({', '.join(matched[:5])})")
            if tfidf_score > 0:
                reason_parts.append(f"Semantic similarity: {round(tfidf_score, 3)}")
            reason_text = "; ".join(reason_parts) if reason_parts else "Low relevance."

            scored.append({
                "course_id": getattr(c, "course_id", None),
                "lesson_name": getattr(c, "lesson_name", None),
                "university": getattr(getattr(c, "university", None), "university_name", None),
                "final_score": round(final_score, 3),
                "tfidf_score": round(tfidf_score, 3),
                "overlap_ratio": round(overlap_ratio, 3),
                "matching_skills": matched,
                "missing_skills": missing,
                "reason": reason_text,
                "skills": course_raw_skills[i],
                "website": getattr(c, "website", None),

            })

        # ---------------------------
        # Filter, sort and limit to top_n
        # ---------------------------
        scored_filtered = [c for c in scored if c["final_score"] >= 0.1]
        scored_sorted = sorted(scored_filtered, key=lambda x: x["final_score"], reverse=True)[:max(1, top_n)]

        return {
            "recommended_electives": scored_sorted,
            "meta": {
                "n_electives_considered": len(course_objs),
                "n_returned": len(scored_sorted),
                "tfidf_weight": self.tfidf_weight,
                "overlap_weight": self.overlap_weight,
            }
        }
