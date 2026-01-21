from collections import defaultdict
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from recommendation_system.backend.models import University, Course
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseRecommender:
    """
    Recommender responsible for suggesting courses and degrees per university.

    This class builds degree profiles, finds similar degrees using TF-IDF +
    cosine similarity on combined skills/courses text, and generates ranked
    course recommendations for existing or new degrees.
    """

    def __init__(self, db: Session):
        """
        Initialize with a SQLAlchemy session.

        Parameters
        ----------
        db : Session
            SQLAlchemy session used to query models (University, Course, etc.).
        """
        self.db = db

    # ==========================================================
    # Helper methods
    # ==========================================================
    def get_university(self, univ_id: int) -> Optional[University]:
        """
        Return a University object by id or None if not found.
        """
        return self.db.query(University).filter_by(university_id=univ_id).first()

    def get_all_universities(self) -> List[University]:
        """
        Return a list of all University objects.
        """
        return self.db.query(University).all()

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalizes a name string by removing characters that are not letters,
        numbers or whitespace and converting to uppercase.

        This is used for safe name comparisons (degree titles).
        """
        if not name:
            return ""
        # allow Latin, digits, whitespace and Greek Unicode blocks
        cleaned_name = re.sub(r'[^a-zA-Z0-9\s\u0370-\u03FF\u1F00-\u1FFF]', '', name)
        return cleaned_name.strip().upper()

    @staticmethod
    def _parse_titles(raw_titles):
        """
        Parse a program's raw titles field into a list of title strings.

        The raw_titles value can be:
          - a JSON-encoded list (string)
          - a plain string
          - a Python list
          - other values

        Return a cleaned list of titles (non-empty).
        """
        if not raw_titles:
            return []

        titles = []
        try:
            if isinstance(raw_titles, str):
                # Try to parse JSON-encoded string first
                try:
                    parsed = json.loads(raw_titles)
                    titles = parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    # Fallback: treat the string as a single title
                    titles = [raw_titles]
            elif isinstance(raw_titles, list):
                titles = raw_titles
            else:
                titles = [str(raw_titles)]
        except Exception:
            titles = []

        # Clean titles: keep letters, digits, space, dash, ampersand and Greek characters
        return [
            re.sub(r"[^a-zA-Z0-9 \-&\u0370-\u03FF\u1F00-\u1FFF]", "", str(t)).strip()
            for t in titles if t
        ]

    def get_course_details_by_name(self, course_name: str, target_univ_id: Optional[int] = None) -> Dict[str, str]:
        """
        Retrieve textual details for a course (description, objectives, outcomes, content).

        If target_univ_id is provided, restrict the lookup to that university.
        Returns a dict of strings; returns English-friendly default messages when not found.
        """
        query = self.db.query(Course).filter(Course.lesson_name == course_name)
        if target_univ_id is not None:
            query = query.filter(Course.university_id == target_univ_id)
        course = query.first()

        if course:
            return {
                "description": (getattr(course, "description", "") or "No description available.").strip(),
                "objectives": (getattr(course, "objectives", "") or "No objectives available.").strip(),
                "learning_outcomes": (getattr(course, "learning_outcomes", "") or "No learning outcomes available.").strip(),
                "course_content": (getattr(course, "course_content", "") or "No course content available.").strip(),
            }

        # Defaults when course not found
        return {
            "description": "No description available.",
            "objectives": "No objectives available.",
            "learning_outcomes": "No learning outcomes available.",
            "course_content": "No course content available.",
        }

    # ==========================================================
    # 1) Build degree profiles
    # ==========================================================
    def build_degree_profiles(self, univ_id: int) -> List[Dict[str, Any]]:
        """
        Build degree profiles for every program in a university.

        A profile contains:
            - university_id
            - program_id
            - degree_title
            - degree_type
            - skills (sorted list of skill names)
            - courses (sorted list of course names)

        This function reads the relationships from the ORM objects.
        """
        profiles: List[Dict[str, Any]] = []
        university = self.get_university(univ_id)
        if not university or not getattr(university, "programs", []):
            return []

        for program in university.programs:
            program_id = getattr(program, "program_id", None)
            degree_type = (getattr(program, "degree_type", "") or "").strip()
            # degree_titles may be stored in various formats; parse robustly
            titles = self._parse_titles(getattr(program, "degree_titles", []))
            program_courses = getattr(program, "courses", [])
            # Collect unique course names for the program
            courses = sorted({
                (c.lesson_name or "").strip()
                for c in program_courses
                if getattr(c, "lesson_name", None)
            })

            # Collect unique skill names referenced by the program's courses
            skills = set()
            for course in program_courses:
                for cs in getattr(course, "skills", []):
                    if getattr(cs, "skill", None):
                        skill_name = (cs.skill.skill_name or "").strip()
                        if skill_name:
                            skills.add(skill_name)
            skills = sorted(list(skills))

            # Create a profile entry per declared degree title
            for title in titles:
                if not title:
                    continue
                profiles.append({
                    "university_id": univ_id,
                    "program_id": program_id,
                    "degree_title": title,
                    "degree_type": degree_type,
                    "skills": skills,
                    "courses": courses,
                })
        return profiles

    # ==========================================================
    # 2) Find similar degrees
    # ==========================================================
    def find_similar_degrees(
        self,
        target_profile: Dict[str, Any],
        all_profiles: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the top-N most similar degrees to `target_profile` among `all_profiles`.

        Similarity approach:
          - Concatenate skills + courses for the target and candidate profiles into text documents.
          - Use TF-IDF vectorization and cosine similarity between the target doc and each candidate doc.
          - Restrict candidates to the same degree_type and to different universities.

        Returns a list of candidate profile dicts ordered by similarity descending.
        """
        if not target_profile or not all_profiles:
            return []

        degree_type = (target_profile.get("degree_type") or "").strip()

        # Build a single text for target: skills + courses
        target_text = " ".join(
            (target_profile.get("skills") or []) +
            (target_profile.get("courses") or [])
        ).strip()

        # Filter raw candidates by degree_type and not from the same university
        raw_candidates = [
            p for p in all_profiles
            if (p.get("degree_type") or "").strip() == degree_type
            and p.get("university_id") != target_profile.get("university_id")
        ]

        cand_objs, cand_texts = [], []
        for p in raw_candidates:
            text = " ".join((p.get("skills") or []) + (p.get("courses") or [])).strip()
            if text:
                cand_objs.append(p)
                cand_texts.append(text)

        # If there is no textual information, we cannot compute TF-IDF similarities
        if not target_text or not cand_texts:
            return []

        # TF-IDF + cosine similarity: compute similarity of target vs each candidate
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([target_text] + cand_texts)
        sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Rank candidates by similarity score (descending) and return top_n objects
        ranked = sorted(zip(cand_objs, sims), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:top_n]]

    # ==========================================================
    # 3) Suggest courses for an existing degree
    # ==========================================================
    def suggest_courses_for_degree(
        self,
        target_degree: Dict[str, Any],
        similar_degrees: List[Dict[str, Any]],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Suggest courses for an existing target degree using similar degrees.

        Process:
          - Aggregate courses and skills from similar degrees.
          - Exclude courses already present in the target degree.
          - For each candidate course, collect course-specific skills (from DB) and
            combine them with the degree-level skills from similar degrees.
          - Compute a composite score combining:
              * frequency of occurrence across similar degrees (freq_score)
              * compatibility with target skills (compat_score)
              * novelty (new_skill_score)
              * TF-IDF novelty (novelty_score)
            A small compatibility factor is applied when compatibility is low to demote unsuitable matches.
          - Return top_n course dicts with details and computed scores.
        """
        if not target_degree or not similar_degrees:
            return [{"info": "No similar degrees available for recommendation."}]

        target_skills = set(target_degree.get("skills", []))
        target_courses = set(target_degree.get("courses", []))

        # Aggregation structures
        course_freq = defaultdict(int)              # how many similar degrees include this course
        course_skills = defaultdict(set)            # combined skills associated with the course
        course_specific_skills_map = dict()         # skills that are specific to the course (from DB)

        # Collect candidate courses and skills from similar degrees
        for deg in similar_degrees:
            deg_skills = set(deg.get("skills", []))
            for cname in deg.get("courses", []):
                # skip empty names and courses that already exist in target
                if not cname or cname in target_courses:
                    continue

                db_course = self.db.query(Course).filter(Course.lesson_name == cname).first()
                course_specific_skills = set()
                if db_course:
                    # Extract explicit skills attached to this course (ORM relation)
                    for cs in getattr(db_course, "skills", []):
                        if getattr(cs, "skill", None):
                            skill_name = (cs.skill.skill_name or "").strip()
                            if skill_name:
                                course_specific_skills.add(skill_name)

                # If we found any skills (from degree-level or course-level), register the course
                if course_specific_skills or deg_skills:
                    total_skills = deg_skills.union(course_specific_skills)
                    course_freq[cname] += 1
                    course_skills[cname].update(total_skills)
                    course_specific_skills_map[cname] = course_specific_skills

        # No candidate courses found
        if not course_freq:
            logger.info("No new courses found for recommendation.")
            return [{"info": "No new courses found for recommendation."}]

        # Prepare text documents for TF-IDF similarity: one doc per candidate course (skills text)
        courses_list = list(course_freq.keys())
        course_docs = [" ".join(course_skills[c]) for c in courses_list]
        target_doc = " ".join(target_skills)
        docs = course_docs + [target_doc]

        # If there is no textual skill information, bail out
        if all(not doc.strip() for doc in docs):
            logger.info("Recommendation failed due to lack of skill text for comparison.")
            return [{"info": "Recommendation failed due to lack of skill text for comparison."}]

        # Compute TF-IDF vectors and similarities between target and each candidate
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(docs)
        # similarity of last doc (target) to each course doc
        sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        max_freq = max(course_freq.values()) if course_freq else 1
        target_univ_id = target_degree["university_id"]

        results = []
        for i, cname in enumerate(courses_list):
            skills = course_skills[cname]
            freq_score = course_freq[cname] / max_freq                       # normalized frequency
            new_skills = course_specific_skills_map.get(cname, set()) - target_skills
            compat_skills = skills & target_skills

            # Compatibility: Jaccard-like measure between course skills and target skills
            intersection_size = len(compat_skills)
            union_size = len(skills | target_skills)
            compat_score = intersection_size / union_size if union_size else 0.0

            novelty_score = 1.0 - sims[i]                                     # TF-IDF based novelty (higher => more novel)
            new_skill_score = len(new_skills) / (len(skills) + 1)             # proportion of new skills relative to course skill set

            # If compatibility is low, significantly reduce the overall score to avoid poor matches
            compatibility_factor = 1.0 if compat_score >= 0.1 else 0.05

            # Weighted combination of signals — tuned heuristically
            total_score = round(
                compatibility_factor * (
                    0.40 * freq_score +
                    0.35 * compat_score +
                    0.15 * new_skill_score +
                    0.10 * novelty_score
                ), 3
            )

            # Fetch textual course details, restricting to the target university (so the description is relevant)
            course_details = self.get_course_details_by_name(cname, target_univ_id)

            results.append({
                "course_name": cname,
                "score": total_score,
                "new_skills": sorted(list(new_skills)),
                "compatible_skills": sorted(list(compat_skills)),
                "description": course_details["description"],
                "objectives": course_details["objectives"],
                "learning_outcomes": course_details["learning_outcomes"],
                "course_content": course_details["course_content"],
            })

        # Return top-N courses sorted by score (descending)
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

    # ==========================================================
    # 4) Suggest courses for a new degree
    # ==========================================================
    def suggest_courses_for_new_degree(
        self,
        similar_degrees: List[Dict[str, Any]],
        target_skills: Optional[set] = None,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate course suggestions for a NEW degree.

        Differences vs suggest_courses_for_degree:
          - We do not exclude courses that may already appear in a target (because this is a new degree).
          - The algorithm still aggregates course-level and degree-level skills, computes TF-IDF
            similarity against the provided target_skills set (if any), and ranks courses using
            the same composite scoring formula.

        Returns a list of course dictionaries similar to suggest_courses_for_degree.
        """
        if not similar_degrees:
            return [{"info": "No similar degrees available for recommendation."}]

        if target_skills is None:
            target_skills = set()

        course_freq = defaultdict(int)
        course_skills = defaultdict(set)
        course_specific_skills_map = dict()

        # Aggregate candidate courses and skills across all similar degrees
        for deg in similar_degrees:
            deg_skills = set(deg.get("skills", []))
            for cname in deg.get("courses", []):
                if not cname:
                    continue

                db_course = self.db.query(Course).filter(Course.lesson_name == cname).first()
                course_specific_skills = set()
                if db_course:
                    for cs in getattr(db_course, "skills", []):
                        if getattr(cs, "skill", None):
                            skill_name = (cs.skill.skill_name or "").strip()
                            if skill_name:
                                course_specific_skills.add(skill_name)

                if course_specific_skills or deg_skills:
                    total_skills = deg_skills.union(course_specific_skills)
                    course_freq[cname] += 1
                    course_skills[cname].update(total_skills)
                    course_specific_skills_map[cname] = course_specific_skills

        if not course_freq:
            logger.info("No new courses found for recommendation.")
            return [{"info": "No new courses found for recommendation."}]

        # Prepare TF-IDF documents (candidate course skills + target skills)
        courses_list = list(course_freq.keys())
        course_docs = [" ".join(course_skills[c]) for c in courses_list]
        target_doc = " ".join(target_skills)
        docs = course_docs + [target_doc]

        if all(not doc.strip() for doc in docs):
            logger.info("Recommendation failed due to lack of skill text for comparison.")
            return [{"info": "Recommendation failed due to lack of skill text for comparison."}]

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(docs)
        sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        max_freq = max(course_freq.values()) if course_freq else 1

        results = []
        for i, cname in enumerate(courses_list):
            skills = course_skills[cname]
            freq_score = course_freq[cname] / max_freq
            new_skills = course_specific_skills_map.get(cname, set()) - target_skills
            compat_skills = skills & target_skills

            intersection_size = len(compat_skills)
            union_size = len(skills | target_skills)
            compat_score = intersection_size / union_size if union_size else 0.0
            novelty_score = 1.0 - sims[i]
            new_skill_score = len(new_skills) / (len(skills) + 1)
            compatibility_factor = 1.0 if compat_score >= 0.1 else 0.05

            total_score = round(
                compatibility_factor * (
                    0.40 * freq_score +
                    0.35 * compat_score +
                    0.15 * new_skill_score +
                    0.10 * novelty_score
                ), 3
            )

            # Course details are retrieved without restricting university (new degree context)
            course_details = self.get_course_details_by_name(cname, target_univ_id=None)

            results.append({
                "course_name": cname,
                "score": total_score,
                "new_skills": sorted(list(new_skills)),
                "compatible_skills": sorted(list(compat_skills)),
                "description": course_details["description"],
                "objectives": course_details["objectives"],
                "learning_outcomes": course_details["learning_outcomes"],
                "course_content": course_details["course_content"],
            })

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
