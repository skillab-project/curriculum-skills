# backend/core.py
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session
from recommendation_system.backend.models import University, Course
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json
import re
import logging
import math

logger = logging.getLogger(__name__)


class UniversityRecommender:
    """
    University recommender system for suggesting similar universities and
    recommending degree programs with enriched skill sets.

    IMPORTANT:
    This version reads degree names primarily from Course.degree_titles because
    the current curriculum database stores populated degree title information
    at course level. DegreeProgram.degree_titles is used only as a fallback.
    """

    def __init__(
        self,
        db: Session,
        weights: Optional[Dict[str, float]] = None,
        cache_enabled: bool = True,
    ):
        self.db = db
        self._profile_cache: Dict[int, Dict[str, Any]] = {}
        self.cache_enabled = cache_enabled

        default = {
            "frequency": 0.30,
            "novelty": 0.25,
            "compatibility": 0.20,
            "skill_enrichment": 0.15,
        }

        if weights:
            merged = default.copy()
            merged.update(weights)
            total = sum(merged.values())
            self.weights = default if total == 0 else {k: v / total for k, v in merged.items()}
        else:
            self.weights = default

    # -------------------------------------------------------------------------
    # Degree-title parsing helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _clean_degree_title(title: Any) -> str:
        """Return a clean degree-title string."""
        if title is None:
            return ""
        title = str(title).strip()
        if not title:
            return ""

        # Remove noisy punctuation but keep common degree-title symbols and Greek.
        title = re.sub(r"[^a-zA-Z0-9 \-&/().,\u0370-\u03FF\u1F00-\u1FFF]", "", title)
        title = re.sub(r"\s+", " ", title).strip()
        return title

    @classmethod
    def _extract_degree_titles_from_value(cls, raw: Any) -> List[str]:
        """
        Parse degree titles from Course.degree_titles or DegreeProgram.degree_titles.

        Supports:
        - JSON list stored as string
        - JSON object stored as string
        - Python list
        - Python dict
        - plain string
        - nested dict/list structures
        """
        if raw is None:
            return []

        parsed = raw
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return []
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw

        titles: List[str] = []

        def walk(value: Any):
            if value is None:
                return

            if isinstance(value, str):
                cleaned = cls._clean_degree_title(value)
                if cleaned and cleaned.lower() not in {"unknown", "unknown degree", "n/a", "none", "null"}:
                    titles.append(cleaned)

            elif isinstance(value, (int, float)):
                cleaned = cls._clean_degree_title(value)
                if cleaned:
                    titles.append(cleaned)

            elif isinstance(value, dict):
                # Prefer likely title/name fields.
                likely_keys = (
                    "degree_title",
                    "degree_titles",
                    "degree",
                    "title",
                    "name",
                    "program",
                    "programme",
                    "label",
                    "value",
                    "en",
                    "el",
                )
                found_likely = False
                for key in likely_keys:
                    if key in value:
                        found_likely = True
                        walk(value[key])

                # If no obvious title key exists, inspect all values.
                if not found_likely:
                    for v in value.values():
                        walk(v)

            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item)

            else:
                cleaned = cls._clean_degree_title(value)
                if cleaned:
                    titles.append(cleaned)

        walk(parsed)

        # Preserve deterministic order while removing duplicates.
        seen = set()
        out = []
        for title in titles:
            key = title.lower()
            if key not in seen:
                seen.add(key)
                out.append(title)
        return out

    @staticmethod
    def _infer_degree_type_from_title(title: str) -> str:
        deg_lower = (title or "").lower()
        if re.search(r"\b(master|msc|m\.sc|ma|m\.a)\b", deg_lower):
            return "MSc/MA"
        if re.search(r"\b(phd|doctorate|doctoral)\b", deg_lower):
            return "PhD"
        if re.search(r"\b(bachelor|bsc|b\.sc|ba|b\.a)\b", deg_lower):
            return "BSc/BA"
        return "Other"

    @staticmethod
    def _course_skill_names(course: Course) -> Set[str]:
        """Extract skill names from a Course.skills / CourseSkill relationship."""
        names: Set[str] = set()
        for cs in getattr(course, "skills", []) or []:
            skill_obj = getattr(cs, "skill", None)
            if not skill_obj:
                continue
            skill_name = (getattr(skill_obj, "skill_name", "") or "").strip()
            if skill_name:
                names.add(skill_name)
        return names

    # -------------------------------------------------------------------------
    # Build university profile from Course.degree_titles
    # -------------------------------------------------------------------------
    def build_university_profile(self, university_id: int) -> Optional[Dict[str, Any]]:
        """
        Construct a profile for a university.

        Degree titles are read primarily from Course.degree_titles because this
        field is populated in the current curriculum database. The profile includes:

        - skills: all unique skill names in the university
        - skills_raw_names: same skill names, preserved for scoring
        - courses: all course names
        - degrees: all degree titles found in Course.degree_titles
        - degree_skills: degree title -> skills observed in courses belonging to it
        - degree_courses: degree title -> course names belonging to it
        """
        if self.cache_enabled and university_id in self._profile_cache:
            return self._profile_cache[university_id]

        university = (
            self.db.query(University)
            .filter_by(university_id=university_id)
            .first()
        )
        if not university:
            return None

        profile = {
            "skills": set(),
            "skills_raw_names": set(),
            "courses": [],
            "degrees": set(),
            "degree_skills": defaultdict(set),
            "degree_courses": defaultdict(set),
        }

        # Main source: Course.degree_titles
        for course in getattr(university, "courses", []) or []:
            lesson_name = (getattr(course, "lesson_name", "") or "").strip()
            if lesson_name:
                profile["courses"].append(lesson_name)

            course_skills = self._course_skill_names(course)
            profile["skills"].update(course_skills)
            profile["skills_raw_names"].update(course_skills)

            course_degree_titles = self._extract_degree_titles_from_value(
                getattr(course, "degree_titles", None)
            )

            # If Course.degree_titles is empty, optionally fall back to program title.
            if not course_degree_titles and getattr(course, "program", None):
                course_degree_titles = self._extract_degree_titles_from_value(
                    getattr(course.program, "degree_titles", None)
                )

            for degree_title in course_degree_titles:
                profile["degrees"].add(degree_title)
                if lesson_name:
                    profile["degree_courses"][degree_title].add(lesson_name)
                profile["degree_skills"][degree_title].update(course_skills)

        # Fallback only: include DegreeProgram.degree_titles if no course-level titles exist.
        if not profile["degrees"]:
            for program in getattr(university, "programs", []) or []:
                for degree_title in self._extract_degree_titles_from_value(getattr(program, "degree_titles", None)):
                    profile["degrees"].add(degree_title)

        # Convert sets/defaultdicts to stable JSON-safe lists.
        profile["skills"] = sorted(profile["skills"])
        profile["skills_raw_names"] = sorted(profile["skills_raw_names"])
        profile["courses"] = sorted(set(profile["courses"]))
        profile["degrees"] = sorted(profile["degrees"])
        profile["degree_skills"] = {
            degree: sorted(skills)
            for degree, skills in profile["degree_skills"].items()
        }
        profile["degree_courses"] = {
            degree: sorted(courses)
            for degree, courses in profile["degree_courses"].items()
        }

        if self.cache_enabled:
            self._profile_cache[university_id] = profile

        return profile

    # -------------------------------------------------------------------------
    # Find similar universities based on skills + courses + Course.degree_titles
    # -------------------------------------------------------------------------
    def find_similar_universities(self, target_univ_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        target_profile = self.build_university_profile(target_univ_id)
        if not target_profile:
            return []

        all_univs = (
            self.db.query(University)
            .filter(University.university_id != target_univ_id)
            .all()
        )

        docs = []
        valid_univs = []

        for u in all_univs:
            p = self.build_university_profile(getattr(u, "university_id"))
            if not p:
                continue

            combined_text = " ".join(
                (p.get("skills") or []) +
                (p.get("courses") or []) +
                (p.get("degrees") or [])
            ).strip()

            if combined_text:
                docs.append(combined_text)
                valid_univs.append(u)

        if not docs:
            return []

        target_text = " ".join(
            (target_profile.get("skills") or []) +
            (target_profile.get("courses") or []) +
            (target_profile.get("degrees") or [])
        ).strip()

        if not target_text:
            return []

        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(docs + [target_text])
            sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        except Exception as e:
            logger.exception("Error computing similarity for universities: %s", e)
            return []

        ranked = sorted(zip(valid_univs, sims), key=lambda x: x[1], reverse=True)[:top_n]

        return [
            {
                "university_id": getattr(u, "university_id"),
                "name": getattr(u, "university_name", "Unknown"),
                "country": getattr(u, "country", "Unknown"),
                "similarity_score": round(float(score), 4),
            }
            for u, score in ranked
        ]

    # -------------------------------------------------------------------------
    # Recommend skills for a degree using Course.degree_titles grouping
    # -------------------------------------------------------------------------
    def _get_degree_skills_similarity(
        self,
        similar_univ_ids: List[int],
        target_degree: str,
        target_skills_raw: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        Recommend skill additions for a degree by reading the skills attached to
        courses whose Course.degree_titles contain target_degree.
        """
        skill_counter = defaultdict(int)
        all_skills = []
        target_skills_lc = {s.lower() for s in target_skills_raw}

        for univ_id in similar_univ_ids:
            profile = self.build_university_profile(univ_id)
            if not profile:
                continue

            degree_skills = set((profile.get("degree_skills") or {}).get(target_degree, []))
            if not degree_skills:
                continue

            filtered = [
                s for s in degree_skills
                if s and s.lower() not in target_skills_lc
            ]

            all_skills.extend(filtered)
            for skill in filtered:
                skill_counter[skill.strip()] += 1

        if not skill_counter:
            return []

        try:
            vectorizer = TfidfVectorizer(lowercase=True)
            vectors = vectorizer.fit_transform([" ".join(all_skills)])
            weights = dict(zip(vectorizer.get_feature_names_out(), vectors.toarray()[0]))
        except Exception:
            weights = {}

        max_count = max(skill_counter.values())
        raw_scores = []

        for skill, count in skill_counter.items():
            base_score = count / max_count
            tfidf_weight = weights.get(skill.lower(), 0.4)
            combined = 0.6 * base_score + 0.4 * tfidf_weight
            raw_scores.append((skill, combined))

        min_s = min(v for _, v in raw_scores)
        max_s = max(v for _, v in raw_scores)
        spread = max(max_s - min_s, 1.0)

        ranked_skills = []
        for skill, val in raw_scores:
            normalized = (val - min_s) / spread
            boosted = math.pow(normalized, 0.8)
            final_score = round(0.7 + 0.25 * boosted, 3)
            ranked_skills.append({
                "skill_name": skill,
                "skill_score": round(final_score, 3),
            })

        ranked_skills.sort(key=lambda x: x["skill_score"], reverse=True)
        return ranked_skills[:5]

    # -------------------------------------------------------------------------
    # Recommend degrees + enriched skills using Course.degree_titles
    # -------------------------------------------------------------------------
    def suggest_degrees_with_skills(self, target_univ_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest degree titles for a university.

        Degree candidates are taken from Course.degree_titles of similar
        universities and filtered against Course.degree_titles already present
        in the target university.
        """
        similar_univs = self.find_similar_universities(target_univ_id, top_n=10)
        target_profile = self.build_university_profile(target_univ_id)

        if not target_profile or not similar_univs:
            return []

        similar_univ_ids = [u["university_id"] for u in similar_univs]
        target_skills_raw = set(target_profile.get("skills_raw_names") or [])
        target_skills = set(target_profile.get("skills") or [])
        target_degrees = set(target_profile.get("degrees") or [])

        target_text = " ".join(
            (target_profile.get("skills") or []) +
            (target_profile.get("courses") or []) +
            (target_profile.get("degrees") or [])
        )

        degree_texts = {}
        degree_freq = defaultdict(int)
        degree_compat = defaultdict(float)
        degree_skill_bonus = defaultdict(int)

        for u in similar_univs:
            p = self.build_university_profile(u["university_id"])
            if not p:
                continue

            candidate_degrees = set(p.get("degrees") or []) - target_degrees
            p_skills = set(p.get("skills") or [])
            p_skills_raw_lc = {s.lower() for s in (p.get("skills_raw_names") or [])}
            target_skills_lc = {s.lower() for s in (target_profile.get("skills_raw_names") or [])}

            for deg in candidate_degrees:
                deg_skills = set((p.get("degree_skills") or {}).get(deg, []))
                deg_courses = set((p.get("degree_courses") or {}).get(deg, []))

                if not deg_skills and not deg_courses:
                    continue

                degree_freq[deg] += 1

                # Degree-specific text from course-level degree group.
                degree_texts[deg] = degree_texts.get(deg, "") + " " + " ".join(
                    list(deg_skills) + list(deg_courses) + [deg]
                )

                # Compatibility with target university skill profile.
                deg_skills_lc = {s.lower() for s in deg_skills}
                overlap = len(deg_skills_lc & target_skills_lc)
                union_count = len(deg_skills_lc | target_skills_lc)
                compat = overlap / (union_count + 1) if union_count else 0.0
                degree_compat[deg] += compat

                # Skill enrichment: new degree-specific skills not present in target.
                degree_skill_bonus[deg] += len(deg_skills - target_skills)

        if not degree_texts:
            return []

        degrees = list(degree_texts.keys())
        docs = [degree_texts[d] for d in degrees] + [target_text]

        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(docs)
            sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        except Exception as e:
            logger.exception("Error computing degree similarities: %s", e)
            sims = [0.0] * len(degrees)

        final = []
        max_freq = max(degree_freq.values()) if degree_freq else 1
        max_skill_bonus = max(degree_skill_bonus.values()) if degree_skill_bonus else 1

        for i, deg in enumerate(degrees):
            freq_score = degree_freq[deg] / max_freq
            novelty_score = max(0.0, min(1.0, 1.0 - float(sims[i])))
            compat_score = degree_compat[deg] / degree_freq[deg] if degree_freq[deg] else 0.0
            skill_enrichment_score = degree_skill_bonus[deg] / max_skill_bonus

            total_score = (
                self.weights["frequency"] * freq_score +
                self.weights["novelty"] * novelty_score +
                self.weights["compatibility"] * compat_score +
                self.weights["skill_enrichment"] * skill_enrichment_score
            )

            top_skills = self._get_degree_skills_similarity(
                similar_univ_ids,
                deg,
                target_skills_raw,
            )

            final.append({
                "degree": deg,
                "score": round(total_score, 3),
                "degree_type": self._infer_degree_type_from_title(deg),
                "top_skills": top_skills,
                "metrics": {
                    "frequency": round(freq_score * 100),
                    "compatibility": round(compat_score * 100),
                    "skill_enrichment": int(degree_skill_bonus[deg]),
                    "novelty": round(novelty_score * 100),
                    "degree_source": "Course.degree_titles",
                },
            })

        return sorted(final, key=lambda x: x["score"], reverse=True)[:top_n]
