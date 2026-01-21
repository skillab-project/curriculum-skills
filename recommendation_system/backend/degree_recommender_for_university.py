# backend/core.py
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session
from recommendation_system.backend.models import University
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

    The recommender combines:
    - TF-IDF semantic similarity
    - Cosine similarity
    - Jaccard-like skill compatibility
    - Frequency-based scoring
    - Novelty scoring
    - Skill enrichment scoring

    It also includes:
    - Case-insensitive matching
    - Error-safe handling
    - Optional caching for faster lookup
    - Weighted scoring model
    """

    def __init__(
        self,
        db: Session,
        weights: Optional[Dict[str, float]] = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize the recommender.

        :param db: SQLAlchemy session
        :param weights: Optional custom scoring weights
        :param cache_enabled: Enables caching of university profiles
        """
        self.db = db
        self._profile_cache: Dict[int, Dict[str, Any]] = {}
        self.cache_enabled = cache_enabled

        # Default scoring weights
        default = {
            "frequency": 0.30,
            "novelty": 0.25,
            "compatibility": 0.20,
            "skill_enrichment": 0.15,
        }

        # Merge and normalize custom weights
        if weights:
            merged = default.copy()
            merged.update(weights)
            total = sum(merged.values())
            if total == 0:
                logger.warning("Provided weights sum to 0. Reverting to defaults.")
                merged = default
            else:
                merged = {k: v / total for k, v in merged.items()}
            self.weights = merged
        else:
            self.weights = default

    # -------------------------------------------------------------------------
    # Build university profile (skills, courses, degrees)
    # -------------------------------------------------------------------------
    def build_university_profile(self, university_id: int) -> Optional[Dict[str, Any]]:
        """
        Construct a profile for a university, including:
        - All unique skills present in university courses
        - All course names
        - All degree titles

        Uses caching for performance.

        :param university_id: The university to index
        :return: A structured dictionary profile
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
        }

        # Extract courses and their skills
        for course in getattr(university, "courses", []) or []:
            lesson_name = getattr(course, "lesson_name", None)
            if lesson_name:
                profile["courses"].append(lesson_name.strip())

            # Extract raw skill objects linked to the course
            for cs in getattr(course, "skills", []) or []:
                skill_obj = getattr(cs, "skill", None)
                if skill_obj:
                    skill_name = (getattr(skill_obj, "skill_name", "") or "").strip()
                    if skill_name:
                        profile["skills"].add(skill_name)
                        profile["skills_raw_names"].add(skill_name)

        # Extract degree titles
        for program in getattr(university, "programs", []) or []:
            titles = getattr(program, "degree_titles", None)
            if not titles:
                continue

            # Normalize title lists
            if isinstance(titles, str):
                try:
                    titles = json.loads(titles)
                except Exception:
                    titles = [titles]
            if not isinstance(titles, list):
                titles = [titles]

            for title in titles:
                if not title:
                    continue
                clean_title = re.sub(r"[^a-zA-Z0-9 \-&]", "", str(title)).strip()
                if clean_title:
                    profile["degrees"].add(clean_title)

        # Sort to maintain deterministic structure
        profile["skills"] = sorted(profile["skills"])
        profile["skills_raw_names"] = sorted(list(profile["skills_raw_names"]))
        profile["courses"] = sorted(list({c for c in profile["courses"] if c}))
        profile["degrees"] = sorted(profile["degrees"])

        if self.cache_enabled:
            self._profile_cache[university_id] = profile

        return profile

    # -------------------------------------------------------------------------
    # Find similar universities based on skills + courses + degrees
    # -------------------------------------------------------------------------
    def find_similar_universities(self, target_univ_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Computes semantic similarity between the target university and all others
        using TF-IDF + cosine similarity.

        :param target_univ_id: ID of the target university
        :param top_n: Number of results to return
        """
        target_profile = self.build_university_profile(target_univ_id)
        if not target_profile:
            return []

        # Fetch remaining universities
        all_univs = (
            self.db.query(University)
            .filter(University.university_id != target_univ_id)
            .all()
        )

        docs = []
        valid_univs = []

        # Construct text representations for comparison
        for u in all_univs:
            p = self.build_university_profile(getattr(u, "university_id"))
            if not p:
                continue

            parts = []
            if p["skills"]:
                parts.append(" ".join(p["skills"]))
            if p["courses"]:
                parts.append(" ".join(p["courses"]))
            if p["degrees"]:
                parts.append(" ".join(p["degrees"]))

            combined_text = " ".join(parts).strip()
            if combined_text:
                docs.append(combined_text)
                valid_univs.append(u)

        if not docs:
            return []

        # Target university text representation
        target_parts = []
        if target_profile["skills"]:
            target_parts.append(" ".join(target_profile["skills"]))
        if target_profile["courses"]:
            target_parts.append(" ".join(target_profile["courses"]))
        if target_profile["degrees"]:
            target_parts.append(" ".join(target_profile["degrees"]))

        target_text = " ".join(target_parts).strip()

        # Compute cosine similarity
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(docs + [target_text])
            sims = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
        except Exception as e:
            logger.exception("Error computing similarity for universities: %s", e)
            return []

        ranked = sorted(
            zip(valid_univs, sims),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]

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
    # Compute similarity for skills related to a specific degree
    # -------------------------------------------------------------------------
    def _get_degree_skills_similarity(
        self, similar_univ_ids: List[int], target_degree: str, target_skills_raw: Set[str]
    ) -> List[Dict[str, Any]]:
        """
        Computes recommended skill additions for a specific degree by analyzing
        similar universities offering the same degree.

        The score combines:
        - Frequency across universities
        - TF-IDF semantic weight
        - Normalized spread adjustment

        :return: Top skill recommendations sorted by score
        """
        skill_counter = defaultdict(int)
        all_skills = []

        target_skills_lc = {s.lower() for s in target_skills_raw}

        # Collect skills from universities offering the same degree
        for univ_id in similar_univ_ids:
            profile = self.build_university_profile(univ_id)
            if not profile or target_degree not in profile["degrees"]:
                continue

            filtered = [
                s for s in profile["skills_raw_names"]
                if s.lower() not in target_skills_lc
            ]

            all_skills.extend(filtered)
            for skill in filtered:
                skill_counter[skill.strip()] += 1

        if not skill_counter:
            return []

        # Compute TF-IDF weighting
        try:
            vectorizer = TfidfVectorizer(lowercase=True)
            vectors = vectorizer.fit_transform([" ".join(all_skills)])
            weights = dict(zip(vectorizer.get_feature_names_out(), vectors.toarray()[0]))
        except Exception:
            weights = {}

        max_count = max(skill_counter.values())
        raw_scores = []

        # Merge frequency + tf-idf
        for skill, count in skill_counter.items():
            base_score = count / max_count
            tfidf_weight = weights.get(skill.lower(), 0.4)
            combined = 0.6 * base_score + 0.4 * tfidf_weight
            raw_scores.append((skill, combined))

        # Normalize spread for strengthened scoring contrast
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
                "skill_score": round(final_score, 3)
            })

        ranked_skills.sort(key=lambda x: x["skill_score"], reverse=True)
        return ranked_skills[:5]

    # -------------------------------------------------------------------------
    # Recommend degrees + enriched skills
    # -------------------------------------------------------------------------
    def suggest_degrees_with_skills(self, target_univ_id: int, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest new degrees for a university, along with a breakdown of
        enriched skills that the university can add.

        The scoring model considers:
        - Frequency across similar universities
        - Skill compatibility
        - Skill enrichment
        - Novelty w.r.t semantic features

        :param target_univ_id: The university to enrich
        :param top_n: Max number of degrees to return
        """
        similar_univs = self.find_similar_universities(target_univ_id, top_n=10)
        target_profile = self.build_university_profile(target_univ_id)

        if not target_profile or not similar_univs:
            return []

        similar_univ_ids = [u["university_id"] for u in similar_univs]
        target_skills_raw = set(target_profile["skills_raw_names"])
        target_degrees = set(target_profile["degrees"])
        target_text = " ".join(
            target_profile["skills"] +
            target_profile["courses"] +
            target_profile["degrees"]
        )

        degree_texts = {}
        degree_freq = defaultdict(int)
        degree_compat = defaultdict(float)
        degree_skill_bonus = defaultdict(int)

        # Extract features from similar universities
        for u in similar_univs:
            p = self.build_university_profile(u["university_id"])
            if not p:
                continue

            new_degrees = set(p["degrees"]) - target_degrees
            new_skills = set(p["skills"]) - set(target_profile["skills"])

            combined_text = " ".join(p["skills"] + p["courses"])

            for deg in new_degrees:
                degree_freq[deg] += 1
                degree_texts[deg] = degree_texts.get(deg, "") + " " + combined_text

                p_skills_raw = {s.lower() for s in p["skills_raw_names"]}
                target_skills_lc = {s.lower() for s in target_profile["skills_raw_names"]}

                overlap = len(p_skills_raw & target_skills_lc)
                union_count = len(p_skills_raw | target_skills_lc)
                compat = overlap / (union_count + 1)

                degree_compat[deg] += compat
                degree_skill_bonus[deg] += len(new_skills)

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

        # Build final degree scoring
        for i, deg in enumerate(degrees):
            freq_score = degree_freq[deg] / max_freq
            novelty_score = max(0.0, min(1.0, 1.0 - float(sims[i])))
            compat_score = degree_compat[deg] / degree_freq[deg]
            skill_enrichment_score = degree_skill_bonus[deg] / max_skill_bonus

            total_score = (
                self.weights["frequency"] * freq_score +
                self.weights["novelty"] * novelty_score +
                self.weights["compatibility"] * compat_score +
                self.weights["skill_enrichment"] * skill_enrichment_score
            )

            # Simple degree type classifier
            deg_lower = deg.lower()
            if re.search(r'\b(master|msc|ma|m\.sc|msc)\b', deg_lower):
                degree_type = 'MSc/MA'
            elif re.search(r'\b(phd|doctorate|doctoral)\b', deg_lower):
                degree_type = 'PhD'
            else:
                degree_type = 'BSc/BA'

            top_skills = self._get_degree_skills_similarity(
                similar_univ_ids, deg, target_skills_raw
            )

            final.append({
                "degree": deg,
                "score": round(total_score, 3),
                "degree_type": degree_type,
                "top_skills": top_skills,
                "metrics": {
                    "frequency": round(freq_score * 100),
                    "compatibility": round(compat_score * 100),
                    "skill_enrichment": int(degree_skill_bonus[deg]),
                    "novelty": round(novelty_score * 100)
                }
            })

        return sorted(final, key=lambda x: x['score'], reverse=True)[:top_n]
