import os
import logging
import requests
import mysql.connector
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

try:
    from config import DB_CONFIG
except ImportError:
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "mysql-curriculum-skill"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "root"),
        "database": os.getenv("DB_NAME", "skillcrawl"),
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EducationRecommendationSystem:

    def __init__(self, service2_url: str):
        self.service2_url = service2_url

    # ------------------------------------------------------------------
    # STEP 1: Skills from Trig — top-N required skills per occupation.
    #         Returns skills with URL + label (matching is done by URL).
    # ------------------------------------------------------------------
    def _fetch_skills_for_occupation(
        self,
        occupation: str,
        min_val: float,
        top_n: Optional[int] = None
    ) -> tuple:
        """
        HTTP request to Trig for a single occupation.

        Trig response item:
          {"Role": ..., "Skill": <label>, "Pillar": ...,
           "Importance": <0..1>, "SkillId": <esco url>}

        Returns (occupation, [ {"url": ..., "label": ..., "importance": ...} ])
        sorted by Importance descending, capped at top_n.
        """
        try:
            payload = {"occupation_name": occupation}
            resp = requests.post(
                f"{self.service2_url}/required_skills_service",
                json=payload,
                timeout=60
            )

            if resp.status_code != 200 or not resp.text:
                logger.warning(
                    f"[{occupation}] Trig status={resp.status_code} "
                    f"body={resp.text[:200]}"
                )
                return occupation, []

            data = resp.json()

            # On failure, the service returns an error string inside a list,
            # e.g. ["cannot open the connection"] / ["argument 1 is not a vector"]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                logger.warning(f"[{occupation}] Trig error: {data[0]}")
                return occupation, []

            if not isinstance(data, list):
                return occupation, []

            # Keep (url, label, importance) and filter by the threshold
            scored = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                importance = item.get('Importance', 0) or 0
                if importance >= min_val:
                    url = item.get('SkillId')
                    label = item.get('Skill')
                    if url:
                        scored.append({
                            "url": url.strip(),
                            "label": (label or "").strip(),
                            "importance": importance
                        })

            # Sort by Importance descending, then cap at top_n
            scored.sort(key=lambda x: x["importance"], reverse=True)
            if top_n is not None:
                scored = scored[:top_n]

            return occupation, scored

        except Exception as e:
            logger.warning(f"Failed to fetch skills for '{occupation}': {e}")
            return occupation, []

    def get_required_skills(
        self,
        occupation_titles: List[str],
        min_val: float = 0.0,
        top_n: Optional[int] = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Parallel HTTP requests for all occupations (max_workers=10).
        Each occupation returns up to top_n skills (url + label + importance).
        """
        occupation_skills = {}
        total = len(occupation_titles)
        logger.info(
            f"Fetching top-{top_n} required skills (min_importance={min_val}) "
            f"for {total} occupations in parallel (max_workers=10)..."
        )

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self._fetch_skills_for_occupation, occ, min_val, top_n): occ
                for occ in occupation_titles
            }

            completed = 0
            for future in as_completed(futures):
                occupation, skills = future.result()
                if skills:
                    occupation_skills[occupation] = skills
                completed += 1
                if completed % 50 == 0:
                    logger.info(f"  Progress: {completed}/{total} occupations processed...")

        logger.info(
            f"✅ Skills fetched for {len(occupation_skills)}/{total} occupations with results."
        )
        return occupation_skills

    # ------------------------------------------------------------------
    # STEP 2: University skills from DB (grouped per university + country).
    #         Also returns skill_url for URL-based matching.
    # ------------------------------------------------------------------
    def get_all_universities_skills(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns skills per university, together with each university's country.
        {
          university_name: {
            "country": str,
            "skill_urls": set([...]),   # for matching against Trig SkillId
            "skill_names": set([...])   # fallback / display
          }
        }
        """
        results = {}
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT u.university_name, u.country, s.skill_name, s.skill_url
                FROM Skill s
                JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                JOIN Course c ON cs.course_id = c.course_id
                JOIN University u ON c.university_id = u.university_id
                WHERE u.university_name IS NOT NULL AND u.university_name != ''
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            grouped = defaultdict(lambda: {"country": None, "skill_urls": set(), "skill_names": set()})
            for r in rows:
                uni = r["university_name"]
                grouped[uni]["country"] = r["country"] or "Unknown"
                if r.get("skill_url"):
                    grouped[uni]["skill_urls"].add(r["skill_url"].strip())
                if r.get("skill_name"):
                    grouped[uni]["skill_names"].add(r["skill_name"].strip())

            results = {uni: dict(data) for uni, data in grouped.items()}

        except Exception as e:
            logger.error(f"DB Error in get_all_universities_skills: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()

        return results

    # ------------------------------------------------------------------
    # STEP 3: Courses from OTHER universities that teach the missing skills
    #         (matched by skill_url).
    # ------------------------------------------------------------------
    def get_courses_from_other_universities(
        self,
        skill_urls: List[str],
        current_university: str
    ) -> Dict[str, List[str]]:
        """
        Batch IN query on s.skill_url. Finds courses from OTHER universities
        that teach the missing skills. The returned dict is keyed by the
        skill label (for readable output).
        """
        if not skill_urls:
            return {}

        skill_courses = defaultdict(list)
        conn = None
        BATCH_SIZE = 50

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            logger.info(
                f"  Fetching courses for {len(skill_urls)} missing skills "
                f"(excluding '{current_university}', batch size={BATCH_SIZE})..."
            )

            for i in range(0, len(skill_urls), BATCH_SIZE):
                batch = skill_urls[i:i + BATCH_SIZE]
                placeholders = ', '.join(['%s'] * len(batch))

                query = f"""
                    SELECT s.skill_name, s.skill_url, c.lesson_name,
                           u.university_name, u.country
                    FROM Skill s
                    JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                    JOIN Course c ON cs.course_id = c.course_id
                    JOIN University u ON c.university_id = u.university_id
                    WHERE s.skill_url IN ({placeholders})
                    AND u.university_name <> %s
                    LIMIT 500
                """
                cursor.execute(query, [*batch, current_university])
                rows = cursor.fetchall()

                for r in rows:
                    label = r.get("skill_name") or r.get("skill_url")
                    entry = f"{r['lesson_name']} ({r['university_name']}) - [{r['country']}]"
                    skill_courses[label].append(entry)

            result = {skill: list(set(courses)) for skill, courses in skill_courses.items()}

        except Exception as e:
            logger.error(f"DB Error in get_courses_from_other_universities: {e}")
            result = {}
        finally:
            if conn and conn.is_connected():
                conn.close()

        return result

    # ------------------------------------------------------------------
    # MAIN: Gap analysis per university (+ country aggregate).
    # ------------------------------------------------------------------
    def run_analysis(
        self,
        occupations: List[str],
        skill_threshold: float = 0.0,
        top_n: int = 100
    ) -> Dict[str, Any]:
        """
        Gap analysis based on a user-selected list of occupations.

        1. For each occupation -> top_n required skills from Trig
           (url + label + importance).
        2. Union of all -> distinct required skills (key: ESCO url).
        3. Compare against each university's skill_urls (DB) -> coverage & gap.
        4. Aggregate per country (union coverage + average per university).
        """
        occupations = [o.strip() for o in (occupations or []) if o and o.strip()]
        if not occupations:
            return {"error": "No occupations provided"}

        logger.info(
            f"Loading top-{top_n} required skills (threshold={skill_threshold}) "
            f"for {len(occupations)} user-selected occupations..."
        )
        req_skills = self.get_required_skills(occupations, skill_threshold, top_n=top_n)

        # Track which occupations returned no skills (so the caller knows)
        occupations_with_skills = list(req_skills.keys())
        occupations_without_skills = [o for o in occupations if o not in req_skills]

        if not req_skills:
            return {
                "error": "No required skills returned for the selected occupations",
                "occupations_without_skills": occupations_without_skills
            }

        logger.info("Loading university skills from DB...")
        uni_data = self.get_all_universities_skills()

        total_unis = len(uni_data)
        logger.info(f"Starting analysis for {total_unis} universities...")

        # ---- Union of all top-N skills -> distinct required set (by URL) ----
        # url -> label (for readable output)
        url_to_label: Dict[str, str] = {}
        for skills in req_skills.values():
            for sk in skills:
                url_to_label[sk["url"]] = sk["label"] or sk["url"]

        all_req_urls = set(url_to_label.keys())
        total_needed = len(all_req_urls)

        logger.info(
            f"Total unique required skills (union of top-{top_n} across "
            f"{len(req_skills)} occupations): {total_needed}"
        )

        university_results = {}
        country_present = defaultdict(set)     # union of present urls per country
        country_scores = defaultdict(list)     # coverage of each university per country

        for idx, (uni, data) in enumerate(uni_data.items(), start=1):
            country = data["country"]
            uni_urls = data["skill_urls"]

            present_urls = all_req_urls.intersection(uni_urls)
            coverage_score = round((len(present_urls) / total_needed) * 100, 2) if total_needed > 0 else 0.0
            missing_urls = all_req_urls - uni_urls

            # missing skills per occupation (as labels)
            missing_by_occ = {}
            for occ, skills in req_skills.items():
                occ_missing = [sk["label"] for sk in skills if sk["url"] in missing_urls]
                if occ_missing:
                    missing_by_occ[occ] = sorted(set(occ_missing))

            # where the missing skills are taught (other universities) — matched by url
            missing_courses = {}
            if missing_urls:
                missing_courses = self.get_courses_from_other_universities(
                    list(missing_urls), uni
                )

            university_results[uni] = {
                "country": country,
                "coverage_score": coverage_score,
                "present_skills_count": len(present_urls),
                "missing_skills_count": len(missing_urls),
                "present_skills": sorted({url_to_label[u] for u in present_urls}),
                "missing_departments": missing_by_occ,
                "missing_courses": missing_courses
            }

            # aggregate per country
            country_present[country].update(present_urls)
            country_scores[country].append(coverage_score)

            logger.info(
                f"[{idx}/{total_unis}] {uni} ({country}): "
                f"coverage={coverage_score}%, missing={len(missing_urls)}"
            )

        # ---- Country-level aggregate ----
        country_results = {}
        for country, present_set in country_present.items():
            union_coverage = round((len(present_set) / total_needed) * 100, 2) if total_needed > 0 else 0.0
            scores = country_scores[country]
            avg_coverage = round(sum(scores) / len(scores), 2) if scores else 0.0
            country_results[country] = {
                "union_coverage_score": union_coverage,    # what the country covers somewhere
                "avg_university_coverage": avg_coverage,   # average coverage per university
                "universities_count": len(scores)
            }

        logger.info(
            f"🎉 Analysis complete: {total_unis} universities, "
            f"{len(country_results)} countries."
        )

        # required_skills_per_occupation as labels (for readable output)
        req_skills_labels = {
            occ: [sk["label"] for sk in skills]
            for occ, skills in req_skills.items()
        }

        return {
            "selected_occupations": occupations,
            "occupations_with_skills": occupations_with_skills,
            "occupations_without_skills": occupations_without_skills,
            "top_n": top_n,
            "threshold": skill_threshold,
            "total_unique_required_skills": total_needed,
            "required_skills_per_occupation": req_skills_labels,
            "universities": university_results,
            "countries": country_results
        }