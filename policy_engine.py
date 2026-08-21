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
    # STEP 1: Skills from Trig — top-N required skills per occupation
    #         Επιστρέφει skills με URL + label (matching γίνεται με URL)
    # ------------------------------------------------------------------
    def _fetch_skills_for_occupation(
        self,
        occupation: str,
        min_val: float,
        top_n: Optional[int] = None
    ) -> tuple:
        """
        HTTP request για ένα occupation στο Trig.

        Trig response item:
          {"Role": ..., "Skill": <label>, "Pillar": ...,
           "Importance": <0..1>, "SkillId": <esco url>}

        Επιστρέφει (occupation, [ {"url": ..., "label": ..., "importance": ...} ])
        ταξινομημένα κατά Importance φθίνουσα, μέχρι top_n.
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

            # Το service επιστρέφει error string μέσα σε λίστα σε αποτυχία
            # π.χ. ["cannot open the connection"] / ["argument 1 is not a vector"]
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                logger.warning(f"[{occupation}] Trig error: {data[0]}")
                return occupation, []

            if not isinstance(data, list):
                return occupation, []

            # Κράτα (url, label, importance) και φίλτραρε με το threshold
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

            # Ταξινόμηση κατά Importance φθίνουσα, μετά slice στο top_n
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
        min_val: float = 0.1,
        top_n: Optional[int] = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Παράλληλα HTTP requests για όλα τα occupations (max_workers=10).
        Κάθε occupation επιστρέφει μέχρι top_n skills (url + label + importance).
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
    # STEP 2: University skills from DB (grouped per university + country)
    #         Επιστρέφει και το skill_url για URL-based matching
    # ------------------------------------------------------------------
    def get_all_universities_skills(self) -> Dict[str, Dict[str, Any]]:
        """
        Επιστρέφει skills ανά πανεπιστήμιο, μαζί με τη χώρα του καθενός.
        {
          university_name: {
            "country": str,
            "skill_urls": set([...]),   # για matching με το Trig SkillId
            "skill_names": set([...])   # fallback / εμφάνιση
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
    #         (matching με skill_url)
    # ------------------------------------------------------------------
    def get_courses_from_other_universities(
        self,
        skill_urls: List[str],
        current_university: str
    ) -> Dict[str, List[str]]:
        """
        Batch IN query πάνω στο s.skill_url. Βρίσκει μαθήματα από ΑΛΛΑ
        πανεπιστήμια που διδάσκουν τα missing skills. Το dict έχει key
        το skill label (για ευανάγνωστο output).
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
    # MAIN: Gap analysis per university (+ country aggregate)
    # ------------------------------------------------------------------
    def run_analysis(
        self,
        occupations: List[str],
        skill_threshold: float = 0.1,
        top_n: int = 100
    ) -> Dict[str, Any]:
        """
        Gap analysis με βάση occupations που δίνει ο χρήστης.

        1. Για κάθε occupation -> top_n required skills από το Trig
           (url + label + importance).
        2. Union όλων -> distinct required skills (κλειδί: ESCO url).
        3. Σύγκριση με τα skill_urls ΚΑΘΕ πανεπιστημίου (DB) -> coverage & gap.
        4. Aggregate ανά χώρα (union coverage + μέσος όρος παν/μίων).
        """
        occupations = [o.strip() for o in (occupations or []) if o and o.strip()]
        if not occupations:
            return {"error": "No occupations provided"}

        logger.info(
            f"Loading top-{top_n} required skills (threshold={skill_threshold}) "
            f"for {len(occupations)} user-selected occupations..."
        )
        req_skills = self.get_required_skills(occupations, skill_threshold, top_n=top_n)

        if not req_skills:
            return {"error": "No required skills returned for the selected occupations"}

        logger.info("Loading university skills from DB...")
        uni_data = self.get_all_universities_skills()

        total_unis = len(uni_data)
        logger.info(f"Starting analysis for {total_unis} universities...")

        # ---- Union όλων των top-N skills -> distinct required set (by URL) ----
        # url -> label (για ευανάγνωστο output)
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
        country_present = defaultdict(set)     # union present urls ανά χώρα
        country_scores = defaultdict(list)     # coverage κάθε παν/μίου ανά χώρα

        for idx, (uni, data) in enumerate(uni_data.items(), start=1):
            country = data["country"]
            uni_urls = data["skill_urls"]

            present_urls = all_req_urls.intersection(uni_urls)
            coverage_score = round((len(present_urls) / total_needed) * 100, 2) if total_needed > 0 else 0.0
            missing_urls = all_req_urls - uni_urls

            # missing skills ανά occupation (με labels)
            missing_by_occ = {}
            for occ, skills in req_skills.items():
                occ_missing = [sk["label"] for sk in skills if sk["url"] in missing_urls]
                if occ_missing:
                    missing_by_occ[occ] = sorted(set(occ_missing))

            # πού διδάσκονται τα missing skills (άλλα παν/μια) — matching με url
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

            # aggregate ανά χώρα
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
                "union_coverage_score": union_coverage,    # τι καλύπτει η χώρα κάπου
                "avg_university_coverage": avg_coverage,   # μέση κάλυψη ανά παν/μιο
                "universities_count": len(scores)
            }

        logger.info(
            f"🎉 Analysis complete: {total_unis} universities, "
            f"{len(country_results)} countries."
        )

        # required_skills_per_occupation ως labels (για ευανάγνωστο output)
        req_skills_labels = {
            occ: [sk["label"] for sk in skills]
            for occ, skills in req_skills.items()
        }

        return {
            "selected_occupations": occupations,
            "occupations_with_skills": list(req_skills.keys()),
            "top_n": top_n,
            "threshold": skill_threshold,
            "total_unique_required_skills": total_needed,
            "required_skills_per_occupation": req_skills_labels,
            "universities": university_results,
            "countries": country_results
        }