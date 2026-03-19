import os
import pandas as pd
import requests
import logging
import mysql.connector
from typing import List, Dict, Any
from dataclasses import dataclass

try:
    from config import DB_CONFIG
except ImportError:
    import os
    DB_CONFIG = {
        #"host": os.getenv("DB_HOST", "db"),
        "host": os.getenv("DB_HOST", "mysql-curriculum-skill"),
        "port": int(os.getenv("DB_PORT", 3306)),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "root"),
        "database": os.getenv("DB_NAME", "skillcrawl"),
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    missing_departments: Dict[str, List[str]]
    missing_courses: Dict[str, List[str]]


class EducationRecommendationSystem:

    def __init__(self, service2_url: str, unused_service3_url: str, csv_path: str):
        self.service2_url = service2_url
        self.csv_path = csv_path

    def get_available_sectors(self) -> List[str]:
        if not os.path.exists(self.csv_path):
            return []
        try:
            try:
                df = pd.read_csv(self.csv_path, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
            except:
                df = pd.read_csv(self.csv_path, sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')

            df.columns = df.columns.str.strip().str.replace('"', '')

            if 'Label3' in df.columns:
                sectors = df['Label3'].dropna().astype(str).str.strip().str.strip('"').unique().tolist()
                return sorted(sectors)
            return []
        except Exception as e:
            logger.error(f"Error reading sectors: {e}")
            return []

    def load_occupation_titles_from_csv(self, sector_filter: str = None) -> List[str]:
        if not os.path.exists(self.csv_path):
            logger.error(f"CSV file not found at: {self.csv_path}")
            return []

        try:
            try:
                df = pd.read_csv(self.csv_path, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
            except:
                df = pd.read_csv(self.csv_path, sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')

            df.columns = df.columns.str.strip().str.replace('"', '')

            if sector_filter and 'Label3' in df.columns:
                logger.info(f"Filtering occupations by sector: '{sector_filter}'")
                df = df[df['Label3'].astype(str).str.contains(sector_filter, case=False, na=False)]

            occupations = set()

            def clean_text(text):
                if not isinstance(text, str): return ""
                return text.replace("β€™", "'").replace("â€™", "'").strip().strip('"').strip("'")

            if 'Label4' in df.columns:
                occupations.update(df['Label4'].dropna().apply(clean_text).tolist())

            if 'Label3' in df.columns:
                occupations.update(df['Label3'].dropna().apply(clean_text).tolist())

            result = [occ for occ in occupations if occ and len(occ) > 2]

            logger.info(f"✅ Loaded {len(result)} occupations (Sector: {sector_filter if sector_filter else 'ALL'}).")
            return result

        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []

    def get_required_skills(self, occupation_titles: List[str], min_val: float = 0.7) -> Dict[str, List[str]]:
        occupation_skills = {}
        # Slice για ταχύτητα
        for occupation in occupation_titles[:30]:
            try:
                payload = {"occupation_name": occupation}
                resp = requests.post(f"{self.service2_url}/required_skills_service", json=payload, timeout=10)

                if resp.status_code == 200 and resp.text:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                        if "cannot open" in data[0].lower() or "error" in data[0].lower():
                            occupation_skills[occupation] = []
                            continue
                    filtered = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get('Value', 0) >= min_val:
                                filtered.append(item.get('Skill'))
                    if filtered:
                        occupation_skills[occupation] = filtered
                else:
                    occupation_skills[occupation] = []
            except Exception:
                occupation_skills[occupation] = []
        return occupation_skills

    def get_all_countries_skills(self) -> Dict[str, List[str]]:
        results = {}
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT u.country, s.skill_name
                FROM Skill s
                JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                JOIN Course c ON cs.course_id = c.course_id
                JOIN University u ON c.university_id = u.university_id
                WHERE s.skill_name IS NOT NULL AND s.skill_name != '' AND u.country IS NOT NULL
            """
            cursor.execute(query)
            rows = cursor.fetchall()

            from collections import defaultdict
            grouped = defaultdict(list)

            for r in rows:
                country = r["country"]
                skill = r["skill_name"]
                grouped[country].append(skill)

            results = dict(grouped)

        except Exception as e:
            logger.error(f"DB Error in get_all_countries_skills: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()

        return results

    def get_courses_from_other_countries(self, skills: List[str], current_country: str) -> Dict[str, List[str]]:
        skill_courses = {}
        conn = None
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            for skill in skills[:20]:
                query = """
                    SELECT c.lesson_name, u.university_name, u.country
                    FROM Skill s
                    JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                    JOIN Course c ON cs.course_id = c.course_id
                    JOIN University u ON c.university_id = u.university_id
                    WHERE s.skill_name LIKE %s 
                    AND u.country <> %s 
                    AND u.country <> 'Unknown'
                    LIMIT 10
                """
                cursor.execute(query, (f"%{skill}%", current_country))
                rows = cursor.fetchall()

                courses = []
                for r in rows:
                    c_name = r["lesson_name"]
                    u_name = r["university_name"]
                    u_country = r["country"]

                    if current_country.lower() in u_name.lower():
                        continue

                    courses.append(f"{c_name} ({u_name}) - [{u_country}]")

                if courses:
                    skill_courses[skill] = list(set(courses))

        except Exception as e:
            logger.error(f"DB Error in get_courses_from_other_countries: {e}")
        finally:
            if conn and conn.is_connected():
                conn.close()

        return skill_courses

    def run_analysis(self, skill_threshold: float = 0.7, sector: str = None) -> Dict[str, Any]:
        occupations = self.load_occupation_titles_from_csv(sector_filter=sector)

        if not occupations:
            return {"error": f"No occupations found for sector '{sector}'" if sector else "No occupations found"}

        logger.info(f"Loading required skills (threshold={skill_threshold})...")
        req_skills = self.get_required_skills(occupations, skill_threshold)

        logger.info("Loading university skills from DB...")
        country_data = self.get_all_countries_skills()

        final_results = {}

        for country, c_skills in country_data.items():
            logger.info(f"Analyzing country: {country}")

            all_req_list = []
            for s_list in req_skills.values():
                all_req_list.extend(s_list)

            all_req_set = set(all_req_list)
            c_skills_set = set(c_skills)

            # --- ΥΠΟΛΟΓΙΣΜΟΣ COVERAGE SCORE ---
            present_skills = all_req_set.intersection(c_skills_set)
            total_needed = len(all_req_set)

            if total_needed > 0:
                coverage_score = round((len(present_skills) / total_needed) * 100, 2)
            else:
                coverage_score = 0.0
            # ----------------------------------

            missing = all_req_set - c_skills_set

            missing_depts = {}
            for occ, skills in req_skills.items():
                inter = set(skills).intersection(missing)
                if inter:
                    missing_depts[occ] = list(inter)

            missing_courses = {}
            if missing:
                missing_courses = self.get_courses_from_other_countries(list(missing), country)

            final_results[country] = {
                "coverage_score": coverage_score,
                "missing_departments": missing_depts,
                "missing_courses": missing_courses
            }

        return final_results