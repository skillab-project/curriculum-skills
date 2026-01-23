import os
import pandas as pd
import requests
import logging
import mysql.connector
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = {
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

    def load_occupation_titles_from_csv(self) -> List[str]:
        if not os.path.exists(self.csv_path):
            logger.error(f"CSV file not found at: {self.csv_path}")
            return []
        try:
            df = pd.read_csv(self.csv_path, sep=',', quotechar='"')
            df.columns = df.columns.str.strip().str.replace('"', '')
            if 'Label3' not in df.columns or 'Label4' not in df.columns:
                return []
            level3 = df['Label3'].dropna().tolist()
            level4 = df['Label4'].dropna().tolist()
            return list(set(level3 + level4))
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []

    def get_required_skills(self, occupation_titles: List[str], min_val: float = 0.7) -> Dict[str, List[str]]:
        occupation_skills = {}
        # Slice για ταχύτητα (π.χ. 30).
        for occupation in occupation_titles[:30]:
            try:
                payload = {"occupation_name": occupation}
                resp = requests.post(f"{self.service2_url}/required_skills_service", json=payload, timeout=10)

                if resp.status_code == 200 and resp.text:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
                        if "cannot open" in data[0].lower():
                            occupation_skills[occupation] = []
                            continue
                    filtered = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get('Value', 0) >= min_val:
                                filtered.append(item.get('Skill'))
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
            # Σύνδεση με το διορθωμένο DB_CONFIG
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

    def run_analysis(self, skill_threshold: float = 0.7) -> Dict[str, Any]:
        occupations = self.load_occupation_titles_from_csv()
        if not occupations:
            return {"error": "No occupations found"}

        logger.info(f"Loading required skills (threshold={skill_threshold})...")
        req_skills = self.get_required_skills(occupations, skill_threshold)

        logger.info("Loading university skills from DB...")
        country_data = self.get_all_countries_skills()

        final_results = {}

        for country, c_skills in country_data.items():
            logger.info(f"Analyzing country: {country}")
            all_req = set()
            for s_list in req_skills.values():
                all_req.update(s_list)

            missing = set(all_req) - set(c_skills)

            missing_depts = {}
            for occ, skills in req_skills.items():
                inter = set(skills).intersection(missing)
                if inter:
                    missing_depts[occ] = list(inter)

            missing_courses = {}
            if missing:
                missing_courses = self.get_courses_from_other_countries(list(missing), country)

            final_results[country] = {
                "missing_departments": missing_depts,
                "missing_courses": missing_courses
            }


        return final_results
