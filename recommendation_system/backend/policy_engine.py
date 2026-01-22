import os
import json
import pandas as pd
import requests
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

# Ρύθμιση logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    missing_departments: Dict[str, List[str]]
    missing_courses: Dict[str, List[str]]


class EducationRecommendationSystem:

    def __init__(self, service2_url: str, service3_url: str, csv_path: str):
        self.service2_url = service2_url
        self.service3_url = service3_url
        self.csv_path = csv_path

        # Cache για να θυμόμαστε τη χώρα κάθε πανεπιστημίου και να μην ρωτάμε συνέχεια το API
        # Μορφή: { "University Name": "Country" }
        self.uni_country_cache = {}

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
        # ΣΗΜΕΙΩΣΗ: Βάζουμε slice [:30] για να μην αργεί πολύ το demo.
        # Για πλήρη ανάλυση αφαίρεσε το [:30].
        for occupation in occupation_titles[:30]:
            try:
                payload = {"occupation_name": occupation}
                resp = requests.post(f"{self.service2_url}/required_skills_service", json=payload, timeout=10)

                if resp.status_code == 200 and resp.text:
                    data = resp.json()
                    # Έλεγχος για error messages που επιστρέφουν 200 ΟΚ
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
        try:
            # Endpoint 2: Skills Location
            resp = requests.get(f"{self.service3_url}/exploratory/skills_location", timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                results = {}
                for entry in data:
                    c_name = entry.get('country')
                    skills = [s.get('skill') for s in entry.get('skills', []) if 'skill' in s]
                    if c_name:
                        results[c_name] = skills
                return results
        except Exception as e:
            logger.error(f"Error fetching countries: {e}")
        return {}

    def get_university_country(self, uni_name: str) -> str:
        """
        Endpoint 4: Ρωτάει το API σε ποια χώρα ανήκει το πανεπιστήμιο.
        """
        # 1. Έλεγχος Cache
        if uni_name in self.uni_country_cache:
            return self.uni_country_cache[uni_name]

        # 2. Κλήση API
        try:
            params = {"university_name": uni_name, "page": 1, "per_page": 1}
            resp = requests.get(f"{self.service3_url}/university/all", params=params, timeout=5)

            if resp.status_code == 200:
                data = resp.json()
                # Δομή response: { "university": { "country": "Greece", ... } }
                uni_data = data.get("university", {})
                country = uni_data.get("country", "Unknown")

                # Αποθήκευση στη μνήμη
                self.uni_country_cache[uni_name] = country
                return country
        except Exception:
            pass

        return "Unknown"

    def get_courses_from_other_countries(self, skills: List[str], current_country: str) -> Dict[str, List[str]]:
        skill_courses = {}

        # Endpoint 3: Search Skill (Παίρνουμε δείγμα 20 skills για ταχύτητα)
        for skill in skills[:20]:
            try:
                resp = requests.get(f"{self.service3_url}/search_skill", params={"skill": skill}, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    courses = []

                    if isinstance(data, dict) and 'universities' in data:
                        for uni_name, c_list in data['universities'].items():

                            # === ΕΞΥΠΝΟ ΦΙΛΤΡΑΡΙΣΜΑ ===
                            # Βρίσκουμε την πραγματική χώρα του Πανεπιστημίου
                            real_country = self.get_university_country(uni_name)

                            # Αν η χώρα του Πανεπιστημίου είναι η ίδια με τη χώρα που αναλύουμε, ΤΟ ΑΓΝΟΟΥΜΕ.
                            if real_country == current_country:
                                continue

                            # Fallback: Αν το API δεν βρήκε χώρα (Unknown), κάνουμε έλεγχο με το όνομα
                            if real_country == "Unknown" and current_country.lower() in uni_name.lower():
                                continue

                            # Προσθήκη στη λίστα με ένδειξη χώρας
                            country_str = f" - [{real_country}]" if real_country != "Unknown" else ""

                            for c in c_list:
                                courses.append(f"{c} ({uni_name}){country_str}")

                    skill_courses[skill] = list(set(courses))[:10]
            except Exception:
                skill_courses[skill] = []

        return {k: v for k, v in skill_courses.items() if v}

    def run_analysis(self, skill_threshold: float = 0.7) -> Dict[str, Any]:
        occupations = self.load_occupation_titles_from_csv()
        if not occupations:
            return {"error": "No occupations found"}

        logger.info(f"Loading required skills (threshold={skill_threshold})...")
        req_skills = self.get_required_skills(occupations, skill_threshold)

        logger.info("Loading university skills for all countries...")
        country_data = self.get_all_countries_skills()

        final_results = {}

        for country, c_skills in country_data.items():
            logger.info(f"Analyzing country: {country}")
            all_req = set()
            for s_list in req_skills.values():
                all_req.update(s_list)

            # Τι λείπει από τη χώρα
            missing = all_req - set(c_skills)

            # Προτάσεις Τμημάτων
            missing_depts = {}
            for occ, skills in req_skills.items():
                inter = set(skills).intersection(missing)
                if inter:
                    missing_depts[occ] = list(inter)

            # Προτάσεις Μαθημάτων (από άλλες χώρες)
            missing_courses = {}
            if missing:
                missing_courses = self.get_courses_from_other_countries(list(missing), country)

            final_results[country] = {
                "missing_departments": missing_depts,
                "missing_courses": missing_courses
            }

        return final_results