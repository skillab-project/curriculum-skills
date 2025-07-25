import json
import requests
import mysql.connector
from fuzzywuzzy import fuzz, process
from helpers import load_from_cache, save_to_cache, load_university_cache
from output import print_colored_text, print_horizontal_line, print_loading_line, print_horizontal_small_line, print_green_line
import os
from mysql.connector import Error


def list_available_cached_universities():
    """
    Retrieves the list of available universities from university_cache.json.
    """
    university_cache = load_university_cache()
    return list(university_cache.keys())


CACHE_DIR = 'cache'
CACHE_FILE = 'pdf_cache.json'

def save_cache(data):
    cache_file_path = os.path.join(CACHE_DIR, CACHE_FILE)
    with open(cache_file_path, 'w') as cache_file:
        json.dump(data, cache_file, indent=4)
        
def extract_skills_from_api(description):
    try:
        response = requests.post(
            "https://portal.skillab-project.eu/esco-skill-extractor/extract-skills",
            headers={"Content-Type": "application/json"},
            json=[description],
            verify=False
        )
        return response.json()[0] if response.ok else []
    except Exception as e:
        print(f"[ERROR] Skill extraction API failed: {e}")
        return []



def get_skills_for_lesson(university_name, all_data, lesson_name=None, skillname=True, db_config=None):
    results = {}
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor(dictionary=True)

        cursor.execute("SELECT university_name FROM University")
        university_list = [row['university_name'] for row in cursor.fetchall()]
        
        if university_name:
            best_match = process.extractOne(university_name, university_list)
            if best_match and best_match[1] > 80:
                university_name = best_match[0]
            else:
                return {}
        
        if university_name and lesson_name:
            query = """
                SELECT u.university_name, l.lesson_name, s.skill_name FROM Skills s
                JOIN Lessons l ON s.lesson_id = l.lesson_id
                JOIN University u ON l.university_id = u.university_id
                WHERE u.university_name LIKE %s AND l.lesson_name LIKE %s
            """
            cursor.execute(query, (university_name, f"%{lesson_name}%"))
        
        elif lesson_name:
            query = """
                SELECT u.university_name, l.lesson_name, s.skill_name FROM Skills s
                JOIN Lessons l ON s.lesson_id = l.lesson_id
                JOIN University u ON l.university_id = u.university_id
                WHERE l.lesson_name LIKE %s 
            """
            cursor.execute(query, (f"%{lesson_name}%",))
        
        elif university_name:
            query = """
                SELECT u.university_name, l.lesson_name, s.skill_name FROM Skills s
                JOIN Lessons l ON s.lesson_id = l.lesson_id
                JOIN University u ON l.university_id = u.university_id
                WHERE u.university_name LIKE %s
            """
            cursor.execute(query, (university_name,))
        
        else:
            query = """
                SELECT u.university_name, l.lesson_name, s.skill_name FROM Skills s
                JOIN Lessons l ON s.lesson_id = l.lesson_id
                JOIN University u ON l.university_id = u.university_id
            """
            cursor.execute(query)
        
        rows = cursor.fetchall()
        for row in rows:
            uni = row['university_name']
            lesson = row['lesson_name']
            skill = row['skill_name']

            if skill.lower() == "unknown skill":  
                continue
            
            if uni not in results:
                results[uni] = {}
            if lesson not in results[uni]:
                results[uni][lesson] = []
            results[uni][lesson].append(skill)
        
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
    
    return results


def extract_and_get_title(skill_url):
    try:
        if not skill_url.startswith("http://data.europa.eu/esco/skill/"):
            print("Invalid skill URL format.")
            return "Error: Invalid URL format"
        skill_id = skill_url.split('/skill/')[1]

        api_url = f"https://ec.europa.eu/esco/api/resource/skill?uri={skill_url}"
        response = requests.get(api_url)

        if response.status_code == 200:
            data = response.json()
            skill_title = data.get('preferredLabel', {}).get('en-us', None)
            return skill_title
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return "Error: Exception occurred"


def search_courses_by_skill(all_data, search_skill, db_config, university_name, threshold=52, use_cache=True):
    from database import is_database_connected 
    if not search_skill:
        print_colored_text("No skill provided for search.", 31)
        return []

    print_horizontal_line(50)
    print_colored_text(f"Searching for courses related to skill: {search_skill}", 35)

    found_courses = []

    if is_database_connected(db_config):
        print_colored_text("Database connected. Fetching skills from database...", 32)

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT semester, lesson_name, skill_name 
                FROM skills 
                WHERE LOWER(skill_name) LIKE %s
            """
            cursor.execute(query, (f"%{search_skill.lower()}%",))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            for row in results:
                similarity_score = fuzz.ratio(search_skill.lower(), row["skill_name"].lower())
                if similarity_score >= threshold:
                    found_courses.append((row["semester"], row["lesson_name"], row["skill_name"], similarity_score))

        except mysql.connector.Error as e:
            print_colored_text(f"Database error: {e}", 31)
            return []

    else:
        print_colored_text("Database not connected. Using cache instead.", 33)
        print_horizontal_line(50)
        cache = load_from_cache(university_name)
        if cache is None:
            cache = {}

        for semester, lessons in all_data.items():
            for lesson_name, lesson_data in lessons.items():
                lesson_cache = cache.get(semester, {}).get(lesson_name, {})
                lesson_skills = lesson_cache.get("skill_names", {})
                lesson_urls = lesson_cache.get("skills", {})
                lesson_description = lesson_data.get("description", "")

                if (not lesson_skills or not lesson_urls) and use_cache and lesson_description != "This lesson has no data!":
                    print_colored_text(f"Cache incomplete for '{lesson_name}'! Creating/updating now.", 32)
                    print(f"Extracting skills for lesson '{lesson_name}'...")

                    lesson_description = lesson_data.get("description", "")
                    if isinstance(lesson_description, dict):
                        lesson_description = lesson_description.get("text", "")

                    if isinstance(lesson_description, str):
                        skills_list = [extract_skills_from_api(lesson_description)]
                        new_lesson_skills = []
                        new_lesson_urls = []


                        for skill_set in skills_list:
                            for skill_url in skill_set:
                                if skill_url not in new_lesson_urls:
                                    new_lesson_urls.append(skill_url)
                                    skill_name = extract_and_get_title(skill_url)
                                    if skill_name:
                                        new_lesson_skills.append(skill_name)


                        if semester not in cache:
                            cache[semester] = {}
                        if lesson_name not in cache[semester]:
                            cache[semester][lesson_name] = {}
                        cache[semester][lesson_name]["skill_names"] = new_lesson_skills
                        cache[semester][lesson_name]["skills"] = new_lesson_urls
                        save_to_cache(university_name, cache)
                    else:
                        print(f"Warning: Description for {lesson_name} is not a string. Skipping skill extraction.")

                if isinstance(lesson_skills, dict):
                    for skill_url, skill_name in lesson_skills.items():
                        similarity_score = fuzz.ratio(search_skill.lower(), skill_name.lower())
                        if similarity_score >= threshold:
                            found_courses.append((semester, lesson_name, skill_name, similarity_score))

                elif isinstance(lesson_skills, list):
                    for skill_name in lesson_skills:
                        similarity_score = fuzz.ratio(search_skill.lower(), skill_name.lower())
                        if similarity_score >= threshold:
                            found_courses.append((semester, lesson_name, skill_name, similarity_score))


    if found_courses:
        no_limit1 = True
        no_limit2 = True
        no_limit3 = True
        for semester, lesson_name, matched_skill, score in sorted(found_courses, key=lambda x: x[3], reverse=True):
            if score >= 70 and no_limit1:
                print_colored_text(f"Most Accurate Courses (Score >= 70)", 32)
                print_horizontal_small_line(50)
                no_limit1 = False
            elif score >= 56 and score < 70 and no_limit2:
                print_horizontal_small_line(50)
                print_colored_text(f"Mediumly Accurate Courses (Score >= 56)", 33)
                print_horizontal_small_line(50)
                no_limit2 = False
            elif score >= 40 and score < 56 and no_limit3:
                print_horizontal_small_line(50)
                print_colored_text(f"Least Accurate Courses (Score >= 52)", 34)
                print_horizontal_small_line(50)
                no_limit3 = False
            print(f" {lesson_name} | Matched Skill: {matched_skill} (Score: {score})")
        print_horizontal_line(50)
    else:
        print_horizontal_line(50)
        print_colored_text("No closely matching courses found.", 31)
        print_horizontal_line(50)

    return found_courses




def search_courses_by_skill_database(search_skill, db_config, university_name=None, threshold=52):
    from collections import Counter
    from database import is_database_connected
    if not search_skill:
        print("No skill provided for search.")
        return {}

    print("-" * 50)
    print(f"Searching for courses related to skill: {search_skill}")

    found_courses = {}
    skill_frequency = Counter()

    if not is_database_connected(db_config):
        print("Database not connected. Unable to fetch results.")
        return {}

    print("Database connected. Fetching skills from database...")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT u.university_name, l.semester, l.lesson_name, s.skill_name 
            FROM Skills s
            JOIN Lessons l ON s.lesson_id = l.lesson_id
            JOIN University u ON l.university_id = u.university_id
            WHERE LOWER(s.skill_name) LIKE %s
        """

        cursor.execute(query, [f"%{search_skill.lower()}%"])
        results = cursor.fetchall()

    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        return {}

    finally:
        cursor.close()
        conn.close()

    if university_name:
        matched_universities = {
            row["university_name"]: fuzz.ratio(university_name.lower(), row["university_name"].lower())
            for row in results
        }
        matched_universities = {k: v for k, v in matched_universities.items() if v >= threshold}

        if not matched_universities:
            print("No closely matching universities found.")
            return {}

        best_match = max(matched_universities, key=matched_universities.get)
        results = [row for row in results if row["university_name"] == best_match]

    for row in results:
        university = row["university_name"]
        semester = row["semester"]
        lesson = row["lesson_name"]
        skill = row["skill_name"]

        if fuzz.ratio(search_skill.lower(), skill.lower()) >= threshold:
            skill_frequency[skill] += 1

            if university not in found_courses:
                found_courses[university] = {}
            if semester not in found_courses[university]:
                found_courses[university][semester] = {}
            if lesson not in found_courses[university][semester]:
                found_courses[university][semester][lesson] = []

            found_courses[university][semester][lesson].append({
                "skill": skill,
                "score": fuzz.ratio(search_skill.lower(), skill.lower()),
                "frequency": skill_frequency[skill]
            })

    if found_courses:
        print(found_courses)
    else:
        print("No closely matching courses found.")

    print("-" * 50)
    return found_courses


def search_courses_by_skill_url(search_skill_url, db_config, university_name=None):
    from collections import Counter
    from database import is_database_connected
    import mysql.connector

    if not search_skill_url:
        print("No skill URL provided for search.")
        return {}

    print("-" * 50)
    print(f"Searching for universities offering courses related to skill URL: {search_skill_url}")
    
    found_courses = {}
    skill_frequency = Counter()

    if is_database_connected(db_config):
        print("Database connected. Fetching skills from database...")

        try:
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT u.university_name, l.semester, l.lesson_name, s.skill_name, s.skill_url 
                FROM Skills s
                JOIN Lessons l ON s.lesson_id = l.lesson_id
                JOIN University u ON l.university_id = u.university_id
                WHERE s.skill_url = %s
            """
            
            params = [search_skill_url]
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            conn.close()

            if university_name:
                results = [row for row in results if row["university_name"].lower() == university_name.lower()]
                if not results:
                    print("No exact matching universities found.")
                    return {}
            
            for row in results:
                university = row["university_name"]
                semester = row["semester"]
                lesson = row["lesson_name"]
                skill = row["skill_name"]
                
                skill_frequency[skill] += 1
                
                if university not in found_courses:
                    found_courses[university] = {}
                if semester not in found_courses[university]:
                    found_courses[university][semester] = {}
                if lesson not in found_courses[university][semester]:
                    found_courses[university][semester][lesson] = []
                
                found_courses[university][semester][lesson].append({"skill": skill, "skill_url": search_skill_url, "frequency": skill_frequency[skill]})

        except mysql.connector.Error as e:
            print(f"Database error: {e}")
            return {}
    else:
        print("Database not connected. Unable to fetch results.")
        return {}

    if found_courses:
        print(found_courses)
        print("-" * 50)
    else:
        print("-" * 50)
        print("No exact matching courses found.")
        print("-" * 50)

    return found_courses