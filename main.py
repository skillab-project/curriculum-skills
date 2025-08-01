from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from database import write_to_database, is_database_connected
from skills import get_skills_for_lesson, search_courses_by_skill, search_courses_by_skill_database, extract_and_get_title, search_courses_by_skill_url
from pdf_utils import extract_text_from_pdf, split_by_semester, process_pages_by_lesson, extract_text_after_marker
from config import DB_CONFIG
from collections import Counter, defaultdict
import os
import json
from helpers import find_possible_university, load_from_cache, save_to_cache, load_university_cache, save_cache
from skillcrawl import get_university_country
from typing import List
from fuzzywuzzy import process, fuzz
from fastapi import UploadFile
from typing import Dict, Optional
import re
from crawler import UniversityCrawler
from collections import OrderedDict, defaultdict
import psycopg2
import mysql.connector
from concurrent.futures import ThreadPoolExecutor
import requests
from datetime import datetime
from math import ceil
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Set
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from urllib.parse import quote_plus

from dotenv import load_dotenv
load_dotenv()


UNI_FILE = "university_cache.json"
os.makedirs("cache/occupations", exist_ok=True)

if os.path.exists(UNI_FILE):
    with open(UNI_FILE, "r") as f:
        university_cache = json.load(f)
else:
    university_cache = {}

app = FastAPI(
    title="SkillCrawl API",
    description="API for skill extraction and course search.",
    root_path="/curriculum-skills"
)


class Organization(BaseModel):
    name: str
    location: str

class ExportItem(BaseModel):
    id: List[int]
    title: List[str]
    description: List[str]
    skills: List[List[str]]
    occupations: List[List[str]]
    upload_date: List[str]
    organization: Organization

class ExportResponse(BaseModel):
    items: List[ExportItem]


class CountryUniversities(BaseModel):
    country: str
    universities: int

class SkillPerCountry(BaseModel):
    skill: str
    frequency: int

class SkillsByCountry(BaseModel):
    country: str
    skills: List[SkillPerCountry]

class MonthlyTrend(BaseModel):
    month: str
    count: int

class CountryTrend(BaseModel):
    country: str
    monthly_counts: List[MonthlyTrend]


class ClusterResult(BaseModel):
    cluster: int
    universities: List[Dict[str, object]]
    
class SkillFrequency(BaseModel):
    skill: str
    frequency: int


class CrawlRequest(BaseModel):
    url: str

class SkillListRequest(BaseModel):
    skills: List[str]

class PDFProcessingRequest(BaseModel):
    pdf_name: str

class SkillSearchRequest(BaseModel):
    skill: str
    university: str = None 

class SkillSearchURLRequest(BaseModel):
    skill_url: str
    university: str = None 

class LessonRequest(BaseModel):
    university_name: str
    lesson_name: str

class TopSkillsAllRequest(BaseModel):
    top_n: Optional[int] = 20

class TopSkillsRequest(BaseModel):
    university_name: str
    top_n: Optional[int] = 20


def get_tracker_token() -> str:
    login_url = "https://skillab-tracker.csd.auth.gr/api/login"
    payload = {
        "username": os.getenv("TRACKER_USERNAME"),
        "password": os.getenv("TRACKER_PASSWORD")
    }
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json"
    }

    try:
        response = requests.post(login_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()

        token = response.text.strip().strip('"')
        if not token:
            print(f"[ERROR] Empty token returned from login")
        else:
            print(f"[INFO] Successfully obtained token: {token[:15]}...")

        return token

    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        return ""



@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "running"}

@app.get("/list_pdfs", tags=["PDF"])
def list_pdfs():

    university_cache = load_university_cache()
    curriculum_folder = "curriculum"
    if not os.path.exists(curriculum_folder):
        os.makedirs(curriculum_folder) 
    
    pdf_files = []
    
    for f in os.listdir(curriculum_folder):
        if f.endswith(".pdf"):
            cached_data = load_from_cache(f) or {}

            university_name = cached_data.get("university_name", "").strip()
            university_country = cached_data.get("university_country", "").strip()

            if not university_name or "unknown" in university_name.lower():
                filename = f.replace(".pdf", "")
                university_name = re.sub(r"[_\W]+", " ", filename).strip()
                print(f"Extracted university name from filename in /list_pdfs: {university_name}") 

            university_country = get_university_country(university_name) if university_name else "Unknown"

            pdf_files.append({
                "filename": f,
                "university_name": university_name,
                "university_country": university_country
            })

            university_cache[university_name] = {
                "name": university_name,
                "country": university_country,
                "pdf_file": f
            }
            save_cache()
    
    return {"pdf_files": pdf_files}

@app.post("/process_pdf", tags=["PDF"])
def process_pdf(request: PDFProcessingRequest):

    university_cache = load_university_cache()
    print(f"Received request to process PDF: {request.pdf_name}")
    
    if os.path.isabs(request.pdf_name) and os.path.exists(request.pdf_name):
        pdf_path = request.pdf_name
    else:
        curriculum_folder = "curriculum"
        os.makedirs(curriculum_folder, exist_ok=True)
        matching_files = [f for f in os.listdir(curriculum_folder) if f.endswith(".pdf") and request.pdf_name in f]
        if not matching_files:
            print(f"No PDF matching '{request.pdf_name}' found in 'curriculum/'. Possibly running a test case?")
        pdf_path = os.path.join(curriculum_folder, matching_files[0])

    pages = extract_text_from_pdf(pdf_path)

    cached_data = load_from_cache(pdf_path) or {}
    university_name = cached_data.get("university_name", "").strip()
    university_country = cached_data.get("university_country", "").strip()

    if not university_name or "unknown" in university_name.lower():
        university_name = re.sub(r"[_\W]+", " ", os.path.basename(pdf_path).replace(".pdf", "")).strip()
        print(f"✅ Extracted university name: {university_name}")
        save_cache()


    university_cache = load_university_cache()
    if university_name not in university_cache or university_country not in university_cache:
        university_country = get_university_country(university_name) if university_name else "Unknown"
        university_cache[university_name] = {"name": university_name, "country": university_country}



    marker = ['Course Outlines', 'Course Content']
    text_after_marker = extract_text_after_marker(pages, marker)
    semesters = split_by_semester(text_after_marker)
    all_data = {}

    if semesters:
        for i, semester_text in enumerate(semesters, 1):
            lessons = process_pages_by_lesson([page for page in pages if page in semester_text])
            all_data[f"Semester {i} ({len(lessons)} lessons)"] = {
                lesson: {"description": desc, "skills": list({s for skill_set in [s.get("id") for s in requests.post(
              "https://portal.skillab-project.eu/esco-skill-extractor/extract-skills",
              headers={"accept": "application/json", "Content-Type": "application/json"},
              json={"description": [desc]},
              verify=False
          ).json().get("items", [])] for s in skill_set})}
                for lesson, desc in lessons.items()
            }
    else:
        lessons = process_pages_by_lesson(pages)
        all_data["Lessons Only"] = {
            esson: {"description": desc, "skills": list({s for skill_set in [s.get("id") for s in requests.post(
          "https://portal.skillab-project.eu/esco-skill-extractor/extract-skills",
          headers={"accept": "application/json", "Content-Type": "application/json"},
          json={"description": [desc]},
          verify=False
      ).json().get("items", [])] for s in skill_set})}
            for lesson, desc in lessons.items()
        }

    university_country = get_university_country(university_name) if university_name else "Unknown"

    all_data.update({"university_name": university_name, "university_country": university_country})
    save_cache()
    save_to_cache(university_name, all_data)

    return {"message": "PDF processed successfully.", "data": all_data}

CACHE_DIR = "cache"  

def load_all_cached_data():
    """
    Searches through the cache folder and loads data for all universities.
    Returns a dictionary where keys are university names, and values are their cached lessons.
    """
    all_data = {}

    if not os.path.exists(CACHE_DIR):
        print("⚠️ Cache directory does not exist.")
        return {}

    for filename in os.listdir(CACHE_DIR):
        if filename.endswith(".json"): 
            university_name = filename.replace(".json", "")
            try:
                with open(os.path.join(CACHE_DIR, filename), "r", encoding="utf-8") as file:
                    all_data[university_name] = json.load(file)
            except json.JSONDecodeError:
                print(f"❌ Failed to load cache for {university_name}. Skipping...")
    
    return all_data 

@app.post("/filter_skillnames", tags=["Skills"])
def get_skills(request: LessonRequest):
    """
    API endpoint to get skill names based on university and lesson name.
    - First checks cache.
    - If not in cache, searches the database.
    - If missing in database, writes from cache to the database.
    """


    if request.university_name:
        all_data = load_from_cache(request.university_name)
    else:
        all_data = load_all_cached_data() 


    skills = get_skills_for_lesson(request.university_name, all_data, request.lesson_name, skillname=True, db_config=DB_CONFIG)

    return {"Data": skills or []} 



@app.post("/calculate_skillnames", tags=["Skills"])
def calculate_skillnames(university_name: str, lesson_name: Optional[str] = None):
    token = get_tracker_token()
    all_cached_data = load_all_cached_data()
    university_names = [name.replace("_cache", "").strip() for name in all_cached_data.keys()]

    if not university_names:
        raise HTTPException(status_code=404, detail="No universities found in cache.")

    best_match, score = process.extractOne(university_name, university_names)

    if score < 70:
        raise HTTPException(status_code=404, detail=f"No close match found for university '{university_name}'.")

    print(f"[INFO] Matched university '{university_name}' -> '{best_match}' with score {score}")

    university_key = next((key for key in all_cached_data.keys() if best_match in key), best_match)
    cached_data = all_cached_data[university_key]
    extracted_skills = {}

    university_name = university_key.replace("_cache", "").strip()

    skillab_tracker_url = "https://skillab-tracker.csd.auth.gr/api/track_skills"


    def process_lesson(semester, lesson, lesson_data):
        """ Extracts skills for a lesson using Skillab Tracker API """
        try:
            print(f"[INFO] Processing skills for: {lesson} in {semester}")

            lesson_cache = cached_data.get(semester, {}).get(lesson, {})
            cached_skill_names = lesson_cache.get("skill_names", [])
            cached_skills = lesson_cache.get("skills", [])

            lesson_description = lesson_data.get("description", "")
            if isinstance(lesson_description, dict):
                lesson_description = lesson_description.get("text", "")
            if not isinstance(lesson_description, str) or not lesson_description.strip():
                print(f"[WARNING] No valid description for {lesson}. Skipping skill extraction.")
                return lesson, cached_skill_names

            skills_list = requests.post(
              "https://portal.skillab-project.eu/esco-skill-extractor/extract-skills",
              headers={"Content-Type": "application/json"},
              json=[lesson_description],
              verify=False
          ).json()

            skill_urls = set()

            for skill_set in skills_list:
                for skill_url in skill_set:
                    skill_urls.add(skill_url)

            if not skill_urls:
                print(f"[WARNING] No skills found for {lesson}. Skipping API call.")
                return lesson, cached_skill_names

            skill_params = "&".join([f"ids={requests.utils.quote(skill_url)}" for skill_url in skill_urls])

            skillab_tracker_url = "https://skillab-tracker.csd.auth.gr/api/skills?page=1"
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {token}"
            }
            response = requests.post(skillab_tracker_url, headers=headers, data=skill_params, verify=False)

            if response.status_code != 200:
                print(f"[ERROR] Skillab API failed for {lesson}. Response: {response.status_code} - {response.text}")
                return lesson, cached_skill_names

            skill_data = response.json()
            new_skills = OrderedDict()

            for skill in skill_data.get("items", []):
                skill_url = skill.get("id")
                skill_name = skill.get("label") or (skill.get("alternative_labels")[0] if skill.get("alternative_labels") else "Unknown Skill")

                new_skills[skill_url] = skill_name

            sorted_skills = OrderedDict(sorted(new_skills.items(), key=lambda x: x[1]))

            for skill_url, skill_name in sorted_skills.items():
                if skill_name not in cached_skill_names:
                    cached_skill_names.append(skill_name)
                if skill_url not in cached_skills:
                    cached_skills.append(skill_url)

            cached_data[semester][lesson]["skill_connect"] = sorted_skills
            cached_data[semester][lesson]["skills"] = cached_skills
            cached_data[semester][lesson]["skill_names"] = cached_skill_names

            print(f"[INFO] Successfully extracted {len(sorted_skills)} skills for {lesson}")
            return lesson, cached_skill_names

        except Exception as e:
            print(f"[ERROR] Failed to process lesson '{lesson}' in '{semester}': {e}")
            return lesson, []


    lesson_tasks = []

    with ThreadPoolExecutor() as executor:
        for semester, lessons in cached_data.items():
            if semester in ["university_name", "university_country"]:
                continue

            if lesson_name:
                best_lesson_match, lesson_score = process.extractOne(lesson_name, list(lessons.keys()))

                if lesson_score < 80:
                    raise HTTPException(status_code=404, detail=f"No close match found for lesson '{lesson_name}'.")

                print(f"[INFO] Matched lesson '{lesson_name}' -> '{best_lesson_match}' with score {lesson_score}")

                selected_lessons = {best_lesson_match: lessons[best_lesson_match]}
            else:
                selected_lessons = lessons 

            for lesson, lesson_data in selected_lessons.items():
                lesson_tasks.append(executor.submit(process_lesson, semester, lesson, lesson_data))

    for future in lesson_tasks:
        lesson, skills = future.result()
        extracted_skills[lesson] = skills

    save_to_cache(university_name, cached_data)

    return {"university_name": university_name, "skills": extracted_skills}


@app.get("/search_skill", tags=["Queries"])
def search_skill(
    skill: str = Query(..., description="Skill name"),
    university: Optional[str] = Query(None, description="University name")
):
    """
    You can search if a skill exists in the database.
    In return:
    - University name(s) the skill is in
    - Lesson(s) the skill is in
    - And frequency of appearance
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    results = search_courses_by_skill_database(skill, DB_CONFIG, university)
    return {"results": results}

@app.get("/search_skill_by_URL", tags=["Queries"])
def search_skill_by_url(
    skill_url: str = Query(..., description="Skill URL"),
    university: Optional[str] = Query(None, description="University name")
):
    """
    You can search if a skill exists in the database.
    In return:
    - University name(s) the skill is in
    - Lesson(s) the skill is in
    - And frequency of appearance
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    results = search_courses_by_skill_url(skill_url, DB_CONFIG, university)
    return {"results": results}


@app.get("/get_universities_by_skills", tags=["Queries"])
def get_universities_by_skills(
    skills: List[str] = Query(..., description="List of skills to search for")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    query = """
        SELECT u.university_name, l.lesson_name, s.skill_name 
        FROM Skills s
        JOIN Lessons l ON s.lesson_id = l.lesson_id
        JOIN University u ON l.university_id = u.university_id
        WHERE s.skill_name IN (%s)
    """
    
    skill_placeholders = ', '.join(['%s'] * len(skills))
    formatted_query = query.replace("%s", skill_placeholders)
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(formatted_query, tuple(skills))
        results = cursor.fetchall()
    
        university_skill_counts = {}
        university_courses = {}
        
        for row in results:
            uni = row["university_name"]
            lesson = row["lesson_name"]
            skill = row["skill_name"]
            
            if uni not in university_skill_counts:
                university_skill_counts[uni] = set()
            university_skill_counts[uni].add(skill)
            
            if uni not in university_courses:
                university_courses[uni] = {}
            if lesson not in university_courses[uni]:
                university_courses[uni][lesson] = []
            university_courses[uni][lesson].append(skill)
        
        filtered_universities = {
            uni: courses for uni, courses in university_courses.items()
            if len(university_skill_counts[uni]) == len(skills)
        }
    
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    finally:
        cursor.close()
        conn.close()
    
    return filtered_universities

from fastapi import Query

@app.get("/get_top_skills", tags=["Queries"])
def get_top_skills(
    university_name: str = Query(..., description="University name"),
    top_n: int = Query(20, description="Number of top skills to return")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    query = """
    SELECT s.skill_name
    FROM Skills s
    JOIN Lessons l ON s.lesson_id = l.lesson_id
    JOIN University u ON l.university_id = u.university_id
    WHERE u.university_name = %s
      AND s.skill_name IS NOT NULL
      AND s.skill_name != ''
      AND s.skill_name NOT LIKE 'Unknown%'
    """
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (university_name,))
        results = cursor.fetchall()
        
        skill_counter = Counter(row["skill_name"] for row in results)
        top_skills = [{"skill": skill, "frequency": count} for skill, count in skill_counter.most_common(top_n)]
        
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    finally:
        cursor.close()
        conn.close()
    
    return {"university_name": university_name, "top_skills": top_skills}

@app.get("/get_top_skills_all", tags=["Queries"])
def get_top_skills_all(
    top_n: int = Query(20, description="Number of top skills to return")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    
    query = """
        SELECT s.skill_name, u.university_name
        FROM Skills s
        JOIN Lessons l ON s.lesson_id = l.lesson_id
        JOIN University u ON l.university_id = u.university_id
    """
    
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        
        skill_counter = Counter(row["skill_name"] for row in results)
        university_skill_map = defaultdict(set)
        
        for row in results:
            university_skill_map[row["skill_name"].lower()].add(row["university_name"])
        
        top_skills = [{
            "skill": skill,
            "frequency": count,
            "universities": list(university_skill_map[skill.lower()])
        } for skill, count in skill_counter.most_common(top_n)]
        
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    
    finally:
        cursor.close()
        conn.close()
    
    return {"top_skills": top_skills}



@app.get("/search_json_in_cache", tags=["Cache"])
def search_json_in_cache(university_name: str):
    """
    Searches for a cached JSON file based on a fuzzy match of the university name.
    """
    cache_folder = "cache"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder) 
    
    json_files = [f for f in os.listdir(cache_folder) if f.endswith(".json")]
    
    if not json_files:
        raise HTTPException(status_code=404, detail="No cached files found.")

    best_match, score = process.extractOne(university_name, json_files)

    if score < 60: 
        raise HTTPException(status_code=404, detail=f"No close match found for university: {university_name}")

    file_path = os.path.join(cache_folder, best_match)
    with open(file_path, "r", encoding="utf-8") as file:
        cached_data = json.load(file)

    return {
        "message": "Cached file found.",
        "matched_file": best_match,
        "match_score": score,
        "data": cached_data
    }


@app.post("/save_to_db", tags=["Database"])
def save_to_db(university_name: str):
    """
    Searches for a cached JSON file using fuzzy matching and saves its data to the database.
    """

    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    cache_folder = "cache"
    
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
    
    json_files = [f for f in os.listdir(cache_folder) if f.endswith(".json")]
    
    if not json_files:
        raise HTTPException(status_code=404, detail="No cached files found.")

    print(f"Available cached files: {json_files}")

    best_match, score = process.extractOne(university_name, json_files)

    print(f"Fuzzy match result: {best_match} (score: {score})") 

    if score < 60: 
        raise HTTPException(status_code=404, detail=f"No close match found for university: {university_name}")

    file_path = os.path.join(cache_folder, best_match)
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail=f"Corrupted or invalid JSON file: {best_match}")

    university_name = data.get("university_name", "").strip()
    university_country = data.get("university_country", "").strip()
    number_of_semesters = len(data) - 2 

    if not university_name or not university_country:
        raise HTTPException(status_code=400, detail="Missing university name or country in the cached data.")
    
    write_to_database(data, DB_CONFIG, university_name, university_country, number_of_semesters)

    return {
        "message": "Data saved to database successfully.",
        "matched_file": best_match,
        "match_score": score
    }

CACHE_FOLDER = "cache"

@app.get("/all_university_data", tags=["Database"])
def get_all_data(
    university_name: str = Query(..., description="Full or partial university name")
):
    """
    Fetch all university-related data, including lessons and skills, from the MySQL database.
    If the database is offline, raise an error immediately.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT 
        u.university_name,
        u.country,
        u.number_of_semesters,
        l.lesson_name,
        l.semester,
        l.description,
        s.skill_name,
        s.skill_url
    FROM University u
    LEFT JOIN Lessons l ON u.university_id = l.university_id
    LEFT JOIN Skills s ON l.lesson_id = s.lesson_id
    WHERE u.university_name LIKE %s
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  
        cursor.execute(query, (f"%{university_name}%",))
        results = cursor.fetchall()

        if not results:
            raise HTTPException(status_code=404, detail=f"No data found for university: {university_name}")

        university_data = {
            "university_name": results[0]["university_name"],
            "country": results[0]["country"],
            "number_of_semesters": results[0]["number_of_semesters"],
            "semesters": {}
        }

        for row in results:
            semester = row["semester"]
            lesson_name = row["lesson_name"]
            description = row["description"]
            skill_name = row["skill_name"]
            skill_url = row["skill_url"]

            if semester:
                if semester not in university_data["semesters"]:
                    university_data["semesters"][semester] = {}

                if lesson_name:
                    if lesson_name not in university_data["semesters"][semester]:
                        university_data["semesters"][semester][lesson_name] = {
                            "description": description,
                            "skills": []
                        }

                    if skill_name:
                        university_data["semesters"][semester][lesson_name]["skills"].append(
                            {"name": skill_name, "url": skill_url}
                        )

        return university_data

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        cursor.close()
        conn.close()



@app.post("/save_all_to_db", tags=["Database"])
def save_all_to_db():
    """
    Dynamically finds JSON files in the cache folder and saves their contents to the database.
    Process is sped up with Cache files.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    json_files = [f for f in os.listdir(CACHE_FOLDER) if f.endswith(".json") and f != "pdf_cache.json"]

    if not json_files:
        raise HTTPException(status_code=404, detail="No valid university data found in cache.")

    for json_file in json_files:
        json_path = os.path.join(CACHE_FOLDER, json_file)

        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            university_name = data.get("university_name", "").replace("_cache", "").strip()
            university_country = data.get("university_country", "")
            number_of_semesters = len([key for key in data.keys() if key not in ["university_name", "university_country"]])

            if not university_name or not university_country:
                print(f"[WARNING] Skipping {json_file}: Missing university name or country.")
                continue  

            write_to_database(data, DB_CONFIG, university_name, university_country, number_of_semesters)
            print(f"[INFO] Saved {university_name} to database.")

        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse {json_file}. Skipping.")
            continue

    return {"message": "All valid university data saved to the database successfully."}


@app.get("/descriptive/location", response_model=List[CountryUniversities], tags=["Descriptive"], summary="Universities per Country from DB")
def get_university_counts_by_country_db():
    """
    Returns number of universities per country using DB data.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT country, COUNT(*) AS universities
    FROM University
    WHERE country IS NOT NULL AND country != ''
    GROUP BY country
    ORDER BY universities DESC
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        return cursor.fetchall()
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/exploratory/skills_location", response_model=List[SkillsByCountry], tags=["Exploratory"], summary="Skills per Country from DB")
def get_skills_per_location_db():
    """
    Country list with each country's skills and their frequency
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT u.country, s.skill_name, COUNT(*) AS frequency
    FROM Skills s
    JOIN Lessons l ON s.lesson_id = l.lesson_id
    JOIN University u ON l.university_id = u.university_id
    WHERE s.skill_name IS NOT NULL AND s.skill_name != '' AND u.country IS NOT NULL
    GROUP BY u.country, s.skill_name
    ORDER BY u.country, frequency DESC
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()

        grouped = defaultdict(list)
        for row in rows:
            grouped[row["country"]].append(SkillPerCountry(skill=row["skill_name"], frequency=row["frequency"]))

        return [{"country": country, "skills": skills} for country, skills in grouped.items()]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/trend/location", response_model=List[CountryTrend], tags=["Trend"], summary="University Join Trend per Country from DB")
def get_university_trend_per_location_db():
    """
    Number of universities added per month grouped by country.
    If there's no actual date, it is based on created_at timestamps.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT country, DATE_FORMAT(created_at, '%Y-%m') AS month, COUNT(*) AS count
    FROM University
    WHERE country IS NOT NULL AND created_at IS NOT NULL
    GROUP BY country, month
    ORDER BY country, month
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()

        grouped = defaultdict(list)
        for row in rows:
            grouped[row["country"]].append(MonthlyTrend(month=row["month"], count=row["count"]))

        return [{"country": country, "monthly_counts": counts} for country, counts in grouped.items()]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()



@app.get("/cluster/universities/{k}", response_model=List[ClusterResult], tags=["Clustering"], summary="University Clustering by Skills from DB")
def cluster_universities_db(k: int):
    """
    Clustering universities based on skill profiles, also returning 2D coordinates for plotting.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT u.university_name, GROUP_CONCAT(DISTINCT s.skill_name SEPARATOR ' ') AS skills
    FROM University u
    JOIN Lessons l ON u.university_id = l.university_id
    JOIN Skills s ON l.lesson_id = s.lesson_id
    WHERE s.skill_name IS NOT NULL AND s.skill_name != ''
    GROUP BY u.university_name
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()

        if not rows:
            return []

        universities = [row["university_name"] for row in rows]
        documents = [row["skills"] for row in rows]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(documents)

        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(X.toarray())

        grouped = defaultdict(list)
        for idx, label in enumerate(kmeans.labels_):
            grouped[int(label)].append({
                "pref_label": universities[idx],
                "x": round(float(coords_2d[idx][0]), 3),
                "y": round(float(coords_2d[idx][1]), 3)
            })

        return [{"cluster": cluster, "universities": unis} for cluster, unis in grouped.items()]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()



@app.get("/descriptive/skills_frequency", response_model=List[SkillFrequency], tags=["Descriptive"], summary="Skill frequency across lessons from DB")
def get_skill_frequencies():
    """
    Returns a list of all skills and how many times each appears across lessons.
    Useful for visualizations like bar plots.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
        SELECT skill_name, COUNT(*) AS frequency
        FROM Skills
        WHERE skill_name IS NOT NULL AND skill_name != ''
        GROUP BY skill_name
        ORDER BY frequency DESC
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()

        return [{"skill": row["skill_name"], "frequency": row["frequency"]} for row in results]

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        cursor.close()
        conn.close()

@app.get("/bilateral/labor_market_export", response_model=ExportResponse, tags=["Bilateral"], summary="List of all degrees with their Level 4 skills")
def labor_export_from_database(
    university_name: str = Query(None, description="Optional university name to search for"),
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    limit: int = Query(100, ge=1, le=1000, description="Number of items per page")
):
    try:
        occupation_cache = load_from_cache("occupations/occupations") or {}
        new_entries_added = False

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query_filter = ""
        params = ()

        if university_name:
            cursor.execute("SELECT DISTINCT university_name FROM University")
            all_unis = [row["university_name"] for row in cursor.fetchall()]

            match = process.extractOne(university_name, all_unis, scorer=fuzz.token_sort_ratio)
            if not match or match[1] < 70:
                return JSONResponse(status_code=404, content={"error": "University not found with sufficient confidence"})
            matched_name = match[0]
            query_filter = "WHERE u.university_name = %s"
            params = (matched_name,)

        query = f"""
            SELECT 
                u.university_name,
                u.country,
                u.created_at,
                l.lesson_id,
                l.lesson_name,
                l.description,
                s.skill_url,
                s.skill_name
            FROM University u
            JOIN Lessons l ON u.university_id = l.university_id
            LEFT JOIN Skills s ON l.lesson_id = s.lesson_id
            {query_filter}
            ORDER BY u.university_id, l.lesson_id
        """
        cursor.execute(query, params)
        rows = cursor.fetchall()

        lessons_data = defaultdict(lambda: {
            "university": None,
            "country": None,
            "upload_date": None,
            "description": "",
            "skills": []
        })

        for row in rows:
            key = row["lesson_id"]
            lessons_data[key]["title"] = row["lesson_name"]
            lessons_data[key]["description"] = row.get("description", "") or ""
            lessons_data[key]["university"] = row["university_name"]
            lessons_data[key]["country"] = row["country"]
            lessons_data[key]["upload_date"] = row["created_at"].strftime("%Y-%m-%d") if row.get("created_at") else datetime.today().strftime("%Y-%m-%d")
            if row["skill_url"]:
                lessons_data[key]["skills"].append((row["skill_url"], row.get("skill_name", "")))

        export_items = []

        for lesson_id, lesson in lessons_data.items():
            skill_names = [name for _, name in lesson["skills"] if name]
            occupations = []
            seen_occupations = set()
            token = get_tracker_token()

            for skill_name in skill_names:
                if skill_name in occupation_cache:
                    occupation_list = occupation_cache[skill_name]
                else:
                    payload = [("keywords", skill_name)] + [
                        ("keywords_logic", "or"),
                        ("children_logic", "or"),
                        ("ancestors_logic", "or")
                    ]
                    headers = {
                        "accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Authorization": f"Bearer {token}"
                    }

                    try:
                        response = requests.post(
                            "https://skillab-tracker.csd.auth.gr/api/occupations?page=1",
                            headers=headers,
                            data=payload,
                            verify=False
                        )
                        occupation_list = []
                        if response.ok:
                            for item in response.json().get("items", []):
                                occ_id = item.get("id")
                                occ_label = item.get("label")
                                if occ_id and occ_id not in seen_occupations:
                                    seen_occupations.add(occ_id)
                                    occupation_list.append([occ_id, occ_label])
                        else:
                            print(f"[WARNING] Occupation API failed: {response.status_code} - {response.text}")
                        occupation_cache[skill_name] = occupation_list
                        new_entries_added = True
                    except Exception as e:
                        print(f"[ERROR] Skill extraction failed for lesson: {description[:30]}")
                        occupation_cache[skill_name] = []
                        occupation_list = []

                occupations += [o for o in occupation_cache.get(skill_name, []) if o[0] not in seen_occupations]
                seen_occupations.update([o[0] for o in occupation_cache.get(skill_name, [])])

            export_items.append({
                "id": [lesson_id],
                "title": [lesson["title"]],
                "description": [lesson["description"]],
                "skills": [[url, name] for url, name in lesson["skills"]],
                "occupations": occupations,
                "upload_date": [lesson["upload_date"]],
                "organization": {
                    "name": lesson["university"],
                    "location": lesson["country"]
                }
            })

        if new_entries_added:
            os.makedirs("cache/occupations", exist_ok=True)
            print("[INFO] Saving updated occupation cache to cache/occupations/occupations_cache.json")
            save_to_cache("occupations/occupations", occupation_cache)

        offset = (page - 1) * limit
        paginated_items = export_items[offset:offset + limit]

        return ExportResponse(items=paginated_items)

    except mysql.connector.Error as e:
        return JSONResponse(status_code=500, content={"error": f"Database error: {str(e)}"})

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


@app.get("/bilateral/biodiversity_analysis", tags=["Bilateral"])
def biodiversity_analysis(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)")
):

    cached_data = load_from_cache("biodiversity/bio")
    if cached_data:
        total = len(cached_data)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page),
            "results": cached_data[start:end]
        }
        
    
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT 
        u.university_name, u.country,
        l.lesson_id, l.lesson_name, l.semester, l.description,
        s.skill_url, s.skill_name
    FROM University u
    JOIN Lessons l ON u.university_id = l.university_id
    LEFT JOIN Skills s ON l.lesson_id = s.lesson_id
    """

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"[INFO] Retrieved {len(rows)} rows from database")

        university_map = defaultdict(lambda: defaultdict(lambda: {
            "BSc": set(), "MSc": set()
        }))
        skill_url_map = {}

        for row in rows:
            uni = row["university_name"]
            country = row["country"]
            skill_url = row["skill_url"]
            skill_name = row["skill_name"]
            lesson = row["lesson_name"] or ""
            description = row["description"] or ""

            degree_type = "MSc" if re.search(r"master|msc", lesson, re.IGNORECASE) else "BSc"
            degree_title = "Computer Science"

            if skill_url:
                university_map[(country, uni)][degree_title][degree_type].add((skill_url, skill_name))
                skill_url_map[skill_url] = skill_name

        print(f"[INFO] Collected {len(skill_url_map)} unique skills across all courses")

        def get_level4_skill_ids(skill_ids: List[str]) -> Set[str]:
            token = get_tracker_token()
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {token}"
            }
            data = [("ids", sid) for sid in skill_ids]
            data.append(("min_skill_level", "4"))
            data.append(("max_skill_level", "4"))

            try:
                response = requests.post(
                    "https://skillab-tracker.csd.auth.gr/api/skills",
                    headers=headers,
                    data=data,
                    verify=False
                )
                response.raise_for_status()
                items = response.json().get("items", [])

                print(f"[DEBUG] API returned {len(items)} Level 4 skills from {len(skill_ids)} submitted")
                return {item["id"] for item in items}
            except Exception as e:
                print(f"[ERROR] Failed to fetch skill levels: {e}")
                print(f"[DEBUG] Failed batch: {skill_ids}")
                return set()

        level4_skill_ids = set()
        all_skill_urls = list(skill_url_map.keys())

        for i in range(0, len(all_skill_urls), 50):
            batch = all_skill_urls[i:i + 50]
            level4_skill_ids.update(get_level4_skill_ids(batch))

        print(f"[INFO] Found {len(level4_skill_ids)} Level 4 skills")

        results = []

        for (country, uni), degrees in university_map.items():
            for degree_title, degree_levels in degrees.items():
                for degree_type, skill_pairs in degree_levels.items():
                    level4_skills = [
                        skill_url_map[url]
                        for (url, _) in skill_pairs
                        if url in level4_skill_ids
                    ]
                    if level4_skills:
                        print(f"[DEBUG] Including {degree_type} {degree_title} at {uni} ({country}) with {len(level4_skills)} Level 4 skills")
                        results.append({
                            "country": country,
                            "university": uni,
                            "department": "Computer Science",
                            "degree": {
                                "type": degree_type,
                                "title": degree_title
                            },
                            "skills": sorted(set(level4_skills))
                        })
                    else:
                        print(f"[SKIP] {uni} ({country}) - {degree_type} {degree_title} has no Level 4 skills")

        print(f"[INFO] Final result contains {len(results)} entries")

        total = len(results)
        start = (page - 1) * per_page
        end = start + per_page
        paginated_results = results[start:end]
        
        save_to_cache("biodiversity/bio", results)

        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page),
            "results": paginated_results
        }

    finally:
        cursor.close()
        conn.close()

def update_lesson_info(conn, university_name: str, lesson_name: str, msc_bsc: str, degree_title: str):
    try:
        cursor = conn.cursor()
        query = """
        UPDATE Lessons l
        JOIN University u ON l.university_id = u.university_id
        SET l.msc_bsc = %s, l.degree_title = %s
        WHERE u.university_name = %s AND l.lesson_name = %s
        """
        cursor.execute(query, (msc_bsc, degree_title, university_name, lesson_name))
        conn.commit()
        cursor.close()
    except Exception as e:
        print(f"[ERROR] DB update failed for {lesson_name}: {e}")

@app.get("/course_skills_matrix", tags=["Bilateral"], summary="List of all courses with their Level 4 skills")
def course_skills_matrix(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)")
):
    cached_data = load_from_cache("biodiversity/course")
    if cached_data:
        total = len(cached_data)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page),
            "results": cached_data[start:end]
        }

    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    query = """
    SELECT 
        u.university_name, u.country,
        l.lesson_id, l.lesson_name, l.semester, l.description,
        l.msc_bsc, l.degree_title,
        s.skill_url, s.skill_name
    FROM University u
    JOIN Lessons l ON u.university_id = l.university_id
    LEFT JOIN Skills s ON l.lesson_id = s.lesson_id
    WHERE s.skill_url IS NOT NULL AND s.skill_url != ''
    """

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        print(f"[INFO] Retrieved {len(rows)} rows from database")

        course_map = defaultdict(lambda: {
            "lesson_id": None,
            "lesson_name": "",
            "university": "",
            "country": "",
            "msc_bsc": "",
            "degree_title": "",
            "skill_pairs": set()
        })
        skill_url_map = {}

        for row in rows:
            lesson_id = row["lesson_id"]
            course_map[lesson_id]["lesson_id"] = lesson_id
            course_map[lesson_id]["lesson_name"] = row["lesson_name"] or ""
            course_map[lesson_id]["university"] = row["university_name"]
            course_map[lesson_id]["country"] = row["country"]
            course_map[lesson_id]["msc_bsc"] = row.get("msc_bsc", "Unknown")
            course_map[lesson_id]["degree_title"] = row.get("degree_title", "Unknown")

            skill_url = row["skill_url"]
            skill_name = row["skill_name"]
            if skill_url:
                course_map[lesson_id]["skill_pairs"].add((skill_url, skill_name))
                skill_url_map[skill_url] = skill_name

        print(f"[INFO] Collected {len(skill_url_map)} unique skills across courses")

        def get_level4_skill_ids(skill_ids: List[str]) -> Set[str]:
            token = get_tracker_token()
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {token}"
            }
            data = [("ids", sid) for sid in skill_ids]
            data.append(("min_skill_level", "4"))
            data.append(("max_skill_level", "4"))

            try:
                response = requests.post(
                    "https://skillab-tracker.csd.auth.gr/api/skills",
                    headers=headers,
                    data=data,
                    verify=False
                )
                response.raise_for_status()
                items = response.json().get("items", [])
                print(f"[DEBUG] API returned {len(items)} Level 4 skills")
                return {item["id"] for item in items}
            except Exception as e:
                print(f"[ERROR] Failed to fetch Level 4 skills: {e}")
                return set()

        level4_skill_ids = set()
        all_skill_urls = list(skill_url_map.keys())

        for i in range(0, len(all_skill_urls), 50):
            batch = all_skill_urls[i:i + 50]
            level4_skill_ids.update(get_level4_skill_ids(batch))

        print(f"[INFO] Final count of Level 4 skill URLs: {len(level4_skill_ids)}")

        results = []

        for course in course_map.values():
            level4_skills = [
                skill_url_map[url]
                for (url, _) in course["skill_pairs"]
                if url in level4_skill_ids
            ]
            if level4_skills:
                results.append({
                    "lesson_id": course["lesson_id"],
                    "lesson_name": course["lesson_name"],
                    "university": course["university"],
                    "country": course["country"],
                    "degree": {
                        "type": course["msc_bsc"],
                        "title": course["degree_title"]
                    },
                    "skills": sorted(set(level4_skills))
                })

        print(f"[INFO] Constructed result for {len(results)} courses with Level 4 skills")

        total = len(results)
        start = (page - 1) * per_page
        end = start + per_page
        paginated = results[start:end]

        save_to_cache("biodiversity/course", results)

        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page),
            "results": paginated
        }

    finally:
        cursor.close()
        conn.close()

@app.get("/course_skill_urls_matrix", tags=["Bilateral"], summary="List of all courses with their Level 4 skill URLs")
def course_skill_urls_matrix(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)")
):
    cached_data = load_from_cache("biodiversity/course")

    if cached_data is not None and all("skill_ids" in c for c in cached_data):
        total = len(cached_data)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page),
            "results": [
                {
                    "lesson_id": course["lesson_id"],
                    "lesson_name": course["lesson_name"],
                    "university": course["university"],
                    "country": course["country"],
                    "degree": course["degree"],
                    "skill_ids": course.get("skill_ids", [])
                }
                for course in cached_data[start:end]
            ]
        }

    print("[INFO] Skill IDs missing from cache — recomputing...")

    if cached_data is None:
        raise HTTPException(status_code=500, detail="Cache not found. Run /course_skills_matrix first.")

    query = """
    SELECT 
        l.lesson_id, s.skill_url, s.skill_name
    FROM Lessons l
    LEFT JOIN Skills s ON l.lesson_id = s.lesson_id
    WHERE s.skill_url IS NOT NULL AND s.skill_url != ''
    """

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(query)
        rows = cursor.fetchall()

        lesson_to_urls = defaultdict(set)
        url_to_name = {}

        for row in rows:
            lesson_id = row["lesson_id"]
            url = row["skill_url"]
            name = row["skill_name"]
            if url:
                lesson_to_urls[lesson_id].add((url, name))
                url_to_name[url] = name

        def get_level4_skill_urls(skill_urls: List[str]) -> Set[str]:
            token = get_tracker_token()
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {token}"
            }
            data = [("ids", url) for url in skill_urls]
            data.append(("min_skill_level", "4"))
            data.append(("max_skill_level", "4"))

            try:
                response = requests.post(
                    "https://skillab-tracker.csd.auth.gr/api/skills",
                    headers=headers,
                    data=data,
                    verify=False
                )
                response.raise_for_status()
                return {item["id"] for item in response.json().get("items", [])}
            except Exception as e:
                print(f"[ERROR] Tracker API call failed: {e}")
                return set()

        all_urls = list(url_to_name.keys())
        level4_urls = set()
        for i in range(0, len(all_urls), 50):
            batch = all_urls[i:i + 50]
            level4_urls.update(get_level4_skill_urls(batch))

        print(f"[INFO] Found {len(level4_urls)} Level 4 skill URLs")

        for course in cached_data:
            lesson_id = course["lesson_id"]
            skill_ids = [
                url for (url, _) in lesson_to_urls.get(lesson_id, []) if url in level4_urls
            ]
            course["skill_ids"] = sorted(skill_ids)

        save_to_cache("biodiversity/course", cached_data)

        total = len(cached_data)
        start = (page - 1) * per_page
        end = start + per_page

        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page),
            "results": [
                {
                    "lesson_id": course["lesson_id"],
                    "lesson_name": course["lesson_name"],
                    "university": course["university"],
                    "country": course["country"],
                    "degree": course["degree"],
                    "skill_ids": course["skill_ids"]
                }
                for course in cached_data[start:end]
            ]
        }

    finally:
        cursor.close()
        conn.close()

# Conda PYROOT Implementation - RUN LOCALLY WITH CONDA if want to utilize. Unmark comments.

# @app.get("/PyROOT_skill_frequency_histogram", tags=["PyROOT"], response_class=HTMLResponse)
# def generate_histogram():
#     resp = requests.get("https://portal.skillab-project.eu/curriculum-skills/descriptive/skills_frequency")
#     df = pd.DataFrame(resp.json())
#     conn = mysql.connector.connect(**DB_CONFIG)
#     cursor = conn.cursor(dictionary=True)
#     cursor.execute("SELECT DISTINCT skill_name, skill_url FROM Skills")
#     skill_url_map = {row["skill_name"]: row["skill_url"] for row in cursor.fetchall()}
#     cursor.close()
#     conn.close()

#     def fetch_level(skill_url):
#         try:
#             res = requests.get(skill_url)
#             if res.status_code == 200:
#                 return res.json().get("level")
#         except:
#             return None
#         return None

#     def assign_category(skill_name):
#         url = skill_url_map.get(skill_name)
#         level = fetch_level(url) if url else None
#         if level in [4, 5]:
#             return "hard"
#         elif level in [1, 2, 3]:
#             return "soft"
#         return "unknown"

#     df["category"] = df["skill"].apply(assign_category)
#     f = TFile("skills.root", "RECREATE")
#     tree = TTree("skills", "Skill Frequency Tree")
#     skill = np.array(b"", dtype='S60')
#     freq = np.zeros(1, dtype='i')
#     category = np.array(b"", dtype='S10')
#     tree.Branch("skill", skill, "skill/C")
#     tree.Branch("frequency", freq, "frequency/I")
#     tree.Branch("category", category, "category/C")

#     for _, row in df.iterrows():
#         skill[0] = row["skill"].encode()
#         freq[0] = int(row["frequency"])
#         category[0] = row["category"].encode()
#         tree.Fill()

#     tree.Write()
#     f.Close()
#     f = TFile("skills.root")
#     tree = f.Get("skills")
#     max_freq = max(df["frequency"].max(), 100)

#     h_hard = TH1F("h_hard", "Hard Skills", 100, 0, max_freq)
#     h_soft = TH1F("h_soft", "Soft Skills", 100, 0, max_freq)

#     for entry in tree:
#         if entry.category.decode() == "hard":
#             h_hard.Fill(entry.frequency)
#         elif entry.category.decode() == "soft":
#             h_soft.Fill(entry.frequency)

#     c = TCanvas()
#     h_hard.SetLineColor(4)
#     h_soft.SetLineColor(2)
#     h_hard.SetTitle("Skill Frequency Histogram;Frequency;Count")
#     h_hard.Draw()
#     h_soft.Draw("SAME")

#     legend = TLegend(0.7, 0.7, 0.9, 0.85)
#     legend.AddEntry(h_hard, "Hard Skills", "l")
#     legend.AddEntry(h_soft, "Soft Skills", "l")
#     legend.Draw()

#     c.SaveAs("skills_histogram.png")
#     html = """
#     <html>
#         <head><title>Skill Histogram</title></head>
#         <body>
#             <h2>Hard vs Soft Skills Histogram (ROOT)</h2>
#             <img src="/PyROOT_skill_histogram_image" alt="Skill Histogram" style="width:800px;">
#         </body>
#     </html>
#     """
#     return HTMLResponse(content=html)

# @app.get("/PyROOT_skill_histogram_image", tags=["PyROOT"])
# def get_histogram_image():
#     return FileResponse("skills_histogram.png", media_type="image/png")



@app.post("/crawl", tags=["Crawler"], summary="Start a web crawl")
def crawl_university(request: CrawlRequest):
    """
    [Warning] A very primitive version of the crawler, for accessing university sites and extracting lesson data automatically.
    - Requires a URL, preferrably on the curriculum page 
    """
    url = request.url
    
    crawler = UniversityCrawler(url)
    course_info = crawler.crawl()
    
    return {"university": crawler.university_name, "courses": course_info.semesters}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
