from collections import Counter
import os
import re
import json
import requests
import pdfplumber
from fuzzywuzzy import fuzz

CACHE_DIR = 'cache'
CACHE_FILE = 'pdf_cache.json'

os.makedirs(CACHE_DIR, exist_ok=True)

unknown_university_counter = 0

UNI_FILE = "university_cache.json"

if os.path.exists(UNI_FILE):
    with open(UNI_FILE, "r") as f:
        university_cache = json.load(f)
else:
    university_cache = {}

def save_cache():
    """Saves the university cache to a JSON file."""
    with open(UNI_FILE, "w") as f:
        json.dump(university_cache, f, indent=4)

def load_university_cache():
    """Load the university cache from JSON file."""
    if not os.path.exists(UNI_FILE):
        return {}
    try:
        with open(UNI_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("❌ Error: Corrupted university_cache.json file. Resetting cache.")
        return {}


def load_from_cache(university_name):
    cache_file = os.path.join(CACHE_DIR, f"{university_name}_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_to_cache(university_name, data):
    cache_file = os.path.join(CACHE_DIR, f"{university_name}_cache.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)  


def clean_lesson_name(name: str) -> str:
    return re.sub(r'\s*\(.*?\)\s*', '', name).strip()

def load_cache():
    cache_file_path = os.path.join(CACHE_DIR, CACHE_FILE)
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r') as cache_file:
            return json.load(cache_file)
    return {}


def is_cached(key):
    cache = load_cache()
    return key in cache

def contains_greek_characters(text: str) -> bool:
    return bool(re.search(r'[Α-ω]', text))

def contains_no_lowercase_letters(text: str) -> bool:
    return not any(char.islower() for char in text)

def extract_description(text: str) -> str:
    start_marker = 'General competences'
    end_marker = 'Assessment'
    start_index = text.find(start_marker)
    if start_index == -1:
        return ""
    end_index = text.find(end_marker, start_index)
    if end_index == -1:
        end_index = len(text)
    description = text[start_index + len(start_marker):end_index].strip()
    description = description.replace('Course content', '').strip()
    return description if description else "This lesson has no data!"



def find_possible_university(pdf_file_path):
    from skills import save_cache
    """Extracts university name from PDF or assigns a unique 'Unknown University' label."""
    with pdfplumber.open(pdf_file_path) as pdf:
        all_text = " ".join(page.extract_text() or '' for page in pdf.pages)

    pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)* (?:University|College|Institute|School|Academy|College))\b|\b(?:University|College|Institute|School|Academy|College) of [A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
    match = re.search(pattern, all_text)

    if match:
        return match.group(0).strip()

    if pdf_file_path in university_cache:
        return university_cache[pdf_file_path]

    unknown_count = sum(1 for entry in university_cache.values() if "Unknown University" in entry) + 1
    unknown_name = f"Unknown University {unknown_count}"
    
    university_cache[pdf_file_path] = {"name": unknown_name}
    save_cache(university_cache)

    return unknown_name