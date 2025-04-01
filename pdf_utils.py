import os
import pdfplumber
import requests
import re
import glob
import json
from thefuzz import fuzz, process
from output import print_loading_line
from helpers import is_cached, load_cache, save_cache, contains_no_lowercase_letters, clean_lesson_name, contains_greek_characters
from output import print_colored_text, print_green_line

from concurrent.futures import ThreadPoolExecutor


def download_pdf(url: str, save_path: str) -> str:
    curriculum_folder = "curriculum"
    os.makedirs(curriculum_folder, exist_ok=True)
    save_path = os.path.join(curriculum_folder, save_path)

    if os.path.exists(save_path):
        print("Curriculum already retrieved!")
        return save_path

    print(f"Downloading PDF from {url}")
    response = requests.get(url)
    with open(save_path, 'wb') as file:
        file.write(response.content)
    return save_path

import fitz 
from concurrent.futures import ThreadPoolExecutor

def extract_text_from_pdf(pdf_file_path: str) -> list:
    """Extracts text from a PDF file using PyMuPDF, with caching."""
    cache_key = f"text_{pdf_file_path}"
    if is_cached(cache_key):
        print(f"[CACHE] Loading text from cache for {pdf_file_path}")
        print_loading_line(25)
        return load_cache()[cache_key]

    print(f"[INFO] Extracting text from PDF: {pdf_file_path}")
    print_loading_line(25)

    page_texts = []
    
    try:
        doc = fitz.open(pdf_file_path) 

        def extract_page_text(page_num):
            """Extract text from a single page."""
            text = doc[page_num].get_text("text") or ""
            print(f"[DEBUG] Extracted {len(text)} chars from page {page_num + 1}")
            if len(text.strip()) < 50:
                print(f"[WARNING] Page {page_num + 1} might be empty or improperly read!")
            return text

        with ThreadPoolExecutor() as executor:
            page_texts = list(executor.map(extract_page_text, range(len(doc))))


        cache = load_cache()
        cache[cache_key] = page_texts
        from skills import save_cache
        save_cache(cache)

        print(f"[INFO] Successfully extracted text from {len(page_texts)} pages.")
        return page_texts

    except Exception as e:
        print(f"[ERROR] ❌ Failed to extract text from PDF: {e}")
        return []



def extract_text_after_marker(text: list, markers: list) -> str:
    cache_key = f"text_after_marker_{hash(''.join(markers))}"

    if is_cached(cache_key):
        print(f"Loading text after markers from cache.")
        print_loading_line(25)
        return load_cache()[cache_key]

    print_colored_text(f" >>> Extracting text after markers: {', '.join(markers)}", 34)
    full_text = '\n'.join(text)
    for marker in markers:
        marker_index = full_text.lower().find(marker.lower())
        if marker_index != -1:
            print_colored_text(f"Found marker: {marker}", 32)
            extracted_text = full_text[marker_index + len(marker):]
            return extracted_text.lstrip()
    
    print("No marker found in the text.")
    return full_text 

def split_by_semester(text: str) -> list:
    from skills import save_cache
    cache_key = f"semesters_{hash(text)}"
    if is_cached(cache_key):
        print(f"[CACHE] Loading semesters from cache")
        return load_cache()[cache_key]
    
    print("[INFO] Splitting text by semester or year...")
    
    print(f"[DEBUG] First 500 chars of text:\n{text[:500]}\n---")

    semesters = re.split(r'(?i)(\b(?:Year\s+(?:One|Two|Three|Four|1|2|3|4)|\d+\s*(?:st|nd|rd|th)?\s*Semester)\b)', text)
    
    combined_semesters = []
    seen_semesters = set()

    for i in range(1, len(semesters), 2):
        semester = ''.join(semesters[i:i+2])
        if semester.lower() not in seen_semesters:
            combined_semesters.append(semester)
            seen_semesters.add(semester.lower())

    print(f"[INFO] Found {len(combined_semesters)} semesters")
    
    for i, sem in enumerate(combined_semesters, 1):
        print(f"[DEBUG] Semester/Year {i}: {sem[:100]}...")

    cache = load_cache()
    cache[cache_key] = combined_semesters
    from skills import save_cache
    save_cache(cache)
    
    return combined_semesters


def extract_description(text: str) -> str:
    lines = text.split('\n')
    lesson_description = []
    capture = False
    
    for line in lines:
        line = line.strip()
        
        if line.isupper() and len(line.split()) > 0:
            if capture:
                break
            capture = True
            continue
        
        if capture:
            lesson_description.append(line)
    
    description = '\n'.join(lesson_description).strip()

    description = re.sub(r'(?i)course content', '', description).strip()
    return description if description else "This lesson has no data!"

from concurrent.futures import ThreadPoolExecutor
def process_pages_by_lesson(pages: list) -> dict:
    print("[INFO] Processing pages to extract lessons...")
    lesson_dict = {}

    def process_page(page_data):
        page_num, page = page_data
        print(f"[DEBUG] Processing page {page_num + 1}...")

        lines = page.split('\n')
        if not lines:
            print(f"[WARNING] Page {page_num + 1} is empty, skipping...")
            return None

        potential_lesson_name = None
        lesson_text = []
        capture_text = False

        for line in lines:
            line = line.strip()

            if line.isupper() and len(line.split()) > 0:
                if potential_lesson_name:
                    lesson_dict[potential_lesson_name] = '\n'.join(lesson_text).strip()
                    print(f"[INFO] ✅ Stored lesson: {potential_lesson_name}")

                potential_lesson_name = clean_lesson_name(line).strip()
                print(f"[DEBUG] Detected lesson title: {potential_lesson_name}")

                if re.search(r'[*_=!?\.]', potential_lesson_name) or (',' in potential_lesson_name and 'AND' not in potential_lesson_name.upper()):
                    print(f"[DEBUG] Skipping '{potential_lesson_name}' (contains special characters or comma without AND)")
                    potential_lesson_name = None
                    continue
                if re.match(r'^[a-zA-Z]+\d+$', potential_lesson_name):
                    print(f"[DEBUG] Skipping '{potential_lesson_name}' (matches letter-number pattern)")
                    potential_lesson_name = None
                    continue
                if len(potential_lesson_name) <= 3:
                    print(f"[DEBUG] Skipping '{potential_lesson_name}' (too short)")
                    potential_lesson_name = None
                    continue
                if re.match(r'^[^A-Za-z]', potential_lesson_name):
                    print(f"[DEBUG] Skipping '{potential_lesson_name}' (starts with a symbol or number)")
                    potential_lesson_name = None
                    continue
                if re.search(r'\bISBN\b|\bPUBLISHER\b|\d{4}', potential_lesson_name):
                    print(f"[DEBUG] Skipping '{potential_lesson_name}' (likely a book or ISBN reference)")
                    potential_lesson_name = None
                    continue

                capture_text = True
                lesson_text = []

            elif capture_text and potential_lesson_name:
                lesson_text.append(line) 

        if potential_lesson_name:
            lesson_dict[potential_lesson_name] = '\n'.join(lesson_text).strip()
            print(f"[INFO] ✅ Stored last lesson: {potential_lesson_name}")

    with ThreadPoolExecutor() as executor:
        results = executor.map(process_page, enumerate(pages))

    print(f"[INFO] Extracted {len(lesson_dict)} lessons")
    return lesson_dict



def get_university_name_mapping():
    """Reads the 'university_cache.json' and returns a mapping of PDF paths to university names."""
    try:
        with open('university_cache.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("University cache file not found. Using default names.")
        return {}
    except json.JSONDecodeError:
        print("Error reading the university cache file. Using default names.")
        return {}


def get_pdf_path():
    """Searches for PDF files in the 'curriculum' folder and prompts the user to choose."""
    pdf_files = glob.glob("curriculum/*.pdf") 
    university_mapping = get_university_name_mapping()  

    if not pdf_files:
        print_colored_text("No PDF files found in the 'curriculum' folder.", 31)
        return None

    if len(pdf_files) == 1:
        university_name = university_mapping.get(pdf_files[0], 'Unknown University')
        print_colored_text(f"University: {university_name} - Found PDF file: {pdf_files[0]}", 32)
        return pdf_files[0]

    print_green_line(30)
    print_colored_text("Multiple PDF files found. Please choose one:", 32)
    print_green_line(30)

    for i, file in enumerate(pdf_files):
        university_name = university_mapping.get(file, 'Unknown University')
        print(f"{i + 1}. [{university_name}] - {file}")

    while True:
        try:
            print_green_line(30)
            choice = int(input("Enter the number of your choice: "))
            if 1 <= choice <= len(pdf_files):
                return pdf_files[choice - 1]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")