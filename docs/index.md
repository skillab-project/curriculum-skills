# 🧠 SKILLCRAWL

**SkillCrawl** is a tool that helps you search into university curricula and extract important skills that students are expected to learn — all from PDF files. It uses the official **ESCO** skill framework and leverages the `esco-skill-extractor` to identify real-world competencies and link them back to course content.

You can run it interactively in your terminal, use its web API with Swagger UI, or even connect it to a MySQL database. Whether you're doing academic research, analyzing skill trends, or building a course recommender, SkillCrawl saves you time by automating the skill detection.

> ⚙️ **Coming soon**: An **automated web crawler** is currently under development to let you scan entire university websites for curriculum pages — no PDFs required! It’s still experimental but already crawling basic lesson data from real websites.

---

## 🚀 What Does SkillCrawl Actually Do?

SkillCrawl is like a skill-focused academic detective. Here's how it works:

<img align="right" src="https://easychair.org/images/cfp-logo/ucaat2025.jpg?id=17241655" alt="UCAAT" width="69"/>
<img align="right" src="https://th.bing.com/th/id/OIP.XWHdKPsdkbsZ4HvK6B5jWwHaCH?rs=1&pid=ImgDetMain" alt="Skillab" width="200"/>

- 📥 Takes in PDF files of university curricula.
- 🔍 Reads and processes the text using PDF-aware techniques.
- 🗂️ Splits course content into **semesters** and **individual lessons**, even if the formatting is inconsistent.
- 🧩 For each lesson, it extracts related **ESCO skills** (the EU’s official database of competencies and qualifications).
- 🧠 Lets you **search by skill** (e.g., "data analysis") and find which courses teach it — or browse what skills each course covers.
- 🛢️ Supports **saving everything to MySQL**, so you can build dashboards or analyze data later.
- 🔁 Keeps a local **cache** of all processed files for instant reuse.
- 💬 Comes with both a full **FastAPI/Swagger web interface** and a **terminal menu**.
- 🌐 (Experimental) A **crawler module** is being built to auto-visit university websites and scrape curriculum content without any manual download.

---
## 🛠️ Usage

### ➤ FastAPI Mode (Swagger UI)
Run the API with:

```bash
uvicorn main:app --reload
```

Then open: [http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

Here you can:
- Process PDFs
- Search for skills (by name or URL) or courses -in a specific university or globally-
- Get top N skills per university or globally
- Fetch full lesson/skill breakdowns
- Save data to DB with a click

Ensure:
- Your database (`SkillCrawl`) is set up using `skillcrawl.sql`.
- XAMPP/MySQL is running.

---

## 🧪 Example Output

SkillCrawl prints matching lessons and their associated ESCO skills:

```
Semester 1:
  Introduction to Programming
    Skill: problem solving
    Skill: code debugging
```

Or search in reverse:

```
Skill: machine learning

Matched Course: AI and Ethics (Score: 84)
```

---

## 🏗️ Project Structure

| File/Folder         | Purpose |
|---------------------|---------|
| `main.py`           | FastAPI app |
| `skillcrawl.py`     | Entry point with menu & CLI control |
| `pdf_utils.py`      | PDF parsing, semester/lesson splitting |
| `database.py`       | Handles writing to MySQL |
| `skills.py`         | Skill extraction, DB lookup, and fuzzy matching |
| `output.py`         | Visual output helpers (ASCII, color, lines) |
| `config.py`         | DB and cache config |
| `helpers.py`        | Utility functions (caching, validation, detection) |
| `menu.py`           | Terminal ASCII Interface |
| `skillcrawl.sql`    | SQL schema to set up the database |
| `requirements.txt`  | List of required Python libraries |
| `cache/`            | Stores processed data by university |
| `tests/`            | Stores the available test cases you can run |
| `README.md`         | You're reading it! |

---

## 📚 Use Cases

- Curriculum Analysis & Benchmarking
- Educational Skill Mapping
- Automated Course Tagging
- Career Pathway Recommendations
- Academic Skill Graph Construction

---

## 🧩 Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 💾 Database Setup

Initialize the database using:

```bash
mysql -u root -p < skillcrawl.sql
```

Make sure to configure the connection details in `skillcrawl.py` and `skills.py` under `db_config`.

---

## 🔄 Caching

Processed results are cached per university PDF in:
- `cache/pdf_cache.json`
- `university_cache.json`

*Where university is replaced by a respective university name that the cache represents.*

You can delete these files to force re-processing.

-----

## 🔍 Key Method Explanations

Here are the most important methods across the codebase explained for better understanding and easier contribution:


### `main.py` (FastAPI)

- **`/process_pdf`**: Endpoint to process a PDF, extract text, split lessons, run skill extraction, and cache results.
- **`/search_skill`**: Search database for lessons teaching a given skill.
- **`/calculate_skillnames`**: Enriches lessons with missing skill names via Skillab Tracker API.
- **`/get_top_skills` & `/get_top_skills_all`**: Return most frequently taught skills globally or per university.
- **`/filter_skillnames`**: Lookup skill names for a specific university and lesson using either DB or cache.


### `skillcrawl.py`

- **`main(...)`**: The main method that runs the terminal interface. Handles PDF processing, caching, and triggering the desired output based on command-line args.
- **`get_university_country(university_name)`**: Uses an external API to detect a university's country from its name. Updates local cache.


### `pdf_utils.py`

- **`extract_text_from_pdf(pdf_file_path)`**: Uses `PyMuPDF` to extract text from each page of the PDF. Also caches results.
- **`extract_text_after_marker(text, markers)`**: Takes all text and returns everything after specific marker words like "Course Content".
- **`split_by_semester(text)`**: Breaks the PDF text into sections by semester or year using regex.
- **`process_pages_by_lesson(pages)`**: Processes PDF page-by-page to detect lessons and their descriptions using uppercase pattern recognition.



### `skills.py`

- **`get_skills_for_lesson(...)`**: Looks up all skills associated with a given lesson or university. Optionally searches by lesson name.
- **`extract_and_get_title(skill_url)`**: Fetches the readable name of a skill from its ESCO URL using their API.
- **`search_courses_by_skill_database(...)`**: Searches all courses for a fuzzy match of the given skill name in the database.



### `database.py`

- **`write_to_database(...)`**: Saves extracted data to MySQL: university, semesters, lessons, and skills. Also merges new skills with what's already in the cache.



### `helpers.py`

- **`find_possible_university(pdf_file_path)`**: Scans PDF content using regex to guess the university name.
- **`load_from_cache(university_name)` / `save_to_cache(...)`**: Manages university-specific cache JSONs.
- **`contains_greek_characters(...)` / `contains_no_lowercase_letters(...)`**: Utilities used in filtering invalid lesson names.
- **`extract_description(text)`**: Strips out and formats a clean description from lesson blocks.


