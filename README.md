
# 🧠 SKILLCRAWL

<img align="right" src="https://easychair.org/images/cfp-logo/ucaat2025.jpg?id=17241655" alt="UCAAT" width="80"/>

### ✅ **New**: Automatic AI APIFY crawler + CurricuNLP university recognizer on free-form text, PDFs and websites

**SkillCrawl** is a tool that automatically scans university curricula and extracts the **skills** that students actually learn, starting from free-form, structured text, **PDF files** *or* even directly from **university websites**!

It uses the official **ESCO** skill ontology (European Commission) to link course content to real-world competencies, and integrates the **CurricuNLP** model, a powerful NLP engine trained on **40,000 curriculum inputs**, which produced **150,000 JSON files** corresponding to **50,000 courses in over 100 *European* universities**.

You can run it via the **FastAPI / Swagger UI**, the **terminal interface**, or the included **Docker Compose** setup.  
Everything from skill extraction to **automatic university detection** and **website crawling** is handled for you.

> ✅ **New**: Uses **CurricuNLP** on HuggingFace for full NLP processing

> ✅ **Note**: Large `.sql` database dumps are tracked via **Git LFS**

---

<p align="center">
  <a href="https://huggingface.co/spaces/marfoli/CurricuNLP" target="_blank">
    <img src="https://img.shields.io/badge/View%20CurricuNLP%20on-HuggingFace-purple?logo=huggingface" alt="View CurricuNLP on HuggingFace" />
  </a>
</p>



---

## 🚀  What Does SkillCrawl Do?

| Feature | Description |
|--------|-------------|
| 📥 **PDF Processing** | Parses university curriculum PDFs and splits by semesters + lessons |
| 🌐 **Website Crawling** | Visits curriculum pages on university sites and extracts lesson/module content |
| 🧠 **CurricuNLP** | NLP model that recognizes course structure, descriptions & fields |
| 🎓 **Automatic University Detection** | Detects the university name from the content |
| 📡 **ESCO Skill Extraction** | Maps course content to ESCO skill definitions |
| 🔁 **Skill ↔ Course Search** | Find courses by skill, or inspect skills per course |
| 💾 **MySQL Integration** | Save results for dashboards / analysis |
| ⚡ **Caching** | Skipped (now removed) — all data is re-generated on demand |
| 🐳 **Docker Compose Support** | 1-step startup for backend + dependencies |

---

## ▶️ Quick Start (Docker)

```bash
docker compose up --build
````

> This builds the API, installs dependencies, and starts the backend with CurricuNLP support enabled.

Then open the documentation at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🛠️  Manual Usage

### ➤ FastAPI mode

```bash
uvicorn main:app --reload
```

### ➤ Terminal / CLI mode

```bash
python skillcrawl.py
```

---

## 🧪 Example Output

```
University detected: University of Edinburgh

Semester 1:
    Data Structures and Algorithms
      → Skill: algorithm design
      → Skill: data modelling

Skill: machine learning
  → Matched Course: Foundations of Artificial Intelligence (Score: 89)
```

---

## 📂 Project Structure

| File/Folder          | Purpose                                   |
| -------------------- | ----------------------------------------- |
| `main.py`            | FastAPI application                       |
| `skillcrawl.py`      | CLI interface                             |
| `crawler/`           | Playwright-based crawler (URL processing) |
| `pdf_utils.py`       | PDF parsing + text extraction             |
| `skills.py`          | Skill extraction & ESCO lookup            |
| `database.py`        | Writes data to MySQL                      |
| `helpers.py`         | University detection / general utilities  |
| `.gitattributes`     | Git LFS rules (`*.sql` files)             |
| `skillcrawl.sql`     | Database schema                           |
| `docker-compose.yml` | Docker + CurricuNLP setup                 |
| `requirements.txt`   | Python dependencies                       |

---

## ✅ Requirements (Manual Run)

```bash
pip install -r requirements.txt
playwright install
git lfs install   # to download the SQL files tracked via LFS
```

---

## ℹ️ Notes

* The **CurricuNLP** model is pulled automatically via HuggingFace in both CLI and API mode.
* Large `.sql` files (database dumps) are tracked via **Git LFS**, so you must have Git LFS installed **before** cloning/pushing.
* The crawler respects depth limits and only follows relevant curriculum links.

---


