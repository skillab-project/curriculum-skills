# 🧪 SkillCrawl Unit Test Suite

This test suite ensures the quality of SkillCrawl’s outputs across multiple dimensions such as correctness, integrity, and completeness.

Tests are categorized into the following quality dimensions:

| Category              | Goal                                                                 |
|-----------------------|----------------------------------------------------------------------|
| ✅ Data Integrity      | No duplicates or corrupted entries                                   |
| 📐 Consistency         | Stable, explainable patterns in results                             |
| 📊 Completeness        | No missing or under-analyzed lessons                                |
| 🎯 Accuracy            | Extracted skills match expected outcomes from trusted datasets       |

---

## 📁 Structure

```
tests/
├── json/
│   └── [Expected output JSON files]
├── sample_pdfs/
│   └── test_curriculum.pdf
├── test_compare_extracted_titles.py
├── test_compare_extracted_skills.py
├── test_compare_top_skills.py
├── test_compare_skill_search.py
└── ...
```

---

## ✅ Existing Test Cases

### 🎯 Accuracy Testing

#### 1. Compare Extracted Titles from PDF  
**File:** `test_compare_extracted_titles.py`   
**Compares:** Method output → `extracted_titles_expected.json`  
**Difficulty:** 🟩 Easy  

**Purpose:** Extracts titles from a sample PDF using `process_pdf()` and checks them against the confirmed set.

- **Method Called:** `process_pdf(PDFProcessingRequest)`
- **Expected File:** `json/extracted_titles_expected.json`

🔍 Compares returned lesson titles (per semester) directly to the expected JSON.
✔️ Validates that all course titles are correctly extracted per semester.


---

#### 2. Skill Extraction from Course Descriptions  
**File:** `test_compare_extracted_skills.py`  
**Compares:** Method output → `extracted_skills_expected.json`  
**Difficulty:** 🟨 Medium  
**Purpose:** Uses `process_pdf()` and `calculate_skillnames()` to extract skills from course descriptions and compare to expected.

- **Methods Called:**  
  - `process_pdf(PDFProcessingRequest)`  
  - `calculate_skillnames(university_name)`

- **Expected File:** `json/extracted_skills_expected.json`

🧠 Confirms that all expected skills per lesson are detected, including names.
✔️ Ensures that skills extracted from lesson descriptions match confirmed ones.


---

#### 3. Top N Skills for a University  
**File:** `test_compare_top_skills.py`  
**Compares:** Method output → `top_skills_cambridge_expected.json`  
**Difficulty:** 🟩 Easy  
**Purpose:** Verifies that `get_top_skills()` returns the correct top N skills for a university with the expected frequency.

- **Method Called:** `get_top_skills(TopSkillsRequest)`
- **Expected File:** `json/top_skills_cambridge_expected.json`

📊 Checks that skill names and counts match expected results.
✔️ Checks top ESCO skills returned by the system and their frequency.

---

#### 4. Skill-based Search Results  
**File:** `test_compare_skill_search.py`  
**Method:** `search_skill()`  
**Difficulty:** 🟨 Medium  
**Purpose:** Runs a live skill-based course search and compares the results to verified course/skill/university matches.

- **Method Called:** `search_skill(SkillSearchRequest)`
- **Expected File:** `json/C_skill_search_expected.json`

🔎 Ensures that returned results match the expected structure and content, including frequency and fuzzy score.
✔️ Verifies courses matched to a given skill (across all universities).

---

## 🧪 Proposed New Tests (for expansion)

These are **not yet implemented** purposefully — they are ideal for a challenge at the testathon!

---

### ✅ Data Integrity Testing

#### 5. Duplicate Skills Per Lesson  
**Goal:** Ensure no skill is assigned more than once to the same lesson.  
**Method(s):** Check `calculate_skillnames()` output  
**Difficulty:** 🟩 Easy  
✔️ Loop through all lessons and ensure all skill names are unique.

---

#### 6. Duplicate Skills Across University  
**Goal:** Detect redundant skill entries across multiple lessons in the same university.  
**Method(s):** Use `get_skills_for_lesson()`  
**Difficulty:** 🟨 Medium  
✔️ Count how often each skill appears and detect unnecessary repetition.

---

### 📐 Consistency Testing

#### 7. Cross-Course Skill Similarity  
**Goal:** Ensure that related lessons (e.g., under the same department or semester) share logically consistent skill sets.  
**Method(s):** Compare lesson skill sets using similarity scoring.  
**Difficulty:** 🟥 Hard  
✔️ Flag anomalies where unrelated skill sets appear within similar contexts.

---

#### 8. Inconsistent Skill Distribution  
**Goal:** Detect universities where similar lesson names have completely disjoint skills.  
**Method(s):** Cluster lessons and analyze skill overlap.  
**Difficulty:** 🟥 Hard  
✔️ Suggest possible errors or inconsistencies in extraction.

---

### 📊 Completeness Testing

#### 9. Missing Skills in Lessons  
**Goal:** Flag any lessons with zero skills after processing.  
**Method(s):** `calculate_skillnames()`  
**Difficulty:** 🟩 Easy  
✔️ Useful to catch bad input data or failed OCR.

---

#### 10. Minimum Skill Threshold  
**Goal:** Validate that each lesson has at least N skills (e.g., 3).  
**Method(s):** Compare count from skill output.  
**Difficulty:** 🟨 Medium  
✔️ Helps surface under-analyzed lessons.

---

### 🎯 Accuracy Testing (Advanced)

#### 11. Skill Mapping Validation with ESCO Labels  
**Goal:** Compare actual ESCO preferred labels with what was extracted.  
**Method(s):** Use `extract_and_get_title()`  
**Difficulty:** 🟨 Medium  
✔️ Detect mismatches between ESCO URLs and their resolved names.

---

#### 12. API JSON Schema Validation  
**Goal:** Ensure the API responses match expected structure and required fields.  
**Method(s):** FastAPI schema + `pydantic`  
**Difficulty:** 🟨 Medium  
✔️ Useful for future-proofing API integrations.

---

## ▶️ How to Run Tests

Install requirements:

```bash
pip install -r requirements.txt
```

Run all tests:

```bash
python -m unittest discover -s tests

```

Run a specific test:

```bash
python -m unittest tests/test_compare_skill_search.py
```
> Tip: Make sure you are in the curriculum-skills folder!
> Hint: Running this will return you both statistics and if it passed!

---

## 🧠 Summary Table

| Test Case                            | Type               | Difficulty |
|-------------------------------------|--------------------|------------|
| Title extraction                    | 🎯 Accuracy         | 🟩 Easy     |
| Skill extraction from descriptions  | 🎯 Accuracy         | 🟨 Medium   |
| Top skills per university           | 🎯 Accuracy         | 🟩 Easy     |
| Skill-based search                  | 🎯 Accuracy         | 🟨 Medium   |
| Detect duplicate skills             | ✅ Data Integrity    | 🟩 Easy     |
| Duplicate across university         | ✅ Data Integrity    | 🟨 Medium   |
| Cross-course similarity             | 📐 Consistency       | 🟥 Hard     |
| Inconsistent skill distribution     | 📐 Consistency       | 🟥 Hard     |
| Missing skills                      | 📊 Completeness      | 🟩 Easy     |
| Minimum skills per lesson           | 📊 Completeness      | 🟨 Medium   |
| ESCO label mismatch                 | 🎯 Accuracy         | 🟨 Medium   |
| API schema check                    | 🎯 Accuracy         | 🟨 Medium   |
