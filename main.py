from fastapi import FastAPI, APIRouter, HTTPException, Query, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, AnyUrl
from typing import List, Dict, Optional, Literal, Set, Any
from collections import Counter, defaultdict
from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz
from math import ceil
from fastapi import BackgroundTasks, Body
from gradio_client import Client
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from threading import Lock, Semaphore
from cleaner import clean_file, iter_json_files
import csv

_ESCO_LABEL_CACHE: Optional[Dict[str, Dict[str, Optional[str]]]] = None




def _load_esco_skill_labels() -> Dict[str, Dict[str, Optional[str]]]:
    global _ESCO_LABEL_CACHE

    if _ESCO_LABEL_CACHE is not None:
        return _ESCO_LABEL_CACHE

    csv_path = os.getenv("ESCO_SKILLS_CSV", "/app/data/skills_en.csv")

    if not os.path.exists(csv_path):
        raise RuntimeError(f"ESCO skills CSV not found: {csv_path}")

    mapping: Dict[str, Dict[str, Optional[str]]] = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            uri = (
                row.get("conceptUri")
                or row.get("concept_uri")
                or row.get("uri")
                or row.get("skillUri")
            )

            label = (
                row.get("preferredLabel")
                or row.get("preferred_label")
                or row.get("label")
            )

            skill_type = (
                row.get("skillType")
                or row.get("skill_type")
                or row.get("reuseLevel")
            )

            if uri and label:
                mapping[uri.strip()] = {
                    "label": label.strip(),
                    "esco_id": uri.strip().rsplit("/", 1)[-1],
                    "level": skill_type.strip() if skill_type else None,
                }

    _ESCO_LABEL_CACHE = mapping
    print(f"[ESCO] Loaded {len(mapping)} ESCO skill labels from {csv_path}", flush=True)
    return mapping


def _resolve_urls_to_names_local_esco(all_urls: Set[str]) -> Dict[str, dict]:
    labels = _load_esco_skill_labels()

    out: Dict[str, dict] = {}

    for url in all_urls:
        meta = labels.get(url)

        if meta:
            out[url] = meta
        else:
            out[url] = {
                "label": url,  # fallback only
                "esco_id": url.rsplit("/", 1)[-1],
                "level": None,
            }

    return out

try:
    import orjson as _jsonlib


    def _json_load(f):
        return _jsonlib.loads(f.read())
except Exception:
    import json as _jsonlib


    def _json_load(f):
        return _jsonlib.load(f)

import logging
import asyncio
import os
import json
import re
import requests
from requests.adapters import HTTPAdapter
import mysql.connector
import shutil
import time
import glob
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from crawler import run_apify_crawler

from database import (
    upsert_skill_and_link_with_categories,
    is_database_connected,
    write_json_to_database,
    _labels_to_course,
    _normalize_text_for_ner,
    _norm_text,
    _strip_uni_from_title,
    _canonical_key,
    _best_match,
    _merge_text,
    _merge_list,
    _deep_merge_dict,
    _merge_labels,
    _merge_course_records,
    _prepare_and_merge_courses,
    _infer_msc_bsc_from_text,
    _course_from_curriculnlp_labels_payload,
    _to_decimal,
    _sanitize_course_for_db
)
from pdf_utils import (
    extract_text_from_pdf,
    _chunk_text,
    _clean_pdf_text,
    _merge_ner,
    _detect_lang,
    _translate_hf,
    _file_pdf_path,
    _extract_text_pdfminer,
    _extract_text_ocr,
    _extract_text_fallback,
    extract_text_from_pdf_best
)
from output import (
    _norm,
    _score,
    _find_uni_by_name,
    _load_universities
)
from config import DB_CONFIG

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime



from policy import router as policy_router

from skill_gap.router import router as skill_gap_router
from skill_gap.longterm_pipeline import curriculum_router as sg_curriculum_router
from skill_gap.longterm_pipeline import full_pipeline_router as sg_pipeline_router

logger = logging.getLogger("db_saver")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

TASKS: Dict[str, Dict[str, Any]] = globals().get("TASKS", {})
globals()["TASKS"] = TASKS

load_dotenv()

# -----------------------------------------------------------------------------
# NVIDIA / Ollama / throughput configuration
# -----------------------------------------------------------------------------
# For best results, set these in your shell/systemd service before starting
# Ollama/FastAPI. Ollama reads GPU settings when its server starts.
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ.setdefault("OLLAMA_NUM_GPU", "999")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
os.environ.setdefault("OLLAMA_FLASH_ATTN", "1")

PDF_CPU_WORKERS = int(os.getenv("PDF_CPU_WORKERS", str(max(2, min(8, os.cpu_count() or 4)))))
CURRICUNLP_WORKERS = int(os.getenv("CURRICUNLP_WORKERS", "2"))
CURRICUNLP_MAX_CHUNKS = int(os.getenv("CURRICUNLP_MAX_CHUNKS", "128"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

from recommendation_system.backend.routers.electives import router as electives_router
from recommendation_system.backend.routers.filters import router as filters_router
from recommendation_system.backend.routers.recommendations import router as recommendations_router



_http_session = requests.Session()
_http_adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=2)
_http_session.mount("http://", _http_adapter)
_http_session.mount("https://", _http_adapter)
_curricu_semaphore = Semaphore(max(1, CURRICUNLP_WORKERS))

def _bounded_workers(requested: Optional[int], default: int, cap: int = 64) -> int:
    try:
        value = int(requested or default)
    except Exception:
        value = default
    return max(1, min(value, cap))


def _ensure_text(value: Any) -> str:
    """Normalize PDF/extractor outputs to a plain string.

    Some PDF helpers return a list of page strings, while regex-based cleaners
    like _clean_pdf_text expect a string/bytes-like object.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple, set)):
        return "\n".join(_ensure_text(v) for v in value if v is not None)
    return str(value)

def _json_safe(obj, seen=None, path="root"):
    if seen is None:
        seen = {}

    if isinstance(obj, (dict, list, tuple, set, defaultdict)):
        obj_id = id(obj)
        if obj_id in seen:
            return f"<circular-reference to {seen[obj_id]}>"
        seen[obj_id] = path

    if isinstance(obj, defaultdict):
        obj = dict(obj)

    if isinstance(obj, dict):
        return {
            str(k): _json_safe(v, seen, f"{path}.{k}")
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            _json_safe(v, seen, f"{path}[{i}]")
            for i, v in enumerate(obj)
        ]

    if isinstance(obj, set):
        return [
            _json_safe(v, seen, f"{path}.set[{i}]")
            for i, v in enumerate(obj)
        ]

    return obj

def _find_pdf_path(pdf_name: str, curriculum_folder: str = "curriculum") -> str:
    if os.path.isabs(pdf_name) and os.path.exists(pdf_name):
        return pdf_name
    os.makedirs(curriculum_folder, exist_ok=True)
    matches = [
        f for f in os.listdir(curriculum_folder)
        if f.lower().endswith(".pdf") and pdf_name.lower() in f.lower()
    ]
    if not matches:
        raise HTTPException(status_code=404, detail="PDF not found")
    return os.path.join(curriculum_folder, matches[0])

def _extract_json_array(raw: str) -> List[Dict[str, Any]]:
    """
    Extract first JSON array from LLM output safely.
    """

    if not raw:
        return []

    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", raw)

    if not match:
        return []

    candidate = match.group(0)

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []

    return []

def _ollama_generate(prompt: str, model: Optional[str] = None, temperature: float = 0.0) -> str:
    base = (os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = _http_session.post(f"{base}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    return (r.json() or {}).get("response", "")

def _ollama_tags() -> Dict[str, Any]:
    base = (os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
    r = _http_session.get(f"{base}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json()

try:
    CURRICU_CLIENT = Client("marfoli/CurricuNLP")
except Exception as e:
    logging.warning(f"CurricuNLP client init failed (HF space may be down): {e}")
    CURRICU_CLIENT = None

CURRICUNLP_BASES = [
    "https://marfoli-curriculnlp.hf.space",
    "https://huggingface.co/spaces/marfoli/CurricuNLP",
]


def _parse_gradio_response(out) -> List[Dict]:
    if isinstance(out, list):
        if out and isinstance(out[0], str):
            try:
                j = json.loads(out[0])
                return j if isinstance(j, list) else [j]
            except Exception:
                return []
        return out

    if isinstance(out, dict) and "data" in out:
        data = out["data"]
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, list):
                return first
            if isinstance(first, dict):
                return [first]
            if isinstance(first, str):
                try:
                    j = json.loads(first)
                    return j if isinstance(j, list) else [j]
                except Exception:
                    return []
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return data
    return []


def _predict_curriculnlp_chunk(ch: str, retries: int = 2, pause: float = 0.05) -> List[Dict]:
    """Run one CurricuNLP chunk with bounded concurrency and retries."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            with _curricu_semaphore:
                client = CURRICU_CLIENT or Client("marfoli/CurricuNLP")
                res = client.predict(text=ch, api_name="/predict")

            time.sleep(pause)
            return _parse_gradio_response(res)

        except Exception as e:
            last_error = e
            time.sleep(0.4 * (attempt + 1))

    logger.warning("CurricuNLP chunk failed after retries: %s", last_error)
    return []

def _flatten_labels(items):
    flat = []
    for item in items or []:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(_flatten_labels(item))
    return flat


def call_curriculnlp_on_text(
        full_text: str,
        max_chars=40000,
        chunk_size=2000,
        overlap=250,
        pause=0.05,
        retries=2,
        workers: Optional[int] = None,
):
    full_text = _ensure_text(full_text)
    cleaned = _clean_pdf_text(full_text)
    chunks = _chunk_text(
        cleaned,
        chunk_size=chunk_size,
        overlap=overlap,
        limit=max_chars
    )[:CURRICUNLP_MAX_CHUNKS]

    if not chunks:
        return []

    max_workers = _bounded_workers(workers, CURRICUNLP_WORKERS, cap=8)
    out = []

    if max_workers == 1 or len(chunks) == 1:
        for ch in chunks:
            out.extend(_predict_curriculnlp_chunk(ch, retries=retries, pause=pause))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_predict_curriculnlp_chunk, ch, retries, pause)
                for ch in chunks
            ]
            for fut in as_completed(futures):
                out.extend(fut.result() or [])

    flat_out = _flatten_labels(out)
    return _merge_ner(flat_out)


WORLD_UNI_PATH = os.environ.get(
    "WORLD_UNI_PATH",
    "/app/world_universities_and_domains.json"
)
_world_uni_index: Optional[List[Dict]] = None


def _load_world_universities() -> List[Dict]:
    global _world_uni_index
    if _world_uni_index is None:
        try:
            with open(WORLD_UNI_PATH, "r", encoding="utf-8") as f:
                _world_uni_index = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load {WORLD_UNI_PATH}: {e}")
            _world_uni_index = []
    return _world_uni_index


def _find_uni_by_name(name: str) -> Dict[str, Optional[str]]:
    items = _load_world_universities()
    if not items or not name:
        return {"name": name, "country": None, "domain": None}
    names = [it.get("name", "") for it in items]
    if not names:
        return {"name": name, "country": None, "domain": None}
    best, score = process.extractOne(name, names)
    if score < 70:
        return {"name": name, "country": None, "domain": None}
    it = next(x for x in items if x.get("name") == best)
    domains = it.get("domains") or []
    primary_domain = (domains[0] if domains else None)
    return {"name": it.get("name"), "country": it.get("country"), "domain": primary_domain}

def split_course_blocks_per_page(pages: List[str]) -> List[str]:
    blocks = []

    for page_no, page_text in enumerate(pages or [], start=1):
        cleaned_page = _clean_pdf_text(_ensure_text(page_text))

        for block in split_course_blocks(cleaned_page):
            blocks.append(f"[PAGE {page_no}]\n{block}")

    return blocks

JOIN_SKILL_ON_COURSE = """
FROM Skill s
JOIN CourseSkill cs ON s.skill_id = cs.skill_id
JOIN Course c ON cs.course_id = c.course_id
JOIN University u ON c.university_id = u.university_id
"""

JOIN_COURSE_UNI = """
FROM Course c
JOIN University u ON c.university_id = u.university_id
"""

app = FastAPI(
    title="SkillCrawl API",
    version="0.1.3",
    description="API for skill extraction and course search (DB + domains JSON).",
    root_path="/curriculum-skills"
)

app.include_router(
    electives_router,
    prefix="/recommendation/electives",
    tags=["Recommendation"]
)

app.include_router(
    filters_router,
    prefix="/recommendation/filters",
    tags=["Recommendation"]
)

app.include_router(
    recommendations_router,
    prefix="/recommendation",
    tags=["Recommendation"]
)


# -----------------------------------------------------------------------------
# Recommendation endpoints backed directly by the SkillCrawl MySQL schema
# -----------------------------------------------------------------------------
# These endpoints recommend COURSES. Endpoints that expose/filter by degrees are
# preserved, but degree names are read from Course.degree_titles, not from
# DegreeProgram, so they work even when degree/program rows are sparse.
recommendation_filters_router = APIRouter(prefix="/recommendation/filters", tags=["Recommendation"])
recommendation_courses_router = APIRouter(prefix="/recommendation", tags=["Recommendation"])
recommendation_electives_router = APIRouter(prefix="/recommendation/electives", tags=["Recommendation"])


def _rec_conn():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Recommendation DB connection failed: {e}")


def _json_values(value: Any) -> List[str]:
    """Return clean string values from JSON/text fields such as Course.degree_titles."""
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
    else:
        parsed = value

    out: List[str] = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, (int, float)):
            out.append(str(x))
        elif isinstance(x, dict):
            for key in ("title", "name", "degree_title", "degree", "label", "value"):
                if key in x:
                    walk(x[key])
            if not any(k in x for k in ("title", "name", "degree_title", "degree", "label", "value")):
                for v in x.values():
                    walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)

    walk(parsed)
    return sorted({x for x in out if x})


def _course_row_to_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    degree_titles = _json_values(row.get("degree_titles"))
    skills = _json_values(row.get("skills_json"))
    ects = _json_values(row.get("ects_list"))
    mand_opt = _json_values(row.get("mand_opt_list"))
    msc_bsc = _json_values(row.get("msc_bsc_list"))

    return {
        "course_id": row.get("course_id"),
        "lesson_name": row.get("lesson_name"),
        "course_name": row.get("lesson_name"),
        "university_id": row.get("university_id"),
        "university_name": row.get("university_name"),
        "country": row.get("country"),
        "degree_titles": degree_titles,
        "degree_names": degree_titles,
        "language": row.get("language"),
        "website": row.get("website"),
        "semester_number": row.get("semester_number"),
        "semester_label": row.get("semester_label"),
        "ects": ects,
        "mand_opt": mand_opt,
        "msc_bsc": msc_bsc,
        "description": row.get("description"),
        "learning_outcomes": row.get("learning_outcomes"),
        "course_content": row.get("course_content"),
        "skills": skills,
        "score": float(row.get("score") or 0),
    }


def _fetch_courses(
    country: Optional[str] = None,
    university: Optional[str] = None,
    degree: Optional[str] = None,
    skill: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 50), 200))

    where = []
    params: List[Any] = []

    if country:
        where.append("LOWER(u.country) LIKE LOWER(%s)")
        params.append(f"%{country}%")

    if university:
        where.append("LOWER(u.university_name) LIKE LOWER(%s)")
        params.append(f"%{university}%")

    if degree:
        # Course.degree_titles is JSON but may contain arrays/objects/strings.
        where.append("LOWER(CAST(c.degree_titles AS CHAR)) LIKE LOWER(%s)")
        params.append(f"%{degree}%")

    if skill:
        where.append(
            """EXISTS (
                SELECT 1
                FROM CourseSkill cs2
                JOIN Skill s2 ON s2.skill_id = cs2.skill_id
                WHERE cs2.course_id = c.course_id
                  AND LOWER(s2.skill_name) LIKE LOWER(%s)
            )"""
        )
        params.append(f"%{skill}%")

    score_expr = "0"
    if query:
        like = f"%{query}%"
        where.append(
            """(
                LOWER(c.lesson_name) LIKE LOWER(%s)
                OR LOWER(COALESCE(c.description, '')) LIKE LOWER(%s)
                OR LOWER(COALESCE(c.learning_outcomes, '')) LIKE LOWER(%s)
                OR LOWER(COALESCE(c.course_content, '')) LIKE LOWER(%s)
                OR LOWER(CAST(c.degree_titles AS CHAR)) LIKE LOWER(%s)
                OR EXISTS (
                    SELECT 1
                    FROM CourseSkill cs3
                    JOIN Skill s3 ON s3.skill_id = cs3.skill_id
                    WHERE cs3.course_id = c.course_id
                      AND LOWER(s3.skill_name) LIKE LOWER(%s)
                )
            )"""
        )
        params.extend([like, like, like, like, like, like])
        score_expr = """
            (
                CASE WHEN LOWER(c.lesson_name) LIKE LOWER(%s) THEN 50 ELSE 0 END +
                CASE WHEN LOWER(CAST(c.degree_titles AS CHAR)) LIKE LOWER(%s) THEN 25 ELSE 0 END +
                CASE WHEN LOWER(COALESCE(c.learning_outcomes, '')) LIKE LOWER(%s) THEN 15 ELSE 0 END +
                CASE WHEN LOWER(COALESCE(c.description, '')) LIKE LOWER(%s) THEN 10 ELSE 0 END
            )
        """
        score_params = [like, like, like, like]
    else:
        score_params = []

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    sql = f"""
        SELECT
            c.course_id,
            c.university_id,
            c.lesson_name,
            c.language,
            c.website,
            c.semester_number,
            c.semester_label,
            c.ects_list,
            c.mand_opt_list,
            c.msc_bsc_list,
            c.description,
            c.learning_outcomes,
            c.course_content,
            c.degree_titles,
            u.university_name,
            u.country,
            COALESCE(JSON_ARRAYAGG(s.skill_name), JSON_ARRAY()) AS skills_json,
            {score_expr} AS score
        FROM Course c
        JOIN University u ON u.university_id = c.university_id
        LEFT JOIN CourseSkill cs ON cs.course_id = c.course_id
        LEFT JOIN Skill s ON s.skill_id = cs.skill_id
        {where_sql}
        GROUP BY
            c.course_id, c.university_id, c.lesson_name, c.language, c.website,
            c.semester_number, c.semester_label, c.ects_list, c.mand_opt_list,
            c.msc_bsc_list, c.description, c.learning_outcomes, c.course_content,
            c.degree_titles, u.university_name, u.country
        ORDER BY score DESC, u.university_name ASC, c.lesson_name ASC
        LIMIT %s
    """

    conn = _rec_conn()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, tuple(score_params + params + [limit]))
        return [_course_row_to_dict(r) for r in (cur.fetchall() or [])]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Recommendation query failed: {e}")
    finally:
        cur.close()
        conn.close()


@recommendation_filters_router.get("/filters/countries")
@recommendation_filters_router.get("/countries")
def recommendation_countries():
    conn = _rec_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT DISTINCT country
            FROM University
            WHERE country IS NOT NULL AND country <> ''
            ORDER BY country
        """)
        return {"countries": [r[0] for r in cur.fetchall()]}
    finally:
        cur.close()
        conn.close()


@recommendation_filters_router.get("/filters/universities")
@recommendation_filters_router.get("/universities")
def recommendation_universities(country: Optional[str] = Query(None)):
    conn = _rec_conn()
    cur = conn.cursor(dictionary=True)
    try:
        if country:
            cur.execute("""
                SELECT DISTINCT university_id, university_name, country
                FROM University
                WHERE LOWER(country) LIKE LOWER(%s)
                ORDER BY university_name
            """, (f"%{country}%",))
        else:
            cur.execute("""
                SELECT DISTINCT university_id, university_name, country
                FROM University
                ORDER BY university_name
            """)
        return {"universities": cur.fetchall()}
    finally:
        cur.close()
        conn.close()


@recommendation_filters_router.get("/filters/degrees")
@recommendation_filters_router.get("/degrees")
def recommendation_degrees(
    country: Optional[str] = Query(None),
    university: Optional[str] = Query(None),
):
    # Degree names intentionally come from Course.degree_titles.
    courses = _fetch_courses(country=country, university=university, limit=200)
    degrees = sorted({
        degree
        for course in courses
        for degree in (course.get("degree_titles") or [])
        if degree
    })
    return {"degrees": degrees, "source": "Course.degree_titles"}


@recommendation_filters_router.get("/filters/skills")
@recommendation_filters_router.get("/skills")
def recommendation_skills(
    country: Optional[str] = Query(None),
    university: Optional[str] = Query(None),
    degree: Optional[str] = Query(None),
):
    where = []
    params: List[Any] = []
    if country:
        where.append("LOWER(u.country) LIKE LOWER(%s)")
        params.append(f"%{country}%")
    if university:
        where.append("LOWER(u.university_name) LIKE LOWER(%s)")
        params.append(f"%{university}%")
    if degree:
        where.append("LOWER(CAST(c.degree_titles AS CHAR)) LIKE LOWER(%s)")
        params.append(f"%{degree}%")
    where_sql = "WHERE " + " AND ".join(where) if where else ""

    conn = _rec_conn()
    cur = conn.cursor()
    try:
        cur.execute(f"""
            SELECT DISTINCT s.skill_name
            FROM Skill s
            JOIN CourseSkill cs ON cs.skill_id = s.skill_id
            JOIN Course c ON c.course_id = cs.course_id
            JOIN University u ON u.university_id = c.university_id
            {where_sql}
            ORDER BY s.skill_name
        """, tuple(params))
        return {"skills": [r[0] for r in cur.fetchall() if r[0]]}
    finally:
        cur.close()
        conn.close()


@recommendation_courses_router.get("/courses")
@recommendation_courses_router.get("/recommendations/courses")
@recommendation_electives_router.get("/courses")
def recommend_courses(
    q: Optional[str] = Query(None, description="Text query matched against course text, degree names, and skills"),
    skill: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    university: Optional[str] = Query(None),
    degree: Optional[str] = Query(None, description="Matched against Course.degree_titles"),
    limit: int = Query(25, ge=1, le=200),
):
    courses = _fetch_courses(
        country=country,
        university=university,
        degree=degree,
        skill=skill,
        query=q,
        limit=limit,
    )
    return {
        "type": "course_recommendations",
        "count": len(courses),
        "filters": {
            "q": q,
            "skill": skill,
            "country": country,
            "university": university,
            "degree": degree,
            "degree_source": "Course.degree_titles",
        },
        "courses": courses,
    }


@recommendation_courses_router.get("/degrees")
@recommendation_courses_router.get("/recommendations/degrees")
def recommend_degrees_from_courses(
    q: Optional[str] = Query(None),
    skill: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    university: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    # Kept for compatibility: returns degree names, but calculates them from
    # matching courses and includes representative courses.
    courses = _fetch_courses(
        country=country,
        university=university,
        skill=skill,
        query=q,
        limit=limit,
    )

    grouped: Dict[str, Dict[str, Any]] = {}
    for course in courses:
        degree_names = course.get("degree_titles") or ["Unknown degree"]
        for degree_name in degree_names:
            item = grouped.setdefault(degree_name, {
                "degree_name": degree_name,
                "source": "Course.degree_titles",
                "course_count": 0,
                "courses": [],
            })
            item["course_count"] += 1
            if len(item["courses"]) < 10:
                item["courses"].append(course)

    degrees = sorted(grouped.values(), key=lambda x: (-x["course_count"], x["degree_name"]))
    return {
        "type": "degree_recommendations_from_courses",
        "count": len(degrees),
        "degrees": degrees,
    }


@recommendation_courses_router.post("/recommend")
@recommendation_courses_router.post("/recommendations")
@recommendation_electives_router.post("/recommend")
def recommend_courses_post(payload: Dict[str, Any] = Body(default_factory=dict)):
    q = payload.get("q") or payload.get("query") or payload.get("text")
    skill = payload.get("skill")
    country = payload.get("country")
    university = payload.get("university") or payload.get("university_name")
    degree = payload.get("degree") or payload.get("degree_name") or payload.get("degree_title")
    limit = int(payload.get("limit") or 25)

    courses = _fetch_courses(
        country=country,
        university=university,
        degree=degree,
        skill=skill,
        query=q,
        limit=limit,
    )
    return {
        "type": "course_recommendations",
        "count": len(courses),
        "courses": courses,
    }


app.include_router(recommendation_filters_router)
app.include_router(recommendation_courses_router)
app.include_router(recommendation_electives_router)

app.include_router(
    policy_router,
    tags=["Education Policy"]
)

app.include_router(skill_gap_router, prefix="/skill-gap")
app.include_router(sg_curriculum_router, prefix="/skill-gap")
app.include_router(sg_pipeline_router, prefix="/skill-gap")

class DebugPDFRequest(BaseModel):
    pdf_name: str
    run_ner: bool = False
    translate: bool = False
    ocr_if_short: bool = True
    ocr_max_pages: int = 20
    chunk_size: int = 1200
    overlap: int = 150
    max_chars: int = 40000
    sample_chars: int = 600


class Organization(BaseModel):
    name: str
    location: str


class LabelItem(BaseModel):
    class_or_confidence: str = Field(...,
                                     description="NER label name (e.g., 'lesson_name', 'ects', 'language', 'professor').")
    token: str = Field(..., description="Extracted text value for the label.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional model confidence in [0,1].")


class CourseLabels(BaseModel):
    lesson_name: Optional[str] = Field(None, description="Course/lesson title, if known.")
    website: Optional[AnyUrl] = Field(None, description="Canonical course URL.")
    labels: List[LabelItem] = Field(default_factory=list, description="CurricuNLP output for this course/page.")


class SaveLabelsRequest(BaseModel):
    university_name: str = Field(..., description="Normalized university name.")
    country: Optional[str] = Field(None, description="Country name; defaults to 'Unknown' if omitted.")
    courses: List[CourseLabels] = Field(default_factory=list, description="List of courses with label arrays.")


class CleanRequest(BaseModel):
    folder: str = Field(..., description="Folder containing JSON files with degree_titles")
    backend: str = Field("openai", description="LLM backend: openai or ollama")
    model: Optional[str] = Field(None, description="Model name (default: gpt-4o-mini for OpenAI, llama3.1 for Ollama)")
    inplace: bool = Field(False, description="Overwrite files in-place")
    outdir: Optional[str] = Field(None, description="Output directory if not inplace (default: <folder>/_cleaned)")
    dry_run: bool = Field(False, description="Run without writing files")
    workers: Optional[int] = Field(None, description="Parallel file cleaners. For Ollama keep this close to OLLAMA_NUM_PARALLEL.")

    class Config:
        schema_extra = {
            "examples": {
                "basic": {
                    "summary": "Write cleaned JSONs into <folder>/_cleaned",
                    "description": "Uses OpenAI gpt-4o-mini by default.",
                    "value": {
                        "folder": "crawler_json",
                        "backend": "openai",
                        "model": "gpt-4o-mini",
                        "inplace": False,
                        "outdir": None,
                        "dry_run": False
                    }
                },
                "inplace": {
                    "summary": "Overwrite files in-place",
                    "description": "Uses Ollama llama3.1 and writes directly into the source folder.",
                    "value": {
                        "folder": "crawler_json",
                        "backend": "ollama",
                        "model": "llama3.1",
                        "inplace": True,
                        "dry_run": False
                    }
                },
                "dry-run": {
                    "summary": "Preview cleaning without saving",
                    "description": "Runs in dry-run mode, only returns summary counts.",
                    "value": {
                        "folder": "crawler_json",
                        "backend": "openai",
                        "inplace": False,
                        "dry_run": True
                    }
                }
            }
        }


class FileSummary(BaseModel):
    file: str
    original_count: int
    kept_count: int
    removed_count: int
    removed_examples: List[str]


class CleanTotals(BaseModel):
    files: int
    titles_before: int
    titles_after: int
    removed: int


class CleanResponse(BaseModel):
    totals: CleanTotals
    summaries: List[FileSummary]
    output_dir: Optional[str] = None


class SaveJSONRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict,
                                    description="Arbitrary JSON to store (must contain 'university_name' unless normalize_university=true).")
    normalize_university: bool = Field(False,
                                       description="If true, try to fill university_name/country from a guessed name via _find_uni_by_name().")


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


class LabelItem(BaseModel):
    class_or_confidence: str = Field(..., description="Label name, e.g. lesson_name, ects, language, professor")
    token: str = Field(..., description="Extracted value for the label")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Optional confidence")


class CourseLabels(BaseModel):
    lesson_name: Optional[str] = Field(None, description="Course title, if known")
    website: Optional[AnyUrl] = Field(None, description="Canonical course URL")
    labels: List[LabelItem] = Field(default_factory=list, description="CurricuNLP output for this course/page")


class SaveLabelsRequest(BaseModel):
    university_name: str = Field(..., description="Normalized university name")
    country: Optional[str] = Field(None, description="Country name")
    courses: List[CourseLabels] = Field(default_factory=list, description="Courses with label arrays")


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


class ApifyCrawlRequest(BaseModel):
    start_urls: List[str]


class CountryTrend(BaseModel):
    country: str
    monthly_counts: List[MonthlyTrend]


class SaveJSONDirRequest(BaseModel):
    directory: str = Field(..., description="Directory containing JSON files to import")
    filename_pattern: str = Field("*.json", description="Glob pattern for files (e.g. *.json)")
    recursive: bool = Field(False, description="Recurse into subdirectories")
    normalize_university: bool = Field(True,
                                       description="Fill university_name/country via _find_uni_by_name if possible")
    stop_on_error: bool = Field(False, description="Stop at first error (default: continue)")
    limit: Optional[int] = Field(None, ge=1, description="Import at most this many files")
    dry_run: bool = Field(False, description="Validate and preview only; do not write to DB")
    workers: Optional[int] = Field(None, ge=1, description="Parse workers (threads unless use_process_pool=True)")
    db_workers: Optional[int] = Field(2, ge=1, le=64, description="Concurrent DB writers")
    use_process_pool: bool = Field(False, description="Use ProcessPoolExecutor for CPU-bound parsing/merging")


class CurricuNLPTextRequest(BaseModel):
    text: str
    max_chars: Optional[int] = 40000


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
    max_chars: int = 40000
    chunk_size: int = 2000
    overlap: int = 150
    workers: Optional[int] = None
    use_ollama_summary: bool = False
    ollama_model: Optional[str] = None


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


def _parse_one_file_for_save_json_dir(fp: str, base_dir: str, normalize_university: bool):
    with open(fp, "r", encoding="utf-8") as f:
        payload = _json_load(f)
    if not isinstance(payload, (dict, list)):
        raise ValueError("Top-level JSON must be an object or labels list")

    file_base_hint = os.path.splitext(os.path.basename(fp))[0]
    if isinstance(payload, dict):
        uni_guess = (
                payload.get("university_name")
                or payload.get("university")
                or (payload.get("university_meta") or {}).get("name")
        )
    else:
        uni_guess = None
    if not uni_guess:
        uni_guess = file_base_hint

    if normalize_university:
        meta = _find_uni_by_name(uni_guess)
        uni_name = (payload.get("university_name") if isinstance(payload, dict) else None) or meta.get(
            "name") or uni_guess
        country = (
                (payload.get("country") if isinstance(payload, dict) else None)
                or (payload.get("university_country") if isinstance(payload, dict) else None)
                or meta.get("country")
                or "Unknown"
        )
    else:
        meta = {"name": uni_guess, "country": (payload.get("country") if isinstance(payload, dict) else None)}
        uni_name = (payload.get("university_name") if isinstance(payload, dict) else None) or uni_guess
        country = (payload.get("country") if isinstance(payload, dict) else None) or "Unknown"

    if isinstance(payload, dict) and isinstance(payload.get("courses"), list) and payload["courses"]:
        courses = payload["courses"]
        source = "payload.courses"
    else:
        flat_course = None
        if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
            flat_course = _course_from_curriculnlp_labels_payload(payload, filename_hint=file_base_hint)
            source = "labels->course"
        elif isinstance(payload, list) and payload and all(
                isinstance(x, dict) and "class_or_confidence" in x for x in payload):
            flat_course = _course_from_curriculnlp_labels_payload({"labels": payload}, filename_hint=file_base_hint)
            source = "labels(list)->course"
        elif isinstance(payload, dict):
            course_like = {
                "lesson_name", "title", "name", "website", "url", "description", "objectives", "learning_outcomes",
                "course_content", "assessment", "exam", "prerequisites", "general_competences", "educational_material",
                "ects", "language", "professor", "professors", "hours", "msc_bsc", "msc_bsc_list", "degree_title",
                "degree_titles",
                "semester_number", "semester_label", "mand_opt", "fee_list", "extras", "year", "attendance_type",
                "attendence_type"
            }
            if course_like & set(payload.keys()):
                flat_course = dict(payload)
                if not isinstance(flat_course.get("lesson_name"), str) or not flat_course["lesson_name"].strip():
                    flat_course["lesson_name"] = (flat_course.get("title") or flat_course.get(
                        "name") or file_base_hint or "Untitled Course")[:255]
                if not flat_course.get("website") and flat_course.get("url"):
                    flat_course["website"] = flat_course["url"]
                if flat_course.get("attendence_type") and not flat_course.get("attendance_type"):
                    flat_course["attendance_type"] = flat_course["attendence_type"]
                source = "course-like-dict"
            else:
                source = "unknown"
        courses = [flat_course] if flat_course else []

    merged_courses = _prepare_and_merge_courses(courses, uni_name, file_hint=file_base_hint, fuzzy_threshold=88)

    rel = os.path.relpath(fp, base_dir)
    return {
        "rel": rel,
        "uni_name": uni_name,
        "country": country,
        "courses": merged_courses,
        "source": source,
        "raw_count": len(courses),
        "merged_count": len(merged_courses),
    }


def _save_json_dir_task(task_id: str, req_data: dict):
    task = TASKS.get(task_id, {})

    def set_status(**kw):
        task.update(kw)
        TASKS[task_id] = task

    try:
        set_status(status="running", started_at=time.time(), processed_files=0, total_files=0)
        dry_run = bool(req_data.get("dry_run"))

        if not dry_run and not is_database_connected(DB_CONFIG):
            raise RuntimeError("Database connection failed.")

        base_dir = os.path.abspath(os.path.expanduser(req_data["directory"]))
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Directory not found: {base_dir}")

        pattern = req_data.get("filename_pattern") or "*.json"
        recursive = bool(req_data.get("recursive"))
        glob_pat = os.path.join(base_dir, "**", pattern) if recursive else os.path.join(base_dir, pattern)
        files = [f for f in glob.glob(glob_pat, recursive=recursive) if os.path.isfile(f)]
        files.sort()
        if req_data.get("limit"):
            files = files[: int(req_data["limit"])]

        total = len(files)
        set_status(total_files=total)
        logger.info("[%s] Scanning %d files (pattern=%s, recursive=%s)", task_id, total, pattern, recursive)

        if not files:
            set_status(status="succeeded", finished_at=time.time(),
                       result={"message": "No files matched the pattern."})
            return

        normalize_university = bool(req_data.get("normalize_university"))
        stop_on_error = bool(req_data.get("stop_on_error"))
        use_process_pool = bool(req_data.get("use_process_pool"))
        workers = int(req_data.get("workers") or max(2, (os.cpu_count() or 4)))
        db_workers = int(req_data.get("db_workers") or 2)

        parse_fn = partial(
            _parse_one_file_for_save_json_dir,
            base_dir=base_dir,
            normalize_university=normalize_university
        )

        Executor = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor

        per_uni = {}
        parse_errors = []
        processed = 0

        def _flush_to_db():
            nonlocal per_uni
            if dry_run or not per_uni:
                return

            for uni, g in per_uni.items():
                g["courses"] = _prepare_and_merge_courses(g["courses"], uni, file_hint=None, fuzzy_threshold=88)

            with ThreadPoolExecutor(max_workers=db_workers) as dbex:
                futs = {
                    dbex.submit(write_json_to_database, payload, DB_CONFIG): uni
                    for (uni, payload) in per_uni.items()
                }
                for fut in as_completed(futs):
                    try:
                        fut.result()
                    except Exception as e:
                        logger.exception("[%s] Error writing %s: %s", task_id, futs[fut], e)
            per_uni = {}

        with Executor(max_workers=workers) as ex:
            futures = {ex.submit(parse_fn, fp): fp for fp in files}
            for fut in as_completed(futures):
                fp = futures[fut]
                rel = os.path.relpath(fp, base_dir)
                try:
                    res = fut.result()
                except Exception as e:
                    logger.exception("[%s] Error on %s: %s", task_id, rel, e)
                    parse_errors.append({"file": rel, "status": "error", "error": str(e)})
                    if stop_on_error:
                        break
                    processed += 1
                    set_status(processed_files=processed, last_file=rel)
                    continue

                uni_name = res["uni_name"]
                entry = per_uni.setdefault(uni_name, {
                    "university_name": uni_name,
                    "country": res["country"],
                    "courses": []
                })
                entry["courses"].extend(res["courses"])

                processed += 1
                set_status(processed_files=processed, last_file=rel)

                if processed % 1000 == 0:
                    _flush_to_db()

        _flush_to_db()

        summary = {
            "directory": base_dir,
            "pattern": pattern,
            "recursive": recursive,
            "dry_run": dry_run,
            "imported_files": processed,
            "errors": parse_errors,
            "failed": len(parse_errors)
        }
        set_status(status="succeeded", finished_at=time.time(), result=summary)

    except Exception as e:
        set_status(status="failed", finished_at=time.time(), error=str(e))
        logger.exception("[%s] Task crashed: %s", task_id, e)


def get_tracker_token() -> str:
    login_url = f"{os.environ['API_TRACKER_BASE_URL']}/login"
    payload = {"username": os.getenv("TRACKER_USERNAME"), "password": os.getenv("TRACKER_PASSWORD")}
    headers = {"Content-Type": "application/json", "accept": "application/json"}
    try:
        response = requests.post(login_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        token = response.text.strip().strip('"')
        return token
    except Exception as e:
        print(f"[ERROR] Login failed: {e}")
        return ""


def _save_payload_task(task_id: str, payload: dict):
    task = TASKS.get(task_id)
    if not task:
        return
    task.update({"status": "running", "started_at": time.time()})
    try:
        logger.info("[%s] Saving %d courses for %s",
                    task_id, len(payload.get("courses", [])),
                    payload.get("university_name"))
        result = write_json_to_database(payload, DB_CONFIG)
        task.update({
            "status": "succeeded",
            "finished_at": time.time(),
            "result": result,
            "saved_courses": len(payload.get("courses", []))
        })
        logger.info("[%s] Done", task_id)
    except Exception as e:
        task.update({"status": "failed", "finished_at": time.time(), "error": str(e)})
        logger.exception("[%s] Ingest failed: %s", task_id, e)


@app.get("/health", tags=["Meta"])
def health_check():
    return {"status": "running"}


class RenameSkillsFromESCOResponse(BaseModel):
    status: str
    checked_skills: int
    updated_skills: int
    missing_in_csv: int
    dry_run: bool
    examples: List[Dict[str, Any]]

class RenameSkillsFromESCOResponse(BaseModel):
    status: str
    checked_skills: int
    updated_skills: int
    missing_in_csv: int
    dry_run: bool
    examples: List[Dict[str, Any]]


@app.post(
    "/skills/rename_from_esco_csv",
    response_model=RenameSkillsFromESCOResponse,
    tags=["Skills"],
    summary="Rename Skill.skill_name using local ESCO CSV and skill_url",
)
def rename_skills_from_esco_csv(
    dry_run: bool = Query(
        True,
        description="Preview changes without updating database"
    ),
    limit: Optional[int] = Query(
        None,
        ge=1,
        description="Limit number of checked skills"
    ),
):
    """
    Updates Skill.skill_name from local ESCO CSV using skill_url.
    Also stores alternative labels if your DB has alt_labels column.
    """

    print("[ESCO-RENAME] Loading local ESCO CSV...", flush=True)

    esco_labels = _load_esco_skill_labels()

    print(
        f"[ESCO-RENAME] Loaded {len(esco_labels)} ESCO labels.",
        flush=True
    )

    conn = None
    cur = None

    checked = 0
    updated = 0
    missing = 0

    examples = []

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)

        sql = """
            SELECT
                skill_id,
                skill_url,
                skill_name
            FROM Skill
            WHERE skill_url IS NOT NULL
              AND skill_url != ''
        """

        if limit:
            sql += f" LIMIT {int(limit)}"

        cur.execute(sql)

        skills = cur.fetchall() or []

        total = len(skills)

        print(
            f"[ESCO-RENAME] Found {total} Skill rows to check.",
            flush=True
        )

        update_cur = conn.cursor()

        for idx, row in enumerate(skills, start=1):
            skill_id = row["skill_id"]
            skill_url = (row.get("skill_url") or "").strip()
            old_name = (row.get("skill_name") or "").strip()

            checked += 1

            if not skill_url:
                continue

            meta = esco_labels.get(skill_url)

            if not meta:
                missing += 1

                print(
                    f"[MISSING] {idx}/{total} | "
                    f"skill_id={skill_id} | "
                    f"url={skill_url}",
                    flush=True
                )

                continue

            new_name = (meta.get("label") or "").strip()

            if not new_name:
                missing += 1
                continue

            if old_name == new_name:
                continue

            examples.append({
                "skill_id": skill_id,
                "old": old_name,
                "new": new_name,
            })

            print(
                f"[UPDATE] {idx}/{total} | "
                f"skill_id={skill_id} | "
                f"{old_name} -> {new_name}",
                flush=True
            )

            # Check if another row already has the target name + same URL
            update_cur.execute("""
                SELECT skill_id
                FROM Skill
                WHERE skill_url = %s
                AND skill_name = %s
                AND skill_id <> %s
                LIMIT 1
            """, (skill_url, new_name, skill_id))

            duplicate = update_cur.fetchone()

            if duplicate:
                keep_skill_id = duplicate[0]
                old_skill_id = skill_id

                print(
                    f"[MERGE] skill_id={old_skill_id} -> {keep_skill_id} | {new_name}",
                    flush=True
                )

                if not dry_run:
                    # Move course links from duplicate skill to existing skill
                    update_cur.execute("""
                        INSERT IGNORE INTO CourseSkill (course_id, skill_id, categories)
                        SELECT course_id, %s, categories
                        FROM CourseSkill
                        WHERE skill_id = %s
                    """, (keep_skill_id, old_skill_id))

                    # Remove old links
                    update_cur.execute("""
                        DELETE FROM CourseSkill
                        WHERE skill_id = %s
                    """, (old_skill_id,))

                    # Remove duplicate Skill row
                    update_cur.execute("""
                        DELETE FROM Skill
                        WHERE skill_id = %s
                    """, (old_skill_id,))

                    updated += 1

                continue

            # No duplicate exists, safe rename
            if not dry_run:
                update_cur.execute("""
                    UPDATE Skill
                    SET skill_name = %s
                    WHERE skill_id = %s
                """, (new_name, skill_id))

                updated += 1

        if not dry_run:
            conn.commit()

        print(
            f"[ESCO-RENAME] DONE | "
            f"checked={checked} | "
            f"updated={updated} | "
            f"missing={missing} | "
            f"dry_run={dry_run}",
            flush=True
        )

        return {
            "status": "completed",
            "checked_skills": checked,
            "updated_skills": updated,
            "missing_in_csv": missing,
            "dry_run": dry_run,
            "examples": examples[:50],
        }

    except Exception as e:
        print(f"[ESCO-RENAME-FAILED] {e}", flush=True)
        logger.exception("ESCO rename failed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass

        try:
            if conn:
                conn.close()
        except Exception:
            pass

@app.post("/nlp/curriculnlp", tags=["CurricuNLP"], summary="Run CurricuNLP on raw text and return labels")
def curriculnlp_labels(req: CurricuNLPTextRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    try:
        out = CURRICU_CLIENT.predict(text=text, api_name="/predict")
        labels = out if isinstance(out, list) else []
        return {"labels": labels}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"CurricuNLP call failed: {e}")


@app.post("/crawl/start", tags=["CurricuNLP"], status_code=202)
async def start_apify_crawl(request: ApifyCrawlRequest, background_tasks: BackgroundTasks):
    """
    Starts the Apify/Crawlee crawler in the background with a list of seed URLs.
    """
    if not request.start_urls:
        raise HTTPException(status_code=400, detail="start_urls list cannot be empty.")

    background_tasks.add_task(asyncio.run, run_apify_crawler(request.start_urls))

    return {
        "status": "accepted",
        "message": "Crawler process started in the background.",
        "started_urls": request.start_urls
    }


@app.post("/import/crawler_json", tags=["CurricuNLP"], summary="Import crawler JSON file(s) into the database")
def import_crawler_json(
        directory: str = Query("crawler_json", description="Directory containing crawler JSON files"),
        filename: Optional[str] = Query(None,
                                        description="Specific JSON file to import. If omitted, imports ALL *.json in directory")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    def _import_one_json(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            return {"file": os.path.basename(path), "status": "error", "error": f"Failed to read JSON: {e}"}

        uni_guess = (
                payload.get("university_name")
                or payload.get("university")
                or (payload.get("university_meta") or {}).get("name")
                or re.sub(r"[_\W]+", " ", os.path.splitext(os.path.basename(path))[0]).strip()
        )
        meta = _find_uni_by_name(uni_guess)
        payload["university_name"] = payload.get("university_name") or meta.get("name") or uni_guess
        payload["country"] = payload.get("country") or payload.get("university_country") or meta.get(
            "country") or "Unknown"

        try:
            write_json_to_database(payload, DB_CONFIG)
            return {"file": os.path.basename(path), "status": "imported", "university": payload["university_name"],
                    "country": payload["country"]}
        except Exception as e:
            return {"file": os.path.basename(path), "status": "error", "error": str(e)}

    if filename:
        candidate = filename if os.path.isabs(filename) else os.path.join(directory, filename)
        if not os.path.exists(candidate) and not candidate.lower().endswith(".json"):
            cand_json = candidate + ".json"
            if os.path.exists(cand_json):
                candidate = cand_json
        if not os.path.exists(candidate):
            raise HTTPException(status_code=404, detail=f"JSON file not found: {candidate}")
        result = _import_one_json(candidate)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["error"])
        return {"imported": [result], "success": 1, "failed": 0}

    if not os.path.isdir(directory):
        raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".json")]
    if not files:
        return {"imported": [], "errors": [], "count": {"success": 0, "failed": 0}, "message": "No JSON files found."}

    results = [_import_one_json(fp) for fp in files]
    ok = [r for r in results if r["status"] == "imported"]
    errs = [r for r in results if r["status"] == "error"]
    return {"imported": ok, "errors": errs, "count": {"success": len(ok), "failed": len(errs)}}


@app.get("/debug_one_skill", tags=["Debug"])
def debug_one():
    token = get_tracker_token()
    headers = {"Authorization": f"Bearer {token}", "accept": "application/json"}
    payload = {'ids': ["http://data.europa.eu/esco/skill/ccd0a1d9-afda-43d9-b901-96344886e14d"]}
    r = requests.post(
        f"{os.environ['API_TRACKER_BASE_URL']}/skills",
        headers=headers,
        json=payload,
        verify=False
    )
    return r.json()


@app.post("/crawl/run_and_import", tags=["CurricuNLP"], summary="Run Apify crawler, then import its JSON outputs")
async def run_and_import(
        request: ApifyCrawlRequest,
        outdir: str = Query("crawler_json", description="Directory where crawler writes JSONs")
):
    if not request.start_urls:
        raise HTTPException(status_code=400, detail="start_urls list cannot be empty.")
    await run_apify_crawler(request.start_urls, outdir=outdir)

    if not os.path.isdir(outdir):
        raise HTTPException(status_code=404, detail=f"Output directory not found: {outdir}")
    files = [os.path.join(outdir, f) for f in os.listdir(outdir) if f.lower().endswith(".json")]
    ok, errs = [], []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            uni_guess = (
                    payload.get("university_name")
                    or payload.get("university")
                    or (payload.get("university_meta") or {}).get("name")
                    or os.path.splitext(os.path.basename(fp))[0]
            )
            meta = _find_uni_by_name(uni_guess)
            payload["university_name"] = payload.get("university_name") or meta.get("name") or uni_guess
            payload["country"] = payload.get("country") or payload.get("university_country") or meta.get(
                "country") or "Unknown"

            write_json_to_database(payload, DB_CONFIG)
            ok.append({"file": os.path.basename(fp), "status": "imported"})
        except Exception as e:
            errs.append({"file": os.path.basename(fp), "status": "error", "error": str(e)})
    return {"imported": ok, "errors": errs, "count": {"success": len(ok), "failed": len(errs)}}


@app.get("/list_pdfs", tags=["PDF"])
def list_pdfs():
    curriculum_folder = "curriculum"
    os.makedirs(curriculum_folder, exist_ok=True)
    pdf_files = []
    for f in os.listdir(curriculum_folder):
        if f.endswith(".pdf"):
            filename = f.replace(".pdf", "")
            university_name = re.sub(r"[_\W]+", " ", filename).strip()
            meta = _find_uni_by_name(university_name)
            pdf_files.append({
                "filename": f,
                "university_name": meta.get("name") or university_name,
                "university_country": meta.get("country"),
                "domain": meta.get("domain")
            })
    return {"pdf_files": pdf_files}


@app.post("/process_pdf", tags=["PDF"])
def process_pdf(request: PDFProcessingRequest):
    pdf_path = _find_pdf_path(request.pdf_name)
    full_text = _ensure_text(extract_text_from_pdf_best(pdf_path))

    university_name_guess = re.sub(r"[_\W]+", " ", os.path.basename(pdf_path).replace(".pdf", "")).strip()
    meta = _find_uni_by_name(university_name_guess)
    university_name = meta.get("name") or university_name_guess
    university_country = meta.get("country")
    domain = meta.get("domain")

    labels = call_curriculnlp_on_text(
        full_text,
        max_chars=request.max_chars,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        workers=request.workers,
    )

    response = {
        "file": os.path.basename(pdf_path),
        "university_meta": {"name": university_name, "country": university_country, "domain": domain},
        "gpu_config": {
            "ollama_host": os.getenv("OLLAMA_HOST"),
            "ollama_num_gpu": os.getenv("OLLAMA_NUM_GPU"),
            "ollama_num_parallel": os.getenv("OLLAMA_NUM_PARALLEL"),
            "curriculnlp_workers": _bounded_workers(request.workers, CURRICUNLP_WORKERS, cap=8),
        },
        "labels": labels
    }

    if request.use_ollama_summary:
        prompt = (
            "Extract a concise curriculum summary from the following PDF text. "
            "Return only valid JSON with keys: university, likely_programs, course_topics, notes.\n\n"
            + _clean_pdf_text(full_text)[:12000]
        )
        response["ollama_summary"] = _ollama_generate(prompt, model=request.ollama_model)

    return JSONResponse(content=_json_safe(response))


@app.get("/gpu/ollama_status", tags=["GPU", "Ollama"])
def ollama_status():
    try:
        return {
            "status": "ok",
            "ollama": _ollama_tags(),
            "env": {
                "OLLAMA_HOST": os.getenv("OLLAMA_HOST"),
                "OLLAMA_NUM_GPU": os.getenv("OLLAMA_NUM_GPU"),
                "OLLAMA_NUM_PARALLEL": os.getenv("OLLAMA_NUM_PARALLEL"),
                "OLLAMA_MAX_LOADED_MODELS": os.getenv("OLLAMA_MAX_LOADED_MODELS"),
                "OLLAMA_FLASH_ATTN": os.getenv("OLLAMA_FLASH_ATTN"),
            },
            "note": "Also run `nvidia-smi` while processing to confirm GPU utilization."
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama is not reachable: {e}")


@app.get("/filter_skillnames", tags=["Skills"])
def get_skills_endpoint(
        university_name: str = Query(..., description="University name"),
        lesson_name: str = Query(..., description="Lesson name"),
        match: Literal["exact", "like", "prefix"] = Query("exact", description="Match mode: exact | like | prefix")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    if match == "exact":
        uni_pred = "LOWER(u.university_name) = LOWER(%s)"
        les_pred = "LOWER(c.lesson_name) = LOWER(%s)"
        params = (university_name, lesson_name)
    elif match == "prefix":
        uni_pred = "LOWER(u.university_name) LIKE LOWER(%s)"
        les_pred = "LOWER(c.lesson_name) LIKE LOWER(%s)"
        params = (f"{university_name}%", f"{lesson_name}%")
    else:
        uni_pred = "LOWER(u.university_name) LIKE LOWER(%s)"
        les_pred = "LOWER(c.lesson_name) LIKE LOWER(%s)"
        params = (f"%{university_name}%", f"%{lesson_name}%")

    sql = f"""
        SELECT DISTINCT s.skill_name
        FROM Course c
        JOIN University u ON c.university_id = u.university_id
        LEFT JOIN CourseSkill cs ON c.course_id = cs.course_id
        LEFT JOIN Skill s ON cs.skill_id = s.skill_id
        WHERE {uni_pred}
          AND {les_pred}
          AND s.skill_name IS NOT NULL
          AND s.skill_name <> ''
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        names = sorted({row[0] for row in cursor.fetchall()})
        return {"match": match, "Data": names}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


@app.get("/db/tasks/{task_id}", tags=["Debug"])
def get_task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return JSONResponse(content=_json_safe(task))


FIELD_TO_CATEGORY_DEFAULT = [
    ("description", "description"),
    ("objectives", "objectives"),
    ("learning_outcomes", "learning_outcomes"),
    ("course_content", "course_content"),
    ("assessment", "assessment"),
    ("exam", "exam"),
    ("prerequisites", "prerequisites"),
    ("general_competences", "general_competences"),
    ("educational_material", "educational_material"),
]


def _extract_urls_for_course(course: dict, field_to_category=FIELD_TO_CATEGORY_DEFAULT) -> tuple:
    """
    Returns (course_id, title, url_to_categories: Dict[url, Set[str]])
    """
    cid = course["course_id"]
    title = course.get("lesson_name") or f"course_{cid}"
    url_to_categories: Dict[str, Set[str]] = defaultdict(set)

    def _extract_urls_from_text(text: str) -> Set[str]:
        t = (text or "").strip()
        if not t:
            return set()

        base = os.getenv("API_SKILL_EXTRACTOR_BASE_URL")
        if not base:
            logger.error("API_SKILL_EXTRACTOR_BASE_URL is missing")
            return set()

        extractor_url = base.rstrip("/") + "/extract-skills"

        try:
            resp = _http_session.post(
                extractor_url,
                headers={"Content-Type": "application/json", "accept": "application/json"},
                json=[t],
                verify=False,
                timeout=60
            )

            if not resp.ok:
                logger.error("Skill extractor failed: %s %s", resp.status_code, resp.text[:500])
                return set()

            data = resp.json()
            urls = set()

            if isinstance(data, list):
                for group in data:
                    if isinstance(group, list):
                        urls.update(su for su in group if isinstance(su, str))
                    elif isinstance(group, str):
                        urls.add(group)

            elif isinstance(data, dict):
                for it in data.get("items", []):
                    ids = it.get("id") or it.get("ids") or []
                    if isinstance(ids, str):
                        urls.add(ids)
                    elif isinstance(ids, list):
                        urls.update(su for su in ids if isinstance(su, str))

            logger.info("Extractor returned %d skill URLs", len(urls))
            return urls

        except Exception as e:
            logger.exception("Skill extraction failed")
            return set()
    
    for field, cat in field_to_category:
        urls = _extract_urls_from_text(course.get(field))
        for u in urls:
            url_to_categories[u].add(cat)

    return cid, title, url_to_categories


def _resolve_urls_to_names(all_urls: Set[str], token: str) -> Dict[str, dict]:
    """
    Batch-resolve URLs -> {label, esco_id, level} using the tracker API.
    Returns dict[url] = {"label": str, "esco_id": str|None, "level": str|None}
    """
    if not all_urls:
        return {}
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}"
    }
    data = "&".join([f"ids={requests.utils.quote(u)}" for u in all_urls])
    out: Dict[str, dict] = {}
    try:
        TRACKER_BASE = os.environ.get("API_TRACKER_BASE_URL", "https://skillab-tracker.csd.auth.gr/api").rstrip("/")
        res = requests.post(
            f"{TRACKER_BASE}/skills?page=1",
            headers=headers, data=data, verify=False, timeout=60
        )
        if not res.ok:
            return {}
        for item in res.json().get("items", []):
            su = item.get("id")
            sn = item.get("label") or (item.get("alternative_labels", [None])[0])
            level_value = item.get("level") or item.get("skill_level") or item.get("skillLevel")
            if isinstance(level_value, (int, float)):
                level_value = str(int(level_value))
            elif isinstance(level_value, str):
                level_value = level_value.strip() or None

            esco_id = item.get("esco_id") or item.get("code")
            if not esco_id and su:
                m = re.search(r"/esco/(?:skill|knowledge|competence)/([^/?#]+)", su or "")
                if m:
                    esco_id = m.group(1)
                else:
                    esco_id = su.rsplit("/", 1)[-1] if (su and "/" in su) else su
            if su and sn:
                out[su] = {"label": sn, "esco_id": esco_id, "level": level_value}
        return out
    except Exception:
        return {}


def _db_write_course_skills(
        course_id: int,
        url_to_categories: Dict[str, Set[str]],
        url_meta: Dict[str, dict],
        level_min: Optional[int] = None,
        level_max: Optional[int] = None,
) -> List[str]:
    """
    Opens its own DB connection, upserts skills for one course, closes connection.
    Returns list of skill names for that course.
    Only upserts a skill if its ESCO 'level' fits into the specified range (if provided).
    """
    names: List[str] = []
    if not url_to_categories or not url_meta:
        return names

    conn = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        for su, cats in url_to_categories.items():
            meta = url_meta.get(su)
            if not meta:
                continue

            sn = meta.get("label")
            esco_id = meta.get("esco_id")
            level_value = meta.get("level")

            if level_value is not None:
                try:
                    lv = int(level_value)
                    if level_min is not None and lv < level_min:
                        continue
                    if level_max is not None and lv > level_max:
                        continue
                except (TypeError, ValueError):
                    if level_min is not None or level_max is not None:
                        continue

            if sn:
                names.append(sn)
                upsert_skill_and_link_with_categories(
                    conn, course_id, su, sn, esco_id, level_value, sorted(cats or ["description"])
                )
        return sorted(set(n for n in names if n))
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _calc_all_skillnames_task_parallel(
        master_task_id: str,
        lesson_name: Optional[str],
        match: Literal["exact", "like"],
        workers: int,
        min_skill_level: Optional[int] = None,
        max_skill_level: Optional[int] = None
):
    task = TASKS.get(master_task_id, {})
    lock = Lock()

    def set_status(**kw):
        with lock:
            task.update(kw)
            TASKS[master_task_id] = task

    set_status(status="running", started_at=time.time(), processed=0)

    if not is_database_connected(DB_CONFIG):
        set_status(status="failed", finished_at=time.time(), error="Database connection failed.")
        logger.error("[%s] DB connection failed in master task", master_task_id)
        return

    conn = None
    cur = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT university_name FROM University ORDER BY university_name ASC")
        rows = cur.fetchall() or []
        uni_names = [r["university_name"] for r in rows if (r.get("university_name") or "").strip()]
        total = len(uni_names)
        set_status(total=total)

        if total == 0:
            set_status(status="succeeded", finished_at=time.time(), result={"message": "No universities found."})
            return

        subtasks_info = []
        for uni in uni_names:
            sub_id = str(uuid4())
            TASKS[sub_id] = {
                "status": "queued",
                "queued_at": time.time(),
                "type": "calculate_skillnames",
                "scope": "single_university",
                "university": uni,
                "lesson_filter": lesson_name,
                "match": match
            }
            subtasks_info.append({"task_id": sub_id, "university": uni})

        set_status(subtasks=subtasks_info)

        processed = 0
        succeeded, failed = [], []
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {
                ex.submit(
                    _calc_skillnames_task,
                    st["task_id"],
                    st["university"],
                    lesson_name,
                    match,
                    min_skill_level,
                    max_skill_level
                ): st
                for st in subtasks_info
            }
            for fut in as_completed(futs):
                st = futs[fut]
                processed += 1
                st_obj = TASKS.get(st["task_id"], {})
                if st_obj.get("status") == "succeeded":
                    succeeded.append({"task_id": st["task_id"], "university": st["university"]})
                elif st_obj.get("status") == "failed":
                    failed.append({
                        "task_id": st["task_id"],
                        "university": st["university"],
                        "error": st_obj.get("error")
                    })
                set_status(processed=processed)

        set_status(
            status="succeeded",
            finished_at=time.time(),
            result={
                "total_universities": total,
                "succeeded": succeeded,
                "failed": failed
            }
        )
        logger.info("[%s] Completed ALL universities in parallel. ok=%d failed=%d",
                    master_task_id, len(succeeded), len(failed))

    except mysql.connector.Error as e:
        set_status(status="failed", finished_at=time.time(), error=f"MySQL error: {e}")
        logger.exception("[%s] MySQL error in master parallel task", master_task_id)
    except Exception as e:
        set_status(status="failed", finished_at=time.time(), error=str(e))
        logger.exception("[%s] Unexpected error in master parallel task", master_task_id)
    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception:
            pass




# -----------------------------------------------------------------------------
# Full-course skill extraction from ALL relevant Course text fields
# -----------------------------------------------------------------------------
ALL_COURSE_SKILL_FIELDS = [
    ("lesson_name", "lesson_name"),
    ("degree_titles", "degree_titles"),
    ("description", "description"),
    ("objectives", "objectives"),
    ("learning_outcomes", "learning_outcomes"),
    ("course_content", "course_content"),
    ("assessment", "assessment"),
    ("exam", "exam"),
    ("prerequisites", "prerequisites"),
    ("general_competences", "general_competences"),
    ("educational_material", "educational_material"),
    ("attendance_type", "attendance_type"),
    ("language", "language"),
    ("hours", "hours"),
    ("semester_label", "semester_label"),
    ("ects_list", "ects_list"),
    ("mand_opt_list", "mand_opt_list"),
    ("msc_bsc_list", "msc_bsc_list"),
    ("fee_list", "fee_list"),
    ("professors", "professors"),
    ("extras", "extras"),
]


def _normalize_skill_source_text(value: Any) -> str:
    """Convert DB text/JSON values into extractor-safe plain text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw
    else:
        parsed = value

    out: List[str] = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, (int, float)):
            out.append(str(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)
        else:
            s = str(x).strip()
            if s:
                out.append(s)

    walk(parsed)
    return "\n".join(dict.fromkeys(out))


def _extract_urls_for_course_all_fields(course: dict) -> tuple:
    """
    Extract skill URLs from all relevant Course fields.
    Returns (course_id, title, url_to_categories).
    """
    cid = course["course_id"]
    title = course.get("lesson_name") or f"course_{cid}"
    url_to_categories: Dict[str, Set[str]] = defaultdict(set)

    base = os.getenv("API_SKILL_EXTRACTOR_BASE_URL")
    if not base:
        logger.error("API_SKILL_EXTRACTOR_BASE_URL is missing")
        return cid, title, url_to_categories

    extractor_url = base.rstrip("/") + "/extract-skills"

    for field, category in ALL_COURSE_SKILL_FIELDS:
        text_value = _normalize_skill_source_text(course.get(field))
        if not text_value:
            continue
        

        try:
            detected_lang = _detect_lang(text_value)
        except Exception:
            detected_lang = "unknown"

        if detected_lang != "en":
            try:
                print(
                    f"[TRANSLATE] course_id={cid} field={field} lang={detected_lang} -> en",
                    flush=True
                )
                text_value = _translate_hf(text_value, source_lang=detected_lang, target_lang="en")
            except TypeError:
                text_value = _translate_hf(text_value)
            except Exception as e:
                print(
                    f"[TRANSLATE-WARN] course_id={cid} field={field} translation failed: {e}",
                    flush=True
                )

        

        try:
            resp = _http_session.post(
                extractor_url,
                headers={"Content-Type": "application/json", "accept": "application/json"},
                json=[text_value],
                verify=False,
                timeout=60,
            )

            if not resp.ok:
                logger.error(
                    "Skill extractor failed for course_id=%s field=%s: %s %s",
                    cid,
                    field,
                    resp.status_code,
                    resp.text[:500],
                )
                continue

            data = resp.json()
            urls: Set[str] = set()

            if isinstance(data, list):
                for group in data:
                    if isinstance(group, list):
                        urls.update(su for su in group if isinstance(su, str))
                    elif isinstance(group, str):
                        urls.add(group)

            elif isinstance(data, dict):
                for it in data.get("items", []):
                    ids = it.get("id") or it.get("ids") or []
                    if isinstance(ids, str):
                        urls.add(ids)
                    elif isinstance(ids, list):
                        urls.update(su for su in ids if isinstance(su, str))

            for skill_url in urls:
                url_to_categories[skill_url].add(category)

            logger.info(
                "course_id=%s field=%s extracted_urls=%d",
                cid,
                field,
                len(urls),
            )

        except Exception:
            logger.exception("Skill extraction failed for course_id=%s field=%s", cid, field)

    return cid, title, url_to_categories


def _extract_course_skills_all_fields_task(
    task_id: str,
    university_name: Optional[str] = None,
    lesson_name: Optional[str] = None,
    match: Literal["exact", "like"] = "like",
    workers: int = 8,
    min_skill_level: Optional[int] = None,
    max_skill_level: Optional[int] = None,
):
    task = TASKS.get(task_id, {})

    def set_status(**kw):
        task.update(kw)
        TASKS[task_id] = task

    set_status(
        status="running",
        started_at=time.time(),
        processed=0,
        total=0,
        phase="loading_courses",
        university_filter=university_name,
        lesson_filter=lesson_name,
        match=match,
    )

    if not is_database_connected(DB_CONFIG):
        set_status(status="failed", finished_at=time.time(), error="Database connection failed.")
        return

    token = get_tracker_token()
    if not token:
        set_status(status="failed", finished_at=time.time(), error="Tracker token missing")
        return

    def _build_pred(col: str, mode: str) -> str:
        return f"LOWER({col}) = LOWER(%s)" if mode == "exact" else f"LOWER({col}) LIKE LOWER(%s)"

    def _param(value: str, mode: str) -> str:
        return value if mode == "exact" else f"%{value}%"

    conn = None
    cur = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)

        where = []
        params: List[Any] = []

        if university_name:
            where.append(_build_pred("u.university_name", match))
            params.append(_param(university_name, match))

        if lesson_name:
            where.append(_build_pred("c.lesson_name", match))
            params.append(_param(lesson_name, match))

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        cur.execute(
            f"""
            SELECT
                c.course_id,
                c.university_id,
                u.university_name,
                c.lesson_name,
                c.degree_titles,
                c.description,
                c.objectives,
                c.learning_outcomes,
                c.course_content,
                c.assessment,
                c.exam,
                c.prerequisites,
                c.general_competences,
                c.educational_material,
                c.attendance_type,
                c.language,
                c.hours,
                c.semester_label,
                c.ects_list,
                c.mand_opt_list,
                c.msc_bsc_list,
                c.fee_list,
                c.professors,
                c.extras
            FROM Course c
            JOIN University u ON u.university_id = c.university_id
            {where_sql}
            ORDER BY u.university_name, c.lesson_name
            """,
            tuple(params),
        )
        courses = cur.fetchall() or []
        total = len(courses)
        set_status(total=total)

        if total == 0:
            set_status(
                status="succeeded",
                finished_at=time.time(),
                result={"message": "No courses matched the filters.", "updated_courses": 0},
            )
            return

        max_workers = max(1, min(int(workers or 8), 32))
        extracted_maps: Dict[int, Dict[str, Set[str]]] = {}
        titles: Dict[int, str] = {}
        processed = 0
        total_skill_urls = 0

        set_status(phase="extracting_skill_urls")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_extract_urls_for_course_all_fields, course): course["course_id"]
                for course in courses
            }

            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    course_id, title, url_to_categories = fut.result()
                    titles[course_id] = title
                    extracted_maps[course_id] = url_to_categories
                    total_skill_urls += len(url_to_categories)
                except Exception as e:
                    logger.exception("Course extraction failed for course_id=%s: %s", cid, e)
                    titles[cid] = f"course_{cid}"
                    extracted_maps[cid] = {}

                processed += 1
                if processed % 10 == 0 or processed == total:
                    set_status(
                        processed=processed,
                        phase="extracting_skill_urls",
                        extracted_unique_skill_urls=total_skill_urls,
                    )

        all_urls: Set[str] = set()
        for url_map in extracted_maps.values():
            all_urls.update(url_map.keys())

        set_status(phase="resolving_skill_names", unique_skill_urls=len(all_urls))
        url_meta = _resolve_urls_to_names(all_urls, token)

        if all_urls and not url_meta:
            set_status(
                status="failed",
                finished_at=time.time(),
                error="Skill URLs were extracted, but Tracker returned no skill metadata.",
                unique_skill_urls=len(all_urls),
            )
            return

        set_status(phase="upserting_course_skills")

        results: Dict[str, List[str]] = {}
        processed_db = 0
        updated_courses = 0
        total_inserted_names = 0
        db_workers = max(1, min(int(os.getenv("SKILL_DB_WORKERS", "6")), 32))

        with ThreadPoolExecutor(max_workers=db_workers) as dbex:
            futures = {
                dbex.submit(
                    _db_write_course_skills,
                    cid,
                    extracted_maps.get(cid, {}),
                    url_meta,
                    min_skill_level,
                    max_skill_level,
                ): cid
                for cid in extracted_maps
            }

            for fut in as_completed(futures):
                cid = futures[fut]
                title = titles.get(cid, f"course_{cid}")
                try:
                    names = fut.result() or []
                except Exception as e:
                    logger.exception("DB skill upsert failed for course_id=%s: %s", cid, e)
                    names = []

                if names:
                    updated_courses += 1
                    total_inserted_names += len(names)

                results[title] = names
                processed_db += 1

                if processed_db % 10 == 0 or processed_db == total:
                    set_status(
                        processed=processed_db,
                        phase="upserting_course_skills",
                        updated_courses=updated_courses,
                        linked_skill_names=total_inserted_names,
                    )

        set_status(
            status="succeeded",
            finished_at=time.time(),
            result={
                "processed_courses": total,
                "updated_courses": updated_courses,
                "unique_skill_urls": len(all_urls),
                "resolved_skill_urls": len(url_meta),
                "linked_skill_names": total_inserted_names,
                "categories_used": [cat for _, cat in ALL_COURSE_SKILL_FIELDS],
                "skills_by_course": results,
            },
        )

    except Exception as e:
        set_status(status="failed", finished_at=time.time(), error=str(e))
        logger.exception("[%s] Full-field skill extraction task crashed: %s", task_id, e)
    finally:
        try:
            if cur:
                cur.close()
            if conn:
                conn.close()
        except Exception:
            pass


@app.post(
    "/extract_course_skills_all_fields",
    tags=["Skills"],
    status_code=202,
    summary="Extract skills from all Course text/JSON fields and update Skill/CourseSkill",
)
def extract_course_skills_all_fields(
    background_tasks: BackgroundTasks,
    university_name: Optional[str] = Query(None, description="Optional university filter"),
    lesson_name: Optional[str] = Query(None, description="Optional course filter"),
    match: Literal["exact", "like"] = Query("like"),
    workers: int = Query(8, ge=1, le=32),
    min_skill_level: Optional[int] = Query(None),
    max_skill_level: Optional[int] = Query(None),
):
    """
    Runs skill extraction for all matching courses.

    It categorizes each extracted skill by the Course field it came from
    (description, objectives, learning_outcomes, course_content, assessment,
    prerequisites, degree_titles, etc.) and writes those categories into
    CourseSkill.categories.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    task_id = str(uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "queued_at": time.time(),
        "type": "extract_course_skills_all_fields",
        "university_filter": university_name,
        "lesson_filter": lesson_name,
        "match": match,
        "workers": workers,
        "min_skill_level": min_skill_level,
        "max_skill_level": max_skill_level,
    }

    background_tasks.add_task(
        _extract_course_skills_all_fields_task,
        task_id,
        university_name,
        lesson_name,
        match,
        workers,
        min_skill_level,
        max_skill_level,
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "note": "Poll /curriculum-skills/db/tasks/{task_id} for progress and results.",
        "updates": ["Skill", "CourseSkill"],
        "categories_source": "Course fields",
    }


@app.post("/calculate_skillnames", tags=["Skills"], summary="Extract and upsert skills (background task)",
          status_code=202)
def calculate_skillnames(
        university_name: Optional[str] = "",
        lesson_name: Optional[str] = None,
        match: Literal["exact", "like"] = "like",
        workers: int = 8,
        min_skill_level: Optional[int] = Query(None, description="Minimum skill level (ESCO level)"),
        max_skill_level: Optional[int] = Query(None, description="Maximum skill level (ESCO level)"),
        background_tasks: BackgroundTasks = None
):
    """
    (Optionally) filter by ESCO skill level.  Passing min_skill_level / max_skill_level
    will only insert skills whose level falls in that range.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    if university_name and university_name.strip():
        task_id = str(uuid4())
        TASKS[task_id] = {
            "status": "queued",
            "queued_at": time.time(),
            "type": "calculate_skillnames",
            "scope": "single_university",
            "university": university_name,
            "lesson_filter": lesson_name,
            "match": match,
            "min_skill_level": min_skill_level,
            "max_skill_level": max_skill_level
        }
        background_tasks.add_task(
            _calc_skillnames_task,
            task_id,
            university_name,
            lesson_name,
            match,
            min_skill_level,
            max_skill_level
        )
        return {
            "status": "queued",
            "task_id": task_id,
            "note": "Poll /curriculum-skills/db/tasks/{task_id} for progress and results."
        }

    master_task_id = str(uuid4())
    TASKS[master_task_id] = {
        "status": "queued",
        "queued_at": time.time(),
        "type": "calculate_skillnames_all",
        "scope": "all_universities",
        "lesson_filter": lesson_name,
        "match": match,
        "min_skill_level": min_skill_level,
        "max_skill_level": max_skill_level,
        "subtasks": [],
        "processed": 0,
        "total": 0,
        "workers": workers
    }
    background_tasks.add_task(
        _calc_all_skillnames_task_parallel,
        master_task_id,
        lesson_name,
        match,
        workers,
        min_skill_level,
        max_skill_level
    )
    return {
        "status": "queued",
        "task_id": master_task_id,
        "note": "Poll /curriculum-skills/db/tasks/{task_id} for aggregated progress and results."
    }


def _calc_all_skillnames_task(master_task_id: str, lesson_name: Optional[str], match: Literal["exact", "like"]):
    task = TASKS.get(master_task_id, {})

    def set_status(**kw):
        task.update(kw)
        TASKS[master_task_id] = task

    set_status(status="running", started_at=time.time(), processed=0)

    if not is_database_connected(DB_CONFIG):
        set_status(status="failed", finished_at=time.time(), error="Database connection failed.")
        logger.error("[%s] DB connection failed in master task", master_task_id)
        return

    conn = None
    cur = None
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT university_name FROM University ORDER BY university_name ASC")
        rows = cur.fetchall() or []
        uni_names = [r["university_name"] for r in rows if (r.get("university_name") or "").strip()]
        total = len(uni_names)
        set_status(total=total)

        if total == 0:
            set_status(status="succeeded", finished_at=time.time(), result={"message": "No universities found."})
            return

        subtasks_info = []
        processed = 0

        for uni in uni_names:
            sub_id = str(uuid4())
            TASKS[sub_id] = {
                "status": "queued",
                "queued_at": time.time(),
                "type": "calculate_skillnames",
                "scope": "single_university",
                "university": uni,
                "lesson_filter": lesson_name,
                "match": match
            }
            _calc_skillnames_task(sub_id, uni, lesson_name, match)

            subtasks_info.append({"task_id": sub_id, "university": uni})
            processed += 1
            set_status(processed=processed, subtasks=subtasks_info)

        succeeded = []
        failed = []
        for st in subtasks_info:
            st_obj = TASKS.get(st["task_id"], {})
            if st_obj.get("status") == "succeeded":
                succeeded.append({"task_id": st["task_id"], "university": st["university"]})
            elif st_obj.get("status") == "failed":
                failed.append({
                    "task_id": st["task_id"],
                    "university": st["university"],
                    "error": st_obj.get("error")
                })

        set_status(
            status="succeeded",
            finished_at=time.time(),
            result={
                "total_universities": total,
                "succeeded": succeeded,
                "failed": failed
            }
        )
        logger.info("[%s] Completed ALL universities. ok=%d failed=%d", master_task_id, len(succeeded), len(failed))

    except mysql.connector.Error as e:
        set_status(status="failed", finished_at=time.time(), error=f"MySQL error: {e}")
        logger.exception("[%s] MySQL error in master task", master_task_id)
    except Exception as e:
        set_status(status="failed", finished_at=time.time(), error=str(e))
        logger.exception("[%s] Unexpected error in master task", master_task_id)
    finally:
        try:
            if cur: cur.close()
            if conn: conn.close()
        except Exception:
            pass


def _calc_skillnames_task(
        task_id: str,
        university_name: str,
        lesson_name: Optional[str],
        match: Literal["exact", "like"],
        min_skill_level: Optional[int] = None,
        max_skill_level: Optional[int] = None
):
    task = TASKS.get(task_id, {})

    def set_status(**kw):
        task.update(kw)
        TASKS[task_id] = task

    set_status(
        status="running",
        started_at=time.time(),
        processed=0,
        total=0,
        university=university_name,
        lesson_filter=lesson_name,
        match=match
    )

    if not is_database_connected(DB_CONFIG):
        set_status(status="failed", finished_at=time.time(), error="Database connection failed.")
        return

    COURSE_WORKERS = int(os.getenv("SKILL_COURSE_WORKERS", "32"))
    DB_WORKERS = int(os.getenv("SKILL_DB_WORKERS", "6"))

    def _build_pred(col: str, mode: str) -> str:
        return f"LOWER({col}) = LOWER(%s)" if mode == "exact" else f"LOWER({col}) LIKE LOWER(%s)"

    def _param(val: str, mode: str) -> str:
        return val if mode == "exact" else f"%{val}%"

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            f"SELECT university_id, university_name FROM University WHERE {_build_pred('university_name', match)}",
            (_param(university_name, match),),
        )
        row = cursor.fetchone()
        if not row:
            set_status(status="failed", finished_at=time.time(),
                       error=f"University not found in DB: '{university_name}'")
            return

        university_id = row["university_id"]
        uni_label = row["university_name"]

        where = ["c.university_id = %s"]
        params: List[Any] = [university_id]
        if lesson_name:
            where.append(_build_pred("c.lesson_name", match))
            params.append(_param(lesson_name, match))

        cursor.execute(
            f"""SELECT
                     c.course_id, c.lesson_name, c.description, c.objectives,
                     c.learning_outcomes, c.course_content, c.assessment,
                     c.exam, c.prerequisites, c.general_competences, c.educational_material
                 FROM Course c
                 WHERE {" AND ".join(where)}""",
            tuple(params),
        )
        courses = cursor.fetchall() or []
        total = len(courses)
        set_status(total=total)

        if total == 0:
            set_status(
                status="succeeded",
                finished_at=time.time(),
                result={"university_name": uni_label, "skills": {}},
            )
            return

        token = get_tracker_token()
        if not token:
            set_status(status="failed", finished_at=time.time(), error="Tracker token missing")
            return

        extracted_maps = {}
        titles = {}
        processed = 0
        with ThreadPoolExecutor(max_workers=COURSE_WORKERS) as ex:
            futs = {ex.submit(_extract_urls_for_course, c): c["course_id"] for c in courses}
            for fut in as_completed(futs):
                cid = futs[fut]
                try:
                    course_id, title, url_to_categories = fut.result()
                    titles[course_id] = title
                    extracted_maps[course_id] = url_to_categories
                except Exception:
                    titles[cid] = titles.get(cid) or f"course_{cid}"
                    extracted_maps[cid] = {}
                processed += 1
                if processed % 10 == 0 or processed == total:
                    set_status(processed=processed, phase="extraction")

        all_urls = set()
        for m in extracted_maps.values():
            all_urls.update(m.keys())
        url_meta = _resolve_urls_to_names(all_urls, token)

        results = {}
        processed2 = 0
        with ThreadPoolExecutor(max_workers=DB_WORKERS) as dbex:
            futs = {
                dbex.submit(
                    _db_write_course_skills,
                    cid,
                    extracted_maps[cid],
                    url_meta,
                    min_skill_level,
                    max_skill_level,
                ): cid
                for cid in extracted_maps
            }

            for fut in as_completed(futs):
                cid = futs[fut]
                title = titles.get(cid, f"course_{cid}")
                results.setdefault(title, [])
                try:
                    names = fut.result()
                    results[title] = names
                except Exception:
                    results[title] = []

                processed2 += 1
                if processed2 % 10 == 0 or processed2 == total:
                    set_status(processed=processed2, phase="db_upsert")

        set_status(
            status="succeeded",
            finished_at=time.time(),
            result={"university_name": uni_label, "skills": results},
        )
    finally:
        cursor.close()
        conn.close()


@app.post("/nlp/curriculnlp/debug", tags=["CurricuNLP", "Debug"])
def curriculnlp_labels_debug(req: CurricuNLPTextRequest):
    out = CURRICU_CLIENT.predict(text=req.text, api_name="/predict")
    return {
        "input_sample": req.text[:200],
        "labels_count": len(out) if isinstance(out, list) else 0,
        "labels_preview": out[:5] if isinstance(out, list) else []
    }


@app.post("/debug_pdf", tags=["PDF", "Debug"])
def debug_pdf(request: DebugPDFRequest):
    pdf_path = _file_pdf_path(request.pdf_name)
    file_size = os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0

    try:
        pages = extract_text_from_pdf(pdf_path)
        pymu_text = " ".join(pages) if isinstance(pages, list) else str(pages or "")
        pymu_len = len(pymu_text.strip())
        page_count = len(pages) if isinstance(pages, list) else None
        nonempty_pages = sum(1 for p in (pages or []) if isinstance(p, str) and p.strip()) if isinstance(pages,
                                                                                                         list) else None
    except Exception:
        pymu_text, pymu_len, page_count, nonempty_pages = "", 0, None, None

    pdfminer_text = _extract_text_pdfminer(pdf_path)
    pdfminer_len = len(pdfminer_text.strip())

    ocr_text = ""
    ocr_len = 0
    if request.ocr_if_short and max(pymu_len, pdfminer_len) < 800:
        ocr_text = _extract_text_ocr(pdf_path, max_pages=request.ocr_max_pages)
        ocr_len = len(ocr_text.strip())

    chosen = max([(pymu_len, "pymupdf", pymu_text),
                  (pdfminer_len, "pdfminer", pdfminer_text),
                  (ocr_len, "ocr", ocr_text)], key=lambda x: x[0])
    chosen_len, chosen_strategy, chosen_text = chosen
    cleaned = _clean_pdf_text(chosen_text)
    lang = _detect_lang(cleaned) if cleaned else "en"
    translated = False
    if request.translate and cleaned and lang != "en":
        cleaned = _translate_hf(cleaned)
        translated = True

    chunks = _chunk_text(cleaned, chunk_size=request.chunk_size, overlap=request.overlap, limit=request.max_chars)
    labels = []
    label_counts = {}
    if request.run_ner and cleaned:
        try:
            from gradio_client import Client
            client = Client("marfoli/CurricuNLP")
            outs = []
            for ch in chunks:
                attempt = 0
                while True:
                    try:
                        res = client.predict(text=ch, api_name="/predict")
                        outs.extend(res or [])
                        break
                    except Exception:
                        attempt += 1
                        if attempt > 2:
                            break
                        time.sleep(0.5 * attempt)
                time.sleep(0.15)
            labels = _merge_ner(outs)
            cnt = Counter([x.get("class_or_confidence") for x in labels if x.get("class_or_confidence")])
            label_counts = dict(cnt)
        except Exception:
            labels, label_counts = [], {}

    uni_guess = re.sub(r"[_\W]+", " ", os.path.basename(pdf_path).replace(".pdf", "")).strip()
    meta = _find_uni_by_name(uni_guess)
    university_name = meta.get("name") or uni_guess
    university_country = meta.get("country")
    domain = meta.get("domain")

    env = {
        "BROWSERLESS_WS_present": bool(os.getenv("BROWSERLESS_WS")),
        "HF_API_TOKEN_present": bool(os.getenv("HF_API_TOKEN")),
        "tesseract_present": bool(shutil.which("tesseract")),
        "poppler_present": bool(shutil.which("pdftoppm") or shutil.which("pdfinfo")),
    }

    return {
        "file": os.path.basename(pdf_path),
        "file_size_bytes": file_size,
        "pages_detected": page_count,
        "nonempty_pages": nonempty_pages,
        "university_meta": {"name": university_name, "country": university_country, "domain": domain},
        "text_lengths": {
            "pymupdf": pymu_len,
            "pdfminer": pdfminer_len,
            "ocr": ocr_len,
            "chosen_strategy": chosen_strategy,
            "chosen_len": chosen_len
        },
        "language": {"detected": lang, "translated": translated},
        "chunking": {
            "chunk_size": request.chunk_size,
            "overlap": request.overlap,
            "max_chars": request.max_chars,
            "num_chunks": len(chunks)
        },
        "env": env,
        "sample": cleaned[:request.sample_chars] if cleaned else "",
        "labels": labels,
        "label_counts": label_counts
    }


@app.get("/search_skill", tags=["Queries"])
def search_skill(
        skill: str = Query(..., description="Skill name"),
        university: Optional[str] = Query(None, description="University name (optional, LIKE search)")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    where = ["s.skill_name LIKE %s"]
    params = [f"%{skill}%"]
    if university:
        where.append("u.university_name LIKE %s")
        params.append(f"%{university}%")

    sql = f"""
        SELECT u.university_name, c.lesson_name
        {JOIN_SKILL_ON_COURSE}
        WHERE {" AND ".join(where)}
        ORDER BY u.university_name, c.lesson_name
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        out = defaultdict(list)
        for r in rows:
            out[r["university_name"]].append(r["lesson_name"])

        freq = Counter([r["university_name"] for r in rows])

        return JSONResponse(content=_json_safe({
            "skill_query": skill,
            "universities": out,
            "university_frequency": dict(freq)
        }))
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()

SKILL_AREA_KEYWORDS = {
    "Software Development": [
        "programming", "software", "java", "python", "coding",
        "development", "algorithms", "data structures"
    ],
    "Data Analysis": [
        "data analysis", "statistics", "analytics", "data mining",
        "visualization", "business intelligence"
    ],
    "Artificial Intelligence": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural", "computer vision", "nlp", "natural language processing"
    ],
    "Cloud Computing": [
        "cloud", "docker", "kubernetes", "devops",
        "distributed systems", "microservices"
    ],
    "Cybersecurity": [
        "cybersecurity", "security", "cryptography",
        "network security", "information security"
    ],
}


@app.get("/analytics/course_skill_area_coverage", tags=["Analytics"])
def course_skill_area_coverage(
    university: Optional[str] = Query(None),
    degree: Optional[str] = Query(None),
):
    """
    Computes curriculum coverage per skill area.

    Coverage score = percentage of courses that include at least one skill
    belonging to the selected skill area.
    """

    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    where = []
    params = []

    if university:
        where.append("LOWER(u.university_name) LIKE LOWER(%s)")
        params.append(f"%{university}%")

    if degree:
        where.append("LOWER(CAST(c.degree_titles AS CHAR)) LIKE LOWER(%s)")
        params.append(f"%{degree}%")

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    sql = f"""
        SELECT
            c.course_id,
            c.lesson_name,
            s.skill_name
        FROM Course c
        JOIN University u ON u.university_id = c.university_id
        LEFT JOIN CourseSkill cs ON cs.course_id = c.course_id
        LEFT JOIN Skill s ON s.skill_id = cs.skill_id
        {where_sql}
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        course_skills = defaultdict(set)

        for row in rows:
            course_id = row["course_id"]
            skill_name = row.get("skill_name")
            if skill_name:
                course_skills[course_id].add(skill_name.lower())

        total_courses = len(course_skills)

        if total_courses == 0:
            return {
                "total_courses": 0,
                "coverage": [],
                "message": "No courses with skills found."
            }

        results = []

        for area, keywords in SKILL_AREA_KEYWORDS.items():
            covered_courses = 0

            for skills in course_skills.values():
                matched = any(
                    keyword.lower() in skill
                    for skill in skills
                    for keyword in keywords
                )

                if matched:
                    covered_courses += 1

            coverage_score = round((covered_courses / total_courses) * 100, 2)

            results.append({
                "skill_area": area,
                "covered_courses": covered_courses,
                "total_courses": total_courses,
                "coverage_score": coverage_score
            })

        return {
            "university": university,
            "degree": degree,
            "total_courses": total_courses,
            "coverage": results
        }

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass

@app.post(
    "/db/save_json_dir",
    tags=["Database"],
    summary="Bulk import JSON files from a directory into the database",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Import all .json in ./crawler_json (non-recursive)",
                            "value": {
                                "directory": "crawler_json",
                                "filename_pattern": "*.json",
                                "recursive": False,
                                "normalize_university": True,
                                "dry_run": False
                            }
                        },
                        "recursiveLimit": {
                            "summary": "Recurse with a limit (preview only)",
                            "value": {
                                "directory": "output/json",
                                "filename_pattern": "*.json",
                                "recursive": True,
                                "limit": 25,
                                "normalize_university": True,
                                "dry_run": True
                            }
                        },
                        "parallelImport": {
                            "summary": "High parallel import (process pool + capped DB writers)",
                            "value": {
                                "directory": "crawler_json",
                                "recursive": True,
                                "normalize_university": True,
                                "dry_run": False,
                                "workers": 16,
                                "db_workers": 8,
                                "use_process_pool": True
                            }
                        }
                    }
                }
            }
        }
    },
)
def save_json_dir_to_db(req: SaveJSONDirRequest, background_tasks: BackgroundTasks):
    base_dir = os.path.abspath(os.path.expanduser(req.directory))
    if not os.path.isdir(base_dir):
        raise HTTPException(status_code=404, detail=f"Directory not found: {base_dir}")

    if not req.dry_run and not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    try:
        req_data = req.dict()
    except AttributeError:
        req_data = req.model_dump()

    task_id = str(uuid4())
    TASKS[task_id] = {"status": "queued", "queued_at": time.time(), "type": "save_json_dir"}

    background_tasks.add_task(_save_json_dir_task, task_id, req_data)

    return {
        "status": "queued",
        "task_id": task_id,
        "directory": base_dir,
        "dry_run": bool(req.dry_run),
        "note": "Poll /db/tasks/{task_id} for progress and results."
    }


@app.get("/search_skill_by_URL", tags=["Queries"])
def search_skill_by_url(
        skill_url: str = Query(..., description="Exact or partial Skill URL"),
        university: Optional[str] = Query(None, description="University name (optional, LIKE search)")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    where = ["s.skill_url LIKE %s"]
    params = [f"%{skill_url}%"]
    if university:
        where.append("u.university_name LIKE %s")
        params.append(f"%{university}%")

    sql = f"""
        SELECT u.university_name, c.lesson_name, s.skill_name, s.skill_url
        {JOIN_SKILL_ON_COURSE}
        WHERE {" AND ".join(where)}
        ORDER BY u.university_name, c.lesson_name
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        grouped = defaultdict(lambda: [])
        for r in rows:
            grouped[r["university_name"]].append({
                "course": r["lesson_name"],
                "skill_name": r["skill_name"],
                "skill_url": r["skill_url"]
            })

        return JSONResponse(content=_json_safe({
            "skill_url_query": skill_url,
            "results": grouped
        }))
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()


@app.get("/get_universities_by_skills", tags=["Queries"])
def get_universities_by_skills(
        skills: List[str] = Query(..., description="List of skills (names, partial match)")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    requested_raw = skills[:]
    requested = [s.strip().lower() for s in skills if s and s.strip()]
    requested_set = set(requested)
    if not requested_set:
        raise HTTPException(status_code=400, detail="No valid skills provided.")

    like_clauses = " OR ".join(["s.skill_name LIKE %s" for _ in requested])
    params = [f"%{s}%" for s in requested]

    sql = f"""
        SELECT u.university_name, c.lesson_name, s.skill_name
        {JOIN_SKILL_ON_COURSE}
        WHERE ({like_clauses})
    """

    def map_to_requested(db_skill_name: str) -> Optional[str]:
        if not db_skill_name:
            return None
        d = db_skill_name.lower().strip()
        if d in requested_set:
            return d
        for r in requested_set:
            if r in d or d in r:
                return r
        return None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        uni_to_present: Dict[str, set] = defaultdict(set)
        uni_to_courses: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        for r in rows:
            uni = r["university_name"]
            course = r["lesson_name"]
            db_skill = r["skill_name"]
            mapped = map_to_requested(db_skill)
            if mapped:
                uni_to_present[uni].add(mapped)
                if mapped not in uni_to_courses[uni][course]:
                    uni_to_courses[uni][course].append(mapped)

        full_coverage = {
            uni: courses for uni, courses in uni_to_courses.items()
            if requested_set.issubset(uni_to_present.get(uni, set()))
        }
        if full_coverage:
            return full_coverage

        n_req = len(requested_set)

        def coverage(uni: str) -> float:
            return len(uni_to_present.get(uni, set())) / n_req if n_req else 0.0

        NEAR_THRESHOLD = 0.95
        near_candidates = []
        for uni in uni_to_present.keys():
            cov = coverage(uni)
            if cov >= NEAR_THRESHOLD:
                present = sorted(uni_to_present[uni])
                missing = sorted(requested_set - uni_to_present[uni])
                near_candidates.append({
                    "university": uni,
                    "coverage": round(cov * 100.0, 2),
                    "present_skills": present,
                    "missing_skills": missing,
                    "courses": uni_to_courses[uni]
                })

        near_candidates.sort(key=lambda x: (-x["coverage"], -len(x["present_skills"]), x["university"]))

        if near_candidates:
            return {
                "message": "No university contains all requested skills. Returning closest matches (>=95% coverage).",
                "requested_skills": requested_raw,
                "near_matches": near_candidates
            }

        if uni_to_present:
            fallback = []
            for uni in uni_to_present.keys():
                cov = coverage(uni)
                present = sorted(uni_to_present[uni])
                missing = sorted(requested_set - uni_to_present[uni])
                fallback.append({
                    "university": uni,
                    "coverage": round(cov * 100.0, 2),
                    "present_skills": present,
                    "missing_skills": missing,
                    "courses": uni_to_courses[uni]
                })
            fallback.sort(key=lambda x: (-x["coverage"], -len(x["present_skills"]), x["university"]))
            return {
                "message": "No university meets 95% coverage. Returning best available matches.",
                "requested_skills": requested_raw,
                "best_matches": fallback[:5]
            }

        return {
            "message": "No universities found matching the requested skills (even loosely).",
            "requested_skills": requested_raw,
            "near_matches": []
        }

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


@app.get("/get_top_skills", tags=["Queries"])
def get_top_skills(
        university_name: str = Query(..., description="University name (LIKE match)"),
        top_n: int = Query(20, description="Number of top skills to return")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    sql = f"""
    SELECT s.skill_name
    {JOIN_SKILL_ON_COURSE}
    WHERE u.university_name LIKE %s
      AND s.skill_name IS NOT NULL
      AND s.skill_name != ''
      AND s.skill_name NOT LIKE 'Unknown%%'
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql, (f"%{university_name}%",))
        names = [row[0] for row in cursor.fetchall()]
        counter = Counter(names)
        top = [{"skill": k, "frequency": v} for k, v in counter.most_common(top_n)]
        return {"university_name_query": university_name, "top_skills": top}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()


@app.post(
    "/extract_course_skills_all_fields_direct",
    tags=["Skills"],
    summary="Directly extract skills from all Course fields using local ESCO extractor",
)
def extract_course_skills_all_fields_direct(
    university_name: Optional[str] = Query(None),
    lesson_name: Optional[str] = Query(None),
    match: Literal["exact", "like"] = Query("like"),
    workers: int = Query(8, ge=1, le=32),
    min_skill_level: Optional[int] = Query(None),
    max_skill_level: Optional[int] = Query(None),
):
    """
    Blocking endpoint.
    No background task.
    No task_id.
    Prints progress directly in CMD / Docker logs.
    """

    print("\n==============================")
    print("STARTING DIRECT SKILL EXTRACTION")
    print("==============================")
    print(f"University filter: {university_name}")
    print(f"Lesson filter: {lesson_name}")
    print(f"Match mode: {match}")
    print(f"Workers: {workers}")
    print(f"Extractor base: {os.getenv('API_SKILL_EXTRACTOR_BASE_URL')}")
    print("==============================\n")

    print("[0/5] Checking database connection...", flush=True)

    try:
        test_conn = mysql.connector.connect(
            **DB_CONFIG,
            connection_timeout=5
        )
        test_conn.close()
        print("[OK] Database connection works.", flush=True)
    except Exception as e:
        print(f"[FAILED] Database connection failed: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")

    def _build_pred(col: str, mode: str) -> str:
        return f"LOWER({col}) = LOWER(%s)" if mode == "exact" else f"LOWER({col}) LIKE LOWER(%s)"

    def _param(value: str, mode: str) -> str:
        return value if mode == "exact" else f"%{value}%"

    conn = None
    cur = None

    try:
        print("[1/5] Connecting to database...")
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)

        where = []
        params: List[Any] = []

        if university_name:
            where.append(_build_pred("u.university_name", match))
            params.append(_param(university_name, match))

        if lesson_name:
            where.append(_build_pred("c.lesson_name", match))
            params.append(_param(lesson_name, match))

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        print("[2/5] Loading courses from database...")

        cur.execute(
            f"""
            SELECT
                c.course_id,
                c.university_id,
                u.university_name,
                c.lesson_name,
                c.degree_titles,
                c.description,
                c.objectives,
                c.learning_outcomes,
                c.course_content,
                c.assessment,
                c.exam,
                c.prerequisites,
                c.general_competences,
                c.educational_material,
                c.attendance_type,
                c.language,
                c.hours,
                c.semester_label,
                c.ects_list,
                c.mand_opt_list,
                c.msc_bsc_list,
                c.fee_list,
                c.professors,
                c.extras
            FROM Course c
            JOIN University u ON u.university_id = c.university_id
            {where_sql}
            ORDER BY u.university_name, c.lesson_name
            """,
            tuple(params),
        )

        courses = cur.fetchall() or []
        total = len(courses)

        print(f"[OK] Loaded {total} courses.")

        if not courses:
            return {
                "status": "completed",
                "message": "No courses found.",
                "processed_courses": 0,
            }

        print("[3/5] Extracting ESCO skill URLs from course fields...")

        extracted_maps: Dict[int, Dict[str, Set[str]]] = {}
        titles: Dict[int, str] = {}
        all_urls: Set[str] = set()

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(_extract_urls_for_course_all_fields, course): course
                for course in courses
            }

            processed = 0

            for fut in as_completed(futures):
                cid, title, url_to_categories = fut.result()

                titles[cid] = title
                extracted_maps[cid] = url_to_categories

                for url in url_to_categories.keys():
                    all_urls.add(url)

                processed += 1

                print(
                    f"[EXTRACT] {processed}/{total} | "
                    f"course_id={cid} | {title} | "
                    f"skills={len(url_to_categories)}"
                )

        print(f"[OK] Unique extracted ESCO skill URLs: {len(all_urls)}")

        print("[4/5] Resolving ESCO URLs using local ESCO CSV...", flush=True)

        url_meta = _resolve_urls_to_names_local_esco(all_urls)

        missing = [
            url for url, meta in url_meta.items()
            if meta.get("label") == url
        ]

        print(
            f"[OK] Resolved {len(url_meta) - len(missing)}/{len(all_urls)} "
            f"ESCO skill names locally. Missing={len(missing)}",
            flush=True
        )

        print(f"[OK] Prepared {len(url_meta)} local ESCO skills.", flush=True)

        if all_urls and not url_meta:
            raise HTTPException(
                status_code=502,
                detail="Skill URLs were extracted, but Tracker returned no metadata."
            )

        print("[5/5] Writing skills into Skill and CourseSkill tables...")

        results: Dict[str, List[str]] = {}
        updated_courses = 0
        total_inserted_names = 0

        db_workers = max(1, min(int(os.getenv("SKILL_DB_WORKERS", "6")), 32))

        with ThreadPoolExecutor(max_workers=db_workers) as dbex:
            futures = {
                dbex.submit(
                    _db_write_course_skills,
                    cid,
                    extracted_maps.get(cid, {}),
                    url_meta,
                    min_skill_level,
                    max_skill_level,
                ): cid
                for cid in extracted_maps
            }

            processed_db = 0

            for fut in as_completed(futures):
                cid = futures[fut]
                title = titles.get(cid, f"course_{cid}")

                try:
                    names = fut.result() or []
                except Exception as e:
                    print(f"[ERROR] DB upsert failed for course_id={cid}: {e}")
                    logger.exception("DB skill upsert failed for course_id=%s", cid)
                    names = []

                if names:
                    updated_courses += 1
                    total_inserted_names += len(names)

                results[title] = names
                processed_db += 1

                print(
                    f"[DB] {processed_db}/{total} | "
                    f"{title} | inserted/linked={len(names)}"
                )

        print("\n==============================")
        print("DIRECT SKILL EXTRACTION FINISHED")
        print("==============================")
        print(f"Processed courses: {total}")
        print(f"Updated courses: {updated_courses}")
        print(f"Unique skill URLs: {len(all_urls)}")
        print(f"Resolved skill URLs: {len(url_meta)}")
        print(f"Linked skill names: {total_inserted_names}")
        print("==============================\n")

        return {
            "status": "completed",
            "processed_courses": total,
            "updated_courses": updated_courses,
            "unique_skill_urls": len(all_urls),
            "resolved_skill_urls": len(url_meta),
            "linked_skill_names": total_inserted_names,
            "categories_used": [cat for _, cat in ALL_COURSE_SKILL_FIELDS],
            "skills_by_course": results,
        }

    except Exception as e:
        print(f"[FAILED] Direct skill extraction crashed: {e}")
        logger.exception("Direct full-field skill extraction crashed")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        try:
            if cur:
                cur.close()
            if conn:
                conn.close()
        except Exception:
            pass

@app.get("/get_top_skills_all", tags=["Queries"])
def get_top_skills_all(
        top_n: int = Query(20, description="Number of top skills to return")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    sql = f"""
        SELECT s.skill_name, u.university_name
        {JOIN_SKILL_ON_COURSE}
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()

        counter = Counter([r["skill_name"] for r in rows if r["skill_name"]])
        uni_map = defaultdict(set)
        for r in rows:
            if r["skill_name"]:
                uni_map[r["skill_name"].lower()].add(r["university_name"])

        top = [{
            "skill": k,
            "frequency": v,
            "universities": list(uni_map[k.lower()])
        } for k, v in counter.most_common(top_n)]

        return {"top_skills": top}
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()


@app.get("/course_skills_matrix", tags=["Bilateral"], summary="List of courses with Level-4 skill names")
def course_skills_matrix(
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)")
):
    """
    Returns a paginated list of courses with **Level-4 skill names** only
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    sql = """
        SELECT c.lesson_name, s.skill_name
        FROM Course c
        JOIN CourseSkill cs ON c.course_id = cs.course_id
        JOIN Skill s ON cs.skill_id = s.skill_id
        WHERE s.esco_level IN ('4', 4)
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        results = defaultdict(list)
        for r in rows:
            results[r["lesson_name"]].append(r["skill_name"])
        results_list = [{"course": k, "skills": v} for k, v in results.items()]

        total = len(results_list)
        start = (page - 1) * per_page
        return results_list[start:start + per_page]
    finally:
        cursor.close()
        conn.close()


@app.get("/course_skill_urls_matrix", tags=["Bilateral"], summary="List of courses with Level-4 skill URLs")
def course_skill_urls_matrix(
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)")
):
    """
    Returns a paginated list of courses with **Level-4 ESCO skill URLs**
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    sql = """
        SELECT c.lesson_name, s.skill_url
        FROM Course c
        JOIN CourseSkill cs ON c.course_id = cs.course_id
        JOIN Skill s ON cs.skill_id = s.skill_id
        WHERE s.esco_level IN ('4', 4)
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
        results = defaultdict(list)
        for r in rows:
            results[r["lesson_name"]].append(r["skill_url"])
        results_list = [{"course": k, "skills": v} for k, v in results.items()]

        total = len(results_list)
        start = (page - 1) * per_page
        return results_list[start:start + per_page]
    finally:
        cursor.close()
        conn.close()


@app.post(
    "/db/save_labels",
    tags=["Database"],
    summary="Save label arrays (CurricuNLP output) as Course rows in the database",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic": {
                            "summary": "Minimal example",
                            "value": {
                                "university_name": "ETHZ - ETH Zurich",
                                "country": "Switzerland",
                                "courses": [{
                                    "lesson_name": "Accountancy",
                                    "website": "https://ethz.ch/en/studies/example-course.html",
                                    "labels": [
                                        {"class_or_confidence": "lesson_name", "token": "Accountancy"},
                                        {"class_or_confidence": "ects", "token": "6"},
                                        {"class_or_confidence": "language", "token": "English"},
                                        {"class_or_confidence": "professor", "token": "Dr. Jane Doe"}
                                    ]
                                }]
                            }
                        }
                    }
                }
            }
        }
    },
)
def save_labels_to_db(req: SaveLabelsRequest, background_tasks: BackgroundTasks):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    uni = (req.university_name or "").strip()
    if not uni:
        raise HTTPException(status_code=400, detail="university_name is required.")

    payload = {"university_name": uni, "country": (req.country or "Unknown").strip(), "courses": []}
    for c in req.courses or []:
        course = _labels_to_course(c.lesson_name, c.website, c.labels)
        if not course.get("lesson_name"):
            course["lesson_name"] = (c.website or "Untitled Course")[:255]
        payload["courses"].append(course)

    task_id = str(uuid4())
    TASKS[task_id] = {"status": "queued", "queued_at": time.time()}
    background_tasks.add_task(_save_payload_task, task_id, payload)
    return {"status": "queued", "task_id": task_id, "queued_courses": len(payload["courses"])}


@app.post(
    "/db/save_json",
    tags=["Database"],
    summary="Save a JSON payload directly into the database",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "withNormalization": {
                            "summary": "Normalize university fields from a guess",
                            "value": {
                                "normalize_university": True,
                                "payload": {
                                    "university": "ETH Zurich",
                                    "courses": [
                                        {
                                            "lesson_name": "Signals and Systems",
                                            "website": "https://ethz.ch/en/studies/signals-systems.html",
                                            "ects": 6,
                                            "language": "English"
                                        }
                                    ]
                                }
                            }
                        },
                        "direct": {
                            "summary": "Direct payload (already normalized)",
                            "value": {
                                "normalize_university": False,
                                "payload": {
                                    "university_name": "ETHZ - ETH Zurich",
                                    "country": "Switzerland",
                                    "courses": [
                                        {
                                            "lesson_name": "Algorithms",
                                            "website": "https://ethz.ch/en/studies/algorithms.html",
                                            "ects": 8
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    },
)
def save_json_to_db(req: SaveJSONRequest, background_tasks: BackgroundTasks):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    payload = dict(req.payload or {})
    if req.normalize_university:
        uni_guess = (
                payload.get("university_name")
                or payload.get("university")
                or (payload.get("university_meta") or {}).get("name")
        )
        if uni_guess:
            meta = _find_uni_by_name(uni_guess)
            payload["university_name"] = payload.get("university_name") or meta.get("name") or uni_guess
            payload["country"] = payload.get("country") or payload.get("university_country") or meta.get(
                "country") or "Unknown"

    if not payload.get("university_name"):
        raise HTTPException(status_code=400,
                            detail="payload.university_name is required (or enable normalize_university).")

    task_id = str(uuid4())
    TASKS[task_id] = {"status": "queued", "queued_at": time.time()}
    background_tasks.add_task(_save_payload_task, task_id, payload)
    return {
        "status": "queued",
        "task_id": task_id,
        "queued_courses": len((payload or {}).get("courses", []))
    }


@app.post("/clean-degrees", response_model=CleanResponse, tags=["Cleaning"])
def clean_degrees(req: CleanRequest):
    folder = os.path.abspath(req.folder)
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    backend = req.backend
    model = req.model or ("gpt-4o-mini" if backend == "openai" else "llama3.1")

    if not req.inplace:
        outdir = req.outdir or os.path.join(folder, "_cleaned")
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = None

    files = iter_json_files(folder)
    if not files:
        return CleanResponse(
            totals=CleanTotals(files=0, titles_before=0, titles_after=0, removed=0),
            summaries=[],
            output_dir=outdir,
        )

    totals = {"files": 0, "titles_before": 0, "titles_after": 0, "removed": 0}
    summaries: List[FileSummary] = []

    # For Ollama, do not blindly use os.cpu_count(); too many concurrent LLM calls
    # can cause GPU memory thrashing and look like a stall.
    default_workers = int(os.getenv("OLLAMA_NUM_PARALLEL", "4")) if backend == "ollama" else PDF_CPU_WORKERS
    max_workers = _bounded_workers(req.workers, default_workers, cap=16)

    def _clean_one(in_path: str) -> Dict[str, Any]:
        out_path = in_path if req.inplace else os.path.join(outdir, os.path.basename(in_path))
        return clean_file(in_path, out_path, backend=backend, model=model, dry_run=req.dry_run)

    if max_workers == 1 or len(files) == 1:
        iterator = [_clean_one(in_path) for in_path in files]
    else:
        iterator = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_clean_one, in_path) for in_path in files]
            for fut in as_completed(futures):
                iterator.append(fut.result())

    for summary in iterator:
        totals["files"] += 1
        totals["titles_before"] += summary["original_count"]
        totals["titles_after"] += summary["kept_count"]
        totals["removed"] += summary["removed_count"]
        summaries.append(FileSummary(**summary))

    return CleanResponse(
        totals=CleanTotals(**totals),
        summaries=summaries,
        output_dir=outdir,
    )


@app.get("/descriptive/location", response_model=List[CountryUniversities], tags=["Descriptive"],
         summary="Universities per Country from DB")
def get_university_counts_by_country_db():
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


@app.post("/calculate_occupations", tags=["Skills"], status_code=202)
def calculate_occupations(
        university_name: Optional[str] = "",
        lesson_name: Optional[str] = None,
        background_tasks: BackgroundTasks = None
):
    """
    Resolve occupations for all skills in the DB and store into Occupation + SkillOccupation.
    - No params > ALL universities
    - university_name (+ optional lesson_name) > specific scope.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    task_id = str(uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "queued_at": time.time(),
        "type": "calculate_occupations",
        "scope": "all" if not university_name else "single",
        "university": university_name,
        "lesson_filter": lesson_name
    }

    background_tasks.add_task(
        _calc_occupations_task,
        task_id,
        university_name if university_name else None,
        lesson_name
    )

    return {
        "status": "queued",
        "task_id": task_id,
        "note": "Poll /curriculum-skills/db/tasks/{task_id} for progress."
    }


def _calc_occupations_task(task_id: str,
                           university_name: Optional[str],
                           lesson_name: Optional[str]):
    task = TASKS.get(task_id, {})

    def set_status(**kw):
        task.update(kw)
        TASKS[task_id] = task

    set_status(status="running", started_at=time.time(), processed=0)

    if not is_database_connected(DB_CONFIG):
        set_status(status="failed", finished_at=time.time(),
                   error="Database connection failed.")
        return

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)

    skill_rows = []
    try:
        if university_name:
            where = ["u.university_name LIKE %s"]
            params = [f"%{university_name}%"]
            if lesson_name:
                where.append("c.lesson_name LIKE %s")
                params.append(f"%{lesson_name}%")

            sql = f"""
                SELECT DISTINCT s.skill_id, s.skill_name
                {JOIN_SKILL_ON_COURSE}
                WHERE {" AND ".join(where)}
            """
            cur.execute(sql, tuple(params))
        else:
            cur.execute("SELECT skill_id, skill_name FROM Skill")

        skill_rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    total = len(skill_rows)
    set_status(total=total)
    if not total:
        set_status(status="succeeded", finished_at=time.time(),
                   result={"message": "No skills found."})
        return

    token = get_tracker_token()

    def save_occ_to_db(skill_id: int, occ):
        occ_id = occ.get("id")
        label = occ.get("label")
        parent = occ.get("occupation_group")
        sector = occ.get("top_level_sector")

        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO Occupation (occupation_id, label, parent_label, top_sector)
                VALUES (%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE label=%s, parent_label=%s, top_sector=%s
            """, (occ_id, label, parent, sector, label, parent, sector))

            cursor.execute("""
                INSERT IGNORE INTO SkillOccupation (skill_id, occupation_id)
                VALUES (%s,%s)
            """, (skill_id, occ_id))
            conn.commit()
        finally:
            cursor.close();
            conn.close()

    processed = 0
    for row in skill_rows:
        skill_id = row["skill_id"]
        skill_nm = row["skill_name"]

        try:
            payload = [("keywords", skill_nm),
                       ("keywords_logic", "or"),
                       ("children_logic", "or"),
                       ("ancestors_logic", "or")]
            headers = {
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {token}"
            }
            resp = requests.post(
                f"{os.environ['API_TRACKER_BASE_URL'].rstrip('/')}/occupations?page=1",
                headers=headers,
                data=payload,
                verify=False
            )
            data = resp.json().get("items", []) if resp.ok else []
            for occ in data:
                save_occ_to_db(skill_id, occ)

        except Exception as ex:
            pass

        processed += 1
        if processed % 10 == 0 or processed == total:
            set_status(processed=processed)

    set_status(status="succeeded", finished_at=time.time(),
               result={"processed_skills": total})


from typing import List, Optional, Literal


@app.get("/theme_search", tags=["Queries"])
def theme_search(
        theme: Optional[str] = Query(None, description="Single keyword (comma-separated allowed)"),
        themes: Optional[List[str]] = Query(None,
                                            description="Repeatable multi-keyword param, e.g. ?themes=ai&themes=robotics"),
        logic: Literal["any", "all"] = Query("any", description="Keyword combination logic"),
        threshold: int = Query(70, ge=0, le=100, description="Fuzzy matching threshold (0-100)"),
        include_skills: bool = Query(True, description="Include skills for each matched course"),
        skills_limit: int = Query(0, ge=0, description="Max skills per course (0 = unlimited)"),
        page: int = Query(1, ge=1, description="Results page (1-based)"),
        per_page: int = Query(25, ge=1, le=100, description="Results per page (max 100)")
):
    """
    Fuzzy semantic search over:
      - course title
      - degree titles
      - skill names
      - occupation labels / parent_label / top_sector

    Multiple keywords:
      - ?themes=ai&themes=robotics  (repeatable)
      - ?theme=ai, robotics         (comma-separated)

    Pagination:
      - page (>=1), per_page (1..100)
    """
    keywords: List[str] = []
    if themes:
        keywords.extend([t for t in themes if t and t.strip()])
    if theme:
        keywords.extend([p.strip() for p in theme.split(",") if p.strip()])
    keywords = sorted(set([k.strip().lower() for k in keywords if k and k.strip()]))
    if not keywords:
        raise HTTPException(status_code=400, detail="Provide at least one keyword via ?theme=... or ?themes=...")

    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    sql = """
        SELECT
            u.university_name,
            c.lesson_name,
            c.degree_titles,
            s.skill_id,
            s.skill_name,
            o.label AS occupation_label,
            o.parent_label,
            o.top_sector
        FROM CourseSkill cs
        JOIN Skill s                 ON cs.skill_id      = s.skill_id
        LEFT JOIN SkillOccupation so ON s.skill_id       = so.skill_id
        LEFT JOIN Occupation o       ON so.occupation_id = o.occupation_id
        JOIN Course c                ON cs.course_id     = c.course_id
        JOIN University u            ON c.university_id  = u.university_id
        WHERE s.skill_name IS NOT NULL AND s.skill_name <> ''
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        try:
            cursor.close();
            conn.close()
        except Exception:
            pass

    from collections import defaultdict
    course_map = defaultdict(lambda: {
        "university": "",
        "degree_titles": set(),
        "skills": set(),
        "occ_labels": set(),
        "occ_parents": set(),
        "occ_sectors": set()
    })

    for r in rows:
        uni = r["university_name"]
        title = r["lesson_name"]
        if not title:
            continue
        key = (uni, title)
        cm = course_map[key]
        cm["university"] = uni

        if r.get("degree_titles"):
            try:
                degs = json.loads(r["degree_titles"])
                if isinstance(degs, list):
                    for d in degs:
                        if isinstance(d, str) and d.strip():
                            cm["degree_titles"].add(d.strip())
            except Exception:
                pass

        if r.get("skill_name"):
            cm["skills"].add(r["skill_name"])
        if r.get("occupation_label"):
            cm["occ_labels"].add(r["occupation_label"])
        if r.get("parent_label"):
            cm["occ_parents"].add(r["parent_label"])
        if r.get("top_sector"):
            cm["occ_sectors"].add(r["top_sector"])

    def build_blob(meta: dict, title: str) -> str:
        items = [title]
        items += list(meta["degree_titles"])
        items += list(meta["skills"])
        items += list(meta["occ_labels"])
        items += list(meta["occ_parents"])
        items += list(meta["occ_sectors"])
        return " ".join(items).lower()

    matches = []
    for (uni, title), meta in course_map.items():
        blob = build_blob(meta, title)
        per_kw_scores = [fuzz.partial_ratio(k, blob) for k in keywords]  # 0..100

        if logic == "all":
            keep = (min(per_kw_scores) >= threshold) if per_kw_scores else False
            agg_score = min(per_kw_scores) if per_kw_scores else 0
        else:  # 'any'
            keep = (max(per_kw_scores) >= threshold) if per_kw_scores else False
            agg_score = max(per_kw_scores) if per_kw_scores else 0

        if not keep:
            continue

        skills_list = sorted(meta["skills"])
        if skills_limit and skills_limit > 0:
            skills_list = skills_list[:skills_limit]

        entry = {
            "university": uni,
            "course": title,
            "score": agg_score,
            "per_keyword_scores": dict(zip(keywords, per_kw_scores))
        }
        if include_skills:
            entry["skills"] = skills_list

        matches.append(entry)

    matches.sort(key=lambda x: (-x["score"], x["university"], x["course"]))

    total_results = len(matches)
    total_pages = (total_results + per_page - 1) // per_page if per_page else 0
    start = (page - 1) * per_page
    end = start + per_page
    paged = matches[start:end]

    grouped = defaultdict(list)
    unique_skills_page = set()
    for m in paged:
        if include_skills:
            grouped[m["university"]].append({"course": m["course"], "skills": m.get("skills", [])})
            for s in (m.get("skills") or []):
                unique_skills_page.add(s)
        else:
            grouped[m["university"]].append({"course": m["course"]})

    return {
        "keywords": keywords,
        "logic": logic,
        "threshold": threshold,
        "page": page,
        "per_page": per_page,
        "total_results": total_results,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
        "matches": paged,
        "grouped_by_university": grouped,
        "unique_skills": sorted(unique_skills_page) if include_skills else []
    }


@app.get("/exploratory/skills_location", response_model=List[SkillsByCountry], tags=["Exploratory"],
         summary="Skills per Country from DB")
def get_skills_per_location_db():
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    query = f"""
    SELECT u.country, s.skill_name, COUNT(*) AS frequency
    {JOIN_SKILL_ON_COURSE}
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
        for r in rows:
            grouped[r["country"]].append(SkillPerCountry(skill=r["skill_name"], frequency=r["frequency"]))
        return [{"country": ct, "skills": skills} for ct, skills in grouped.items()]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()

def _append_continuation_to_course(course: Dict[str, Any], text: str):
    text = _clean_pdf_text(_ensure_text(text)).strip()
    if not text:
        return course

    target_field = "course_content"

    if course.get(target_field):
        course[target_field] = f"{course[target_field]}\n\n{text}"
    elif course.get("description"):
        course["description"] = f"{course['description']}\n\n{text}"
    else:
        course[target_field] = text

    course.setdefault("extras", {})
    course["extras"]["has_continuation_pages"] = True

    return course


@app.get("/trend/location", response_model=List[CountryTrend], tags=["Trend"],
         summary="University Join Trend per Country from DB")
def get_university_trend_per_location_db():
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
        for r in rows:
            grouped[r["country"]].append(MonthlyTrend(month=r["month"], count=r["count"]))
        return [{"country": c, "monthly_counts": counts} for c, counts in grouped.items()]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.get(
    "/university/all",
    tags=["Queries"],
    summary="Full snapshot for a university (metadata, courses, skills)",
)
def get_university_full(
        university_name: str = Query(..., description="University name (LIKE match)"),
        page: int = Query(1, ge=1, description="Results page (1-based)"),
        per_page: int = Query(25, ge=1, le=100, description="Courses per page (max 100)"),
):
    """
    Returns a consolidated view for a university:
      - university metadata
      - paginated courses
      - per-course skills
      - aggregated top skills
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    uni_sql = """
        SELECT university_id, university_name, country, created_at
        FROM University
        WHERE university_name LIKE %s
        ORDER BY university_name ASC
        LIMIT 1
    """

    count_sql = """
        SELECT COUNT(*) AS n
        FROM Course c
        JOIN University u ON c.university_id = u.university_id
        WHERE u.university_name LIKE %s
    """

    courses_sql = """
        SELECT
            c.course_id,
            c.lesson_name,
            c.description,
            c.objectives,
            c.learning_outcomes,
            c.course_content,
            c.assessment,
            c.exam,
            c.general_competences,
            c.educational_material
        FROM Course c
        JOIN University u ON c.university_id = u.university_id
        WHERE u.university_name LIKE %s
        ORDER BY c.lesson_name ASC, c.course_id ASC
        LIMIT %s OFFSET %s
    """

    skills_sql = """
        SELECT
            c.course_id,
            c.lesson_name,
            s.skill_name,
            s.skill_url,
            s.esco_level
        FROM Course c
        JOIN University u  ON c.university_id = u.university_id
        JOIN CourseSkill cs ON c.course_id = cs.course_id
        JOIN Skill s        ON cs.skill_id = s.skill_id
        WHERE u.university_name LIKE %s
          AND c.course_id IN ({placeholders})
          AND s.skill_name IS NOT NULL
          AND s.skill_name <> ''
    """

    top_sql = f"""
        SELECT s.skill_name
        {JOIN_SKILL_ON_COURSE}
        WHERE u.university_name LIKE %s
          AND s.skill_name IS NOT NULL
          AND s.skill_name <> ''
    """

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor(dictionary=True)

        cur.execute(uni_sql, (f"%{university_name}%",))
        uni = cur.fetchone()
        if not uni:
            raise HTTPException(status_code=404, detail=f"University not found: '{university_name}'")

        cur.execute(count_sql, (f"%{university_name}%",))
        total_courses = int((cur.fetchone() or {}).get("n") or 0)
        total_pages = (total_courses + per_page - 1) // per_page if total_courses else 0
        if total_pages and page > total_pages:
            page = total_pages
        offset = (page - 1) * per_page

        cur.execute(courses_sql, (f"%{university_name}%", per_page, offset))
        courses = cur.fetchall() or []
        course_ids = [c["course_id"] for c in courses]

        skills_by_course = {cid: [] for cid in course_ids}
        unique_skills = set()
        total_skill_links = 0
        if course_ids:
            ph = ",".join(["%s"] * len(course_ids))
            cur.execute(skills_sql.format(placeholders=ph), (f"%{university_name}%", *course_ids))
            for r in cur.fetchall() or []:
                cid = r["course_id"]
                nm = r.get("skill_name")
                if nm:
                    unique_skills.add(nm)
                    total_skill_links += 1
                skills_by_course.setdefault(cid, []).append({
                    "skill_name": r.get("skill_name"),
                    "skill_url": r.get("skill_url"),
                    "esco_level": (str(r.get("esco_level")).strip() if r.get("esco_level") is not None else None),
                })

        for c in courses:
            c["skills"] = skills_by_course.get(c["course_id"], [])

        cur.execute(top_sql, (f"%{university_name}%",))
        names = [row["skill_name"] for row in (cur.fetchall() or []) if row.get("skill_name")]
        top_counter = Counter(names)
        top_skills = [{"skill": k, "frequency": v} for k, v in top_counter.most_common(50)]

        return {
            "university": {
                "university_id": uni["university_id"],
                "university_name": uni["university_name"],
                "country": uni.get("country"),
                "created_at": uni.get("created_at"),
            },
            "counts": {
                "total_courses": total_courses,
                "unique_skills": len(unique_skills),
                "total_skill_links": total_skill_links,
            },
            "top_skills": top_skills,
            "page": page,
            "per_page": per_page,
            "total_courses": total_courses,
            "total_pages": total_pages,
            "courses": courses,
        }

    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


@app.get("/cluster/universities/{k}", response_model=List[ClusterResult], tags=["Clustering"],
         summary="University Clustering by Skills from DB")
def cluster_universities_db(k: int):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    query = f"""
    SELECT u.university_name, GROUP_CONCAT(DISTINCT s.skill_name SEPARATOR ' ') AS skills
    {JOIN_SKILL_ON_COURSE}
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
        universities = [r["university_name"] for r in rows]
        documents = [r["skills"] for r in rows]
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
        return [{"cluster": cl, "universities": unis} for cl, unis in grouped.items()]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.get("/descriptive/skills_frequency", response_model=List[SkillFrequency], tags=["Descriptive"],
         summary="Skill frequency across courses from DB")
def get_skill_frequencies():
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")
    query = """
        SELECT s.skill_name, COUNT(*) AS frequency
        FROM Skill s
        JOIN CourseSkill cs ON s.skill_id = cs.skill_id
        WHERE s.skill_name IS NOT NULL AND s.skill_name != ''
        GROUP BY s.skill_name
        ORDER BY frequency DESC
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        return [{"skill": r["skill_name"], "frequency": r["frequency"]} for r in rows]
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()


@app.get("/bilateral/biodiversity_analysis", tags=["Bilateral"],
         summary="Degree/Department biodiversity of Level-4 skills")
def biodiversity_analysis(
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(10, ge=1, le=100, description="Results per page (max 100)")
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    sql = f"""
    SELECT 
        u.university_name, u.country,
        c.course_id, c.lesson_name, c.description,
        c.msc_bsc_list, c.degree_titles, c.extras,
        s.skill_url, s.skill_name, s.esco_level
    {JOIN_SKILL_ON_COURSE}
    WHERE s.esco_level IN ('4', 4)
    """
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    def infer_department(extras_raw: Optional[str], lesson_name: str, degree_titles: list) -> Optional[str]:
        try:
            ex = json.loads(extras_raw) if extras_raw else {}
            if isinstance(ex, dict):
                for key in ["department", "department_name", "school", "faculty", "institute", "unit", "college"]:
                    val = ex.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        except Exception:
            pass
        tokens = ("department", "school", "faculty", "institute", "college")
        for t in degree_titles or []:
            if isinstance(t, str) and any(tok in t.lower() for tok in tokens):
                return t.strip()
        return None

    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        if not rows:
            return {"page": page, "per_page": per_page, "total_results": 0, "total_pages": 0, "results": []}

        combo_map: Dict[tuple, Set[tuple]] = defaultdict(set)
        skill_url_map: Dict[str, str] = {}

        for r in rows:
            lesson_name = r.get("lesson_name")
            if not lesson_name or not lesson_name.strip():
                continue

            uni = r["university_name"]
            country = (r.get("country") or "").strip()
            if not country:
                meta = _find_uni_by_name(uni)
                country = meta.get("country") or ""

            try:
                msc_list = json.loads(r.get("msc_bsc_list") or "[]")
                if not isinstance(msc_list, list): msc_list = []
            except Exception:
                msc_list = []
            try:
                deg_titles = json.loads(r.get("degree_titles") or "[]")
                if not isinstance(deg_titles, list): deg_titles = []
            except Exception:
                deg_titles = []

            msc_list = [t for t in msc_list if isinstance(t, str) and t.strip()] or ["Other"]
            deg_titles_filtered = [t for t in deg_titles if isinstance(t, str) and t.strip()]

            dept = infer_department(r.get("extras"), lesson_name, deg_titles_filtered) or "Unknown"

            deg_titles_for_loop = deg_titles_filtered or ["Unknown"]

            su = r.get("skill_url")
            sn = r.get("skill_name")
            if su:
                skill_url_map[su] = sn or ""
                for deg_type in msc_list:
                    for deg_title in deg_titles_for_loop:
                        key = (country or "Unknown", uni, dept, deg_title, deg_type, lesson_name)
                        combo_map[key].add((su, sn))

        results = []
        seen = set()

        for (country, uni, dept, deg_title, deg_type, lesson_name), pairs in combo_map.items():
            if dept == "Unknown" and deg_title == "Unknown":
                continue

            names = sorted({skill_url_map[u] for (u, _) in pairs if skill_url_map.get(u)})
            if not names:
                continue

            signature = (lesson_name.strip().lower(), tuple(names))
            if signature in seen:
                continue
            seen.add(signature)

            results.append({
                "country": country or "Unknown",
                "university": uni,
                "department": dept,
                "course_title": lesson_name,
                "degree": {"type": deg_type, "title": deg_title},
                "skills": names
            })

        total = len(results)
        start = (page - 1) * per_page
        end = start + per_page
        return {
            "page": page,
            "per_page": per_page,
            "total_results": total,
            "total_pages": ceil(total / per_page) if per_page else 0,
            "results": results[start:end]
        }
    finally:
        cursor.close()
        conn.close()


@app.get("/bilateral/labor_market_export", response_model=ExportResponse, tags=["Bilateral"],
         summary="List of all degrees with their Level 4 skills")
def labor_export_from_database(
        university_name: str = Query(None, description="Optional university name to search for (LIKE)"),
        page: int = Query(1, ge=1, description="Page number (starts from 1)"),
        limit: int = Query(100, ge=1, le=1000, description="Number of items per page")
):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)

        query_filter = ""
        params: List[str] = []
        if university_name:
            query_filter = "WHERE u.university_name LIKE %s"
            params.append(f"%{university_name}%")

        sql = f"""
            SELECT 
                u.university_name,
                u.country,
                u.created_at,
                c.course_id,
                c.lesson_name,
                c.description,
                s.skill_url,
                s.skill_name
            {JOIN_SKILL_ON_COURSE}
            {query_filter}
            ORDER BY u.university_id, c.course_id
        """
        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()

        lessons_data = defaultdict(lambda: {
            "university": None,
            "country": None,
            "upload_date": None,
            "description": "",
            "skills": []
        })

        for row in rows:
            key = row["course_id"]
            lessons_data[key]["title"] = row["lesson_name"]
            lessons_data[key]["description"] = row.get("description", "") or ""
            lessons_data[key]["university"] = row["university_name"]
            lessons_data[key]["country"] = row["country"]
            lessons_data[key]["upload_date"] = row["created_at"].strftime("%Y-%m-%d") if row.get(
                "created_at") else datetime.today().strftime("%Y-%m-%d")
            if row["skill_url"]:
                lessons_data[key]["skills"].append((row["skill_url"], row.get("skill_name", "")))

        occupation_cache: Dict[str, List[List[str]]] = {}
        token = get_tracker_token()

        export_items = []
        for course_id, lesson in lessons_data.items():
            skill_names = [name for _, name in lesson["skills"] if name]
            occupations = []
            seen_occ = set()
            for skill_name in skill_names:
                if skill_name in occupation_cache:
                    occ_list = occupation_cache[skill_name]
                else:
                    payload = [("keywords", skill_name),
                               ("keywords_logic", "or"),
                               ("children_logic", "or"),
                               ("ancestors_logic", "or")]
                    headers = {
                        "accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Authorization": f"Bearer {token}"
                    }
                    occ_list = []
                    try:
                        response = requests.post(
                            f"{os.environ['API_TRACKER_BASE_URL'].rstrip('/')}/occupations?page=1",
                            headers=headers,
                            data=payload,
                            verify=False
                        )

                        if response.ok:
                            for item in response.json().get("items", []):
                                occ_id = item.get("id")
                                occ_label = item.get("label")
                                if occ_id and occ_id not in seen_occ:
                                    seen_occ.add(occ_id)
                                    occ_list.append([occ_id, occ_label])
                    except Exception as e:
                        print(f"[WARNING] Occupation lookup failed for '{skill_name}': {e}")
                    occupation_cache[skill_name] = occ_list
                for oid, lbl in occupation_cache.get(skill_name, []):
                    if oid not in seen_occ:
                        seen_occ.add(oid)
                        occupations.append([oid, lbl])

            export_items.append({
                "id": [course_id],
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

        offset = (page - 1) * limit
        paginated = export_items[offset:offset + limit]
        return ExportResponse(items=paginated)

    except mysql.connector.Error as e:
        return JSONResponse(status_code=500, content={"error": f"Database error: {str(e)}"})
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


@app.get("/lookup_skill_level", tags=["Skills"], summary="Get ESCO level for a given skill name")
def lookup_skill_level(
        skill_name: str = Query(..., description="Exact skill name"),
        min_skill_level: Optional[int] = Query(None, ge=1, le=8),
        max_skill_level: Optional[int] = Query(None, ge=1, le=8)
):
    """
    Calls the tracker endpoint to retrieve the level of a skill.
    Returns the ESCO skill level (as a string) or None.
    """
    token = get_tracker_token()
    if not token:
        raise HTTPException(status_code=500, detail="Tracker token missing")

    payload = [("keywords", skill_name),
               ("keywords_logic", "or")]

    if min_skill_level is not None:
        payload.append(("min_skill_level", str(min_skill_level)))
    if max_skill_level is not None:
        payload.append(("max_skill_level", str(max_skill_level)))

    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}"
    }
    resp = requests.post(
        f"{os.environ['API_TRACKER_BASE_URL'].rstrip('/')}/skills?page=1",
        headers=headers, data=payload, verify=False
    )

    if not resp.ok:
        raise HTTPException(status_code=502, detail="Tracker call failed")

    items = resp.json().get("items", [])
    for item in items:
        if item.get("label") == skill_name:
            return {"skill_name": skill_name, "level": item.get("skill_level") or item.get("level")}

    return {"skill_name": skill_name, "level": None}


@app.post(
    "/refresh_skill_levels",
    tags=["Skills"],
    summary="Populate esco_level values by querying the tracker (adjustable range + parallel workers)",
    status_code=202,
)
def refresh_skill_levels(
        min_skill_level: int = Query(1, ge=1, le=8, description="Minimum skill level to fetch from tracker"),
        max_skill_level: int = Query(8, ge=1, le=8, description="Maximum skill level to fetch from tracker"),
        workers: int = Query(8, ge=1, le=64, description="Number of parallel DB workers"),
        background_tasks: BackgroundTasks = None,
):
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    task_id = str(uuid4())
    TASKS[task_id] = {
        "status": "queued",
        "queued_at": time.time(),
        "type": "refresh_skill_levels",
        "min_skill_level": min_skill_level,
        "max_skill_level": max_skill_level,
        "workers": workers,
    }
    background_tasks.add_task(
        _update_missing_skill_levels_task,
        task_id,
        min_skill_level,
        max_skill_level,
        workers,
    )
    return {"status": "queued", "task_id": task_id}


def _update_missing_skill_levels_task(
        task_id: str,
        min_skill_level: int,
        max_skill_level: int,
        workers: int,
):
    task = TASKS.get(task_id, {})

    def set_status(**kw):
        task.update(kw)
        TASKS[task_id] = task

    set_status(status="running", started_at=time.time(), updated=0, processed=0)

    try:
        token = get_tracker_token()
        if not token:
            logger.error(f"[Task {task_id}] Failed to get tracker token.")
            set_status(status="failed", finished_at=time.time(), error="Tracker token missing")
            return

        db_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name=f"pool_{task_id}", pool_size=workers,
                                                              **DB_CONFIG)

        conn = db_pool.get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT skill_id, skill_url FROM Skill WHERE esco_level IS NULL AND skill_url IS NOT NULL AND skill_url != ''")
        missing_skills = cur.fetchall()
        cur.close()
        conn.close()

        total_missing = len(missing_skills)
        set_status(total=total_missing)

        if not total_missing:
            logger.info(f"[Task {task_id}] No skills with missing levels found.")
            set_status(status="succeeded", finished_at=time.time(), total=0, message="No missing levels.")
            return

        logger.info(f"[Task {task_id}] Starting level refresh for {total_missing} skills.")

        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        def lookup_and_update(skill_row):
            skill_id = skill_row['skill_id']
            url_value = skill_row['skill_url']

            try:
                resp = requests.post(
                    f"{os.environ['API_TRACKER_BASE_URL'].rstrip('/')}/skills?page=1",
                    headers=headers,
                    data=f"ids={requests.utils.quote(url_value)}",
                    verify=False,
                )

                if not resp.ok:
                    logger.warning(
                        f"[Task {task_id}] API call for URL '{url_value}' failed with status {resp.status_code}.")
                    return False

                items = resp.json().get("items", [])
                if not items:
                    logger.warning(f"[Task {task_id}] API call for URL '{url_value}' returned no items.")
                    return False

                item = items[0]

                level_val = None
                skill_levels_list = item.get("skill_levels")
                if isinstance(skill_levels_list, list) and skill_levels_list:
                    level_val = ",".join(map(str, sorted(skill_levels_list)))

                if not level_val:
                    level_val = item.get("level") or item.get("skill_level")

                if level_val is None:
                    logger.warning(
                        f"[Task {task_id}] No level information found for skill_id={skill_id} (URL: {url_value}).")

                level_val_str = str(level_val)

            except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
                logger.error(f"[Task {task_id}] API/Network error for URL '{url_value}': {e}")
                return False

            try:
                db_conn = db_pool.get_connection()
                cursor = db_conn.cursor()

                logger.info(
                    f"[Task {task_id}] ATTEMPTING DB WRITE: SET esco_level = '{level_val_str}' WHERE skill_id = {skill_id}")

                cursor.execute(
                    "UPDATE Skill SET esco_level = %s WHERE skill_id = %s",
                    (level_val_str, skill_id),
                )
                db_conn.commit()

                row_count = cursor.rowcount
                cursor.close()
                db_conn.close()

                if row_count > 0:
                    logger.info(f"[Task {task_id}] DB WRITE SUCCESS for skill_id = {skill_id}.")
                    return True
                else:
                    logger.warning(f"[Task {task_id}] DB WRITE FAILED (0 rows matched) for skill_id = {skill_id}.")
                    return False

            except mysql.connector.Error as e:
                logger.error(f"[Task {task_id}] DATABASE EXCEPTION for skill_id = {skill_id}: {e}")
                return False

        updated_count = 0
        processed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(lookup_and_update, skill): skill for skill in missing_skills}
            for future in as_completed(futures):
                processed_count += 1
                try:
                    if future.result() is True:
                        updated_count += 1
                except Exception as e:
                    skill_url = futures[future]['skill_url']
                    logger.error(f"[Task {task_id}] Thread for URL '{skill_url}' raised an exception: {e}")

                if processed_count % 50 == 0 or processed_count == total_missing:
                    set_status(updated=updated_count, processed=processed_count)

        set_status(status="succeeded", finished_at=time.time(), updated=updated_count, processed=processed_count)

    except Exception as e:
        logger.exception(f"[Task {task_id}] A critical error terminated the task: {e}")
        set_status(status="failed", finished_at=time.time(), error=str(e))


UPLOAD_PDF_DIR = os.environ.get("UPLOAD_PDF_DIR", "./uploaded_pdfs")


def _safe_filename(filename: str) -> str:
    filename = os.path.basename(filename or "uploaded.pdf")
    filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", filename)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    return filename


def _flatten_values(value):
    if value is None:
        return []

    if isinstance(value, list):
        out = []
        for item in value:
            out.extend(_flatten_values(item))
        return out

    if isinstance(value, tuple):
        out = []
        for item in value:
            out.extend(_flatten_values(item))
        return out

    return [value]

def _guess_lesson_name_from_block(block: str) -> Optional[str]:
    lines = [l.strip() for l in (block or "").splitlines() if l.strip()]

    skip = re.compile(
        r"^(ECTS|Credits|Semester|Assessment|Aims|Prerequisites|Module|Course|School|Department|Level|Language)$",
        re.I
    )

    code_pat = re.compile(r"^(?:[A-Z]{2,10}\d{3,6}|\d{5,6})$")

    for i, line in enumerate(lines):
        if code_pat.match(line):
            for nxt in lines[i + 1:i + 5]:
                if not code_pat.match(nxt) and not skip.match(nxt) and len(nxt) > 3:
                    return nxt[:255]

    for line in lines[:15]:
        if not code_pat.match(line) and not skip.match(line) and len(line) > 3:
            return line[:255]

    return None


def _guess_course_code_from_block(block: str) -> Optional[str]:
    m = re.search(r"\b([A-Z]{2,10}\d{3,6}|\d{5,6})\b", block or "")
    return m.group(1) if m else None

def _looks_like_course_block(block: str) -> bool:
    """Cheap validator for a curriculum/course block."""
    b = _clean_pdf_text(_ensure_text(block))[:2500]
    if len(b.strip()) < 120:
        return False

    score = 0
    patterns = [
        r"\b(?:course|unit|module)\s+(?:title|name)\s*:",
        r"\b(?:course|unit|module)\s+(?:code|number|id)\s*:",
        r"\b[A-Z]{1,12}[A-Z_\-]*\d{1,6}[A-Z]?\b",
        r"\bECTS\b|\bcredits?\b|credit\s+rating",
        r"\bdescription\b|\boverview\b|\baims?\b|\bobjectives?\b",
        r"learning\s+outcomes?",
        r"\bassessment\b|\bexam\b",
        r"\bsemester\b|teaching\s+period|\bterm\b",
    ]
    for pat in patterns:
        if re.search(pat, b, re.I):
            score += 1
    return score >= 3


def _extract_title_from_start_match(text: str, pos: int) -> str:
    snippet = text[pos:pos + 500]
    lines = [re.sub(r"\s+", " ", l).strip(" •●▪*-\t") for l in snippet.splitlines() if l.strip()]

    m = re.search(r"(?:course|unit|module)\s+title\s*:\s*([^\n\r]{3,180})", snippet, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    m = re.search(r"^\s*([A-Z][A-Za-z0-9&,.:'’()/+ \-–—]{3,180}?)\s*ECTS\s+credits\s*:", snippet, re.I | re.M)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    for i, line in enumerate(lines[:5]):
        if re.search(r"^(?:unit|course|module)\s+(?:code|number|id)\b", line, re.I):
            if i > 0:
                return lines[i - 1]

    m = re.search(r"^\s*(?:[A-Z]{1,12}[A-Z_\-]*\d{1,6}[A-Z]?|\d{4,8})\s+([A-Z][^\n]{3,180})", snippet, re.M)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()

    return lines[0] if lines else ""


def split_course_blocks(text: str) -> List[str]:
    """
    Universal curriculum splitter.

    It detects common course starts without relying on one university format:
    - Course title: X / o Course title: X
    - X ECTS credits:5 Course code:Y
    - X then Unit code Y / Course code Y / Module code Y
    - CODE Title
    - X: 2025-2026
    """
    text = _clean_pdf_text(_ensure_text(text))
    if not text.strip():
        return []

    course_start_re = re.compile(
        r"""
        (?imx)
        (?=
            ^\s*(?:[•●▪\-\*]\s*)?o?\s*(?:course|unit|module)\s+title\s*:\s*[^\n\r]{3,180}
            |
            ^\s*[A-Z][A-Za-z0-9&,.:'’()/+ \-–—]{3,180}?\s*ECTS\s+credits\s*:\s*\d+(?:\.\d+)?\s*(?:course|unit|module)\s+code\s*:
            |
            ^\s*[A-Z][A-Za-z0-9&,.:'’()/+ \-–—]{3,180}\s*\n+\s*(?:unit|course|module)\s+(?:code|number|id)\s*[:\-]?\s*[A-Z]{1,12}[A-Z_\-]*\d{1,6}[A-Z]?
            |
            ^\s*(?:[A-Z]{1,12}[A-Z_\-]*\d{1,6}[A-Z]?|\d{4,8})\s+[A-Z][A-Za-z0-9&,.:'’()/+ \-–—]{3,180}
            |
            ^\s*[A-Z][A-Za-z0-9&,.:'’()/+ \-–—]{3,180}?\s*[:\-–—]\s*20\d{2}\s*[-–—/]\s*20\d{2}
            |
            ^\s*
            (?:
                [A-Z][A-Z0-9&,\-–—:'’()./+ ]{8,180}
                |
                [A-Z][A-Za-z0-9&,\-–—:'’()./+ ]{8,180}
            )
            \s*\n+
            (?:Aims|Overview|Description|Learning outcomes|Course learning outcomes|Objectives|Assessment)\b
        )
        """,
        re.MULTILINE | re.VERBOSE | re.IGNORECASE,
    )

    bad_titles = {
        "overview", "aims", "assessment", "syllabus", "course unit fact file",
        "pre/co-requisites", "pre/co requisites", "prerequisites", "reading list",
        "references", "bibliography", "learning outcomes", "course structure",
        "details", "description", "objectives", "note", "semester 1", "semester 2",
        "semester 3", "semester 4", "semester 5", "semester 6"
    }

    starts = []
    seen = set()
    for m in course_start_re.finditer(text):
        pos = m.start()
        if pos in seen:
            continue
        seen.add(pos)

        title = _extract_title_from_start_match(text, pos)
        title_low = re.sub(r"\s+", " ", title).strip().lower()
        if not title_low or title_low in bad_titles:
            continue
        if len(title_low) < 3 or len(title_low) > 220:
            continue

        window = text[pos:pos + 2500]
        if not _looks_like_course_block(window):
            strong = re.search(r"(?:course|unit|module)\s+title\s*:|ECTS\s+credits\s*:|(?:unit|course|module)\s+code\s*:", window, re.I)
            if not strong:
                continue

        starts.append(pos)

    starts = sorted(set(starts))

    blocks = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if len(block) < 250:
            continue
        if not _looks_like_course_block(block):
            continue
        blocks.append(block)

    return blocks

def _labels_to_structured_courses(
        labels: List[Dict[str, Any]],
        university_name: str,
        source_file: Optional[str] = None,
        website: Optional[str] = None
) -> List[Dict[str, Any]]:

    courses = []
    current = None

    multi_value_fields = {
        "professor",
        "degree_title",
        "msc_bsc",
        "language",
        "semester",
        "hours",
        "ects",
        "website",
    }

    text_fields = {
        "description",
        "objectives",
        "learning_outcomes",
        "course_content",
        "assessment",
        "exam",
        "prerequisites",
        "general_competences",
        "educational_material",
    }

    for item in labels or []:
        label = (
            item.get("class_or_confidence")
            or item.get("label")
            or item.get("entity_group")
            or ""
        ).strip()

        token = (
            item.get("token")
            or item.get("word")
            or item.get("text")
            or ""
        ).strip()

        confidence = item.get("confidence")

        if not label or not token:
            continue

        if label == "lesson_name":
            if current and current.get("lesson_name"):
                courses.append(current)

            current = {
                "lesson_name": token,
                "website": website,
                "websites": [website] if website else [],
                "source_file": source_file,
                "university_name": university_name,
                "labels": [],
                "extras": {},
            }

        if current is None:
            current = {
                "lesson_name": None,
                "website": website,
                "websites": [website] if website else [],
                "source_file": source_file,
                "university_name": university_name,
                "labels": [],
                "extras": {},
            }

        current["labels"].append({
            "class_or_confidence": label,
            "token": token,
            "confidence": confidence
        })

        if label == "lesson_name":
            current["lesson_name"] = token

        elif label in text_fields:
            if current.get(label):
                current[label] = f"{current[label]}\n{token}"
            else:
                current[label] = token

        elif label in multi_value_fields:
            values = _flatten_values(current.get(label))
            if token not in values:
                values.append(token)
            current[label] = values

            if label == "website":
                urls = _flatten_values(current.get("websites"))
                if token not in urls:
                    urls.append(token)
                current["websites"] = urls

        else:
            existing = _flatten_values(current["extras"].get(label))
            if token not in existing:
                existing.append(token)
            current["extras"][label] = existing if len(existing) > 1 else existing[0]

    if current and (current.get("lesson_name") or current.get("labels")):
        courses.append(current)

    for course in courses:
        course["websites"] = [
            str(w) for w in _flatten_values(course.get("websites"))
            if w and not isinstance(w, (dict, list, tuple))
        ]

        if course.get("website") and isinstance(course["website"], list):
            course["website"] = course["website"][0] if course["website"] else None

    return _prepare_and_merge_courses(
        courses,
        university_name,
        file_hint=source_file,
        fuzzy_threshold=88
    )

def extract_one_course_with_llm(
        block: str,
        university_name: str,
        source_file: str,
        website: Optional[str] = None,
        model: str = "llama3.1",
        allow_missing_lesson_name: bool = False
) -> Optional[Dict[str, Any]]:

    ollama_url = (
        os.getenv("OLLAMA_HOST")
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_URL")
        or "http://host.docker.internal:11434"
    ).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", model)

    prompt = f"""
You extract exactly ONE university course/module from the text below.
A course is only valid if it has a title. You MUST identify the title.
If the course code is present, the title is usually the meaningful line after the code.
Never return null for lesson_name unless the block is definitely not a course.
Also, in case a category of text is titled differently, try to fit it based off meaning to a different category (for example, Aims could be Objectives)

Return ONLY valid JSON with exactly this shape:
{{
  "lesson_name": null,
  "course_code": null,
  "website": null,
  "ects": null,
  "language": null,
  "semester_label": null,
  "description": null,
  "objectives": null,
  "learning_outcomes": null,
  "course_content": null,
  "assessment": null,
  "exam": null,
  "prerequisites": null,
  "general_competences": null,
  "educational_material": null,
  "professor": [],
  "msc_bsc": null,
  "degree_title": null
}}

Rules:
- Return one course only.
- lesson_name must be the title, not the course code.
- Put the code in course_code.
- Do not invent missing values.
- If the text is not a real course/module, return {{"lesson_name": null}}.

University: {university_name}
Source file: {source_file}

TEXT:
{block}
"""

    response = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 4096
            }
        },
        timeout=None
    )
    response.raise_for_status()

    raw = response.json().get("response", "{}")
    logger.info("OLLAMA ONE COURSE RAW:\n%s", raw[:3000])

    try:
        course = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Invalid JSON for block:\n%s", raw[:3000])
        return None

    name = (
        course.get("lesson_name")
        or course.get("Course Title")
        or course.get("Course Name")
        or course.get("Module Title")
        or course.get("Title")
    )

    if not allow_missing_lesson_name:
        name = name or _guess_lesson_name_from_block(block)

    name = str(name or "").strip()

    if not name:
        if allow_missing_lesson_name:
            course["lesson_name"] = None
            course["university_name"] = university_name
            course["source_file"] = source_file
            course["website"] = course.get("website") or website
            course["extras"] = course.get("extras") or {}
            return course

        logger.warning("Skipping block because no lesson name could be inferred. Block sample:\n%s", block[:1000])
        return None

    course["lesson_name"] = name[:255]

    name = str(name or "").strip()

    if not name:
        logger.warning("Skipping block because no lesson name could be inferred. Block sample:\n%s", block[:1000])
        return None

    course["lesson_name"] = name[:255]

    if not course.get("course_code"):
        course["course_code"] = _guess_course_code_from_block(block)

    course["university_name"] = university_name
    course["source_file"] = source_file
    course["website"] = course.get("website") or website
    course["extras"] = course.get("extras") or {}

    if course.get("course_code"):
        course["extras"]["course_code"] = course["course_code"]

    return course

def _sanitize_llm_course(course: Dict[str, Any]) -> Dict[str, Any]:
    course = dict(course or {})

    text_fields = {
        "description",
        "objectives",
        "learning_outcomes",
        "course_content",
        "assessment",
        "exam",
        "prerequisites",
        "general_competences",
        "educational_material",
    }

    scalar_fields = {
        "lesson_name",
        "course_code",
        "website",
        "ects",
        "language",
        "semester_label",
        "msc_bsc",
        "degree_title",
    }

    for field in text_fields:
        if field in course:
            course[field] = _ensure_text(course.get(field)).strip() or None

    for field in scalar_fields:
        if field in course:
            value = course.get(field)
            if isinstance(value, (list, tuple, set)):
                value = next((str(v).strip() for v in value if str(v).strip()), None)
            elif value is not None:
                value = str(value).strip()
            course[field] = value or None

    professors = course.get("professor") or course.get("professors") or []
    if isinstance(professors, str):
        professors = [professors]
    elif not isinstance(professors, list):
        professors = [str(professors)]

    course["professor"] = [str(p).strip() for p in professors if str(p).strip()]

    if isinstance(course.get("extras"), dict):
        course["extras"] = {
            str(k): _ensure_text(v).strip() if isinstance(v, (list, tuple, set, dict)) else v
            for k, v in course["extras"].items()
        }

    return course

def _force_fixed_chunks(
        text: str,
        chunk_size: int = 2000,
        overlap: int = 250,
        limit: int = 40000
) -> List[str]:
    text = _clean_pdf_text(_ensure_text(text))[:limit].strip()

    if not text:
        return []

    chunk_size = max(200, int(chunk_size or 2000))
    overlap = max(0, min(int(overlap or 0), chunk_size - 1))

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def _force_fixed_chunks(text: str, chunk_size: int = 2000, overlap: int = 250, limit: int = 40000) -> List[str]:
    text = _clean_pdf_text(_ensure_text(text))[:limit].strip()

    chunk_size = max(200, int(chunk_size or 2000))
    overlap = max(0, min(int(overlap or 0), chunk_size - 1))

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap

    assert all(len(c) <= chunk_size for c in chunks), "force_chunk failed"

    return chunks

def detect_course_starts_with_ollama(
        pages: List[str],
        model: str,
        batch_size: int = 12,
        overlap_pages: int = 1
) -> List[Dict[str, Any]]:
    """Detect course starts using regex first, then Ollama as fallback per small page batch."""

    def _regex_starts_from_preview(previews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        heading_re = re.compile(
            r"""
            (?mx)
            ^\s*
            (
                (?:[A-Z]{2,10}\d{3,6}|\d{5,6})\s+[^
]{4,160}
                |
                (?:Module|Course|Course\s+Unit|Unit|Subject|Paper)\s*[:\-–—]?\s+[^
]{4,160}
                |
                [A-Z][A-Za-z0-9&,+/().'’:\-–— ]{4,160}(?:\s*[:\-–—|]\s*20\d{2}\s*[-–—/]\s*20\d{2})?
                |
                ^\s*(?:o\s*)?Course\s+title\s*:\s*
                [^\n]{3,180}
                |
                ^\s*
                [A-Z][A-Za-z0-9&,\-–—:'’()./+ ]{3,180}
                \s*
                ECTS\s+credits\s*:\s*\d+
                \s*
                Course\s+code\s*:
                |
                ^\s*
                [A-Z][A-Za-z0-9&,\-–—:'’()./+ ]{3,180}
                \s*\n+
                (?:Unit|Course|Module)\s+(?:code|number|id)\s*[:\-]?
            )
            \s*$
            """,
            re.MULTILINE | re.VERBOSE,
        )

        negative = {
            "home", "contents", "table of contents", "reading list", "references",
            "bibliography", "assessment", "overview", "syllabus", "learning outcomes",
            "lecture notes", "further reading", "recommended reading", "student support",
            "student handbook", "contact details", "timetable", "examinations",
            "coursework", "appendix", "introduction", "department", "school",
        }
        positive_hints = [
            "ects", "credits", "lecturer", "instructor", "module leader", "teacher",
            "overview", "learning outcomes", "prerequisites", "assessment", "syllabus",
            "term", "semester", "course code", "hours", "aims", "objectives",
            "degree", "level", "language", "exam", "coursework",
        ]

        found = []
        for p in previews:
            page_no = int(p["page"])
            page_text = p["text"]
            for m in heading_re.finditer(page_text):
                title = re.sub(r"\s+", " ", m.group(1)).strip(" -–—:	")
                low = title.lower()
                if len(title) < 5 or len(title) > 170:
                    continue
                if low in negative:
                    continue
                if re.fullmatch(r"\d+", title):
                    continue
                if re.search(r"^(figure|table|chapter|section)\s+\d+", low):
                    continue
                if re.search(r"(pp\.|doi|isbn|journal|proceedings)", low):
                    continue

                context = page_text[m.end(): min(len(page_text), m.end() + 1600)].lower()
                score = 0
                if re.search(r"20\d{2}\s*[-–—/]\s*20\d{2}", title):
                    score += 3
                if re.search(r"^(?:[A-Z]{2,10}\d{3,6}|\d{5,6})", title):
                    score += 3
                if re.match(r"^(Module|Course|Course\s+Unit|Unit|Subject|Paper)", title, re.I):
                    score += 2
                score += sum(1 for h in positive_hints if h in context)

                if score < 2:
                    continue

                found.append({
                    "course_title": re.sub(r"\s*[:\-–—|]\s*20\d{2}\s*[-–—/]\s*20\d{2}\s*$", "", title).strip(),
                    "start_page": page_no,
                    "evidence": page_text[m.start(): min(len(page_text), m.end() + 160)][:240],
                })
        return found

    all_starts: List[Dict[str, Any]] = []
    pages = pages or []
    step = max(1, batch_size - overlap_pages)

    for batch_start in range(0, len(pages), step):
        batch_pages = pages[batch_start:batch_start + batch_size]
        previews = []

        for offset, page_text in enumerate(batch_pages):
            page_no = batch_start + offset + 1
            text = _clean_pdf_text(_ensure_text(page_text))
            if text.strip():
                previews.append({"page": page_no, "text": text[:3500]})

        if not previews:
            continue

        regex_starts = _regex_starts_from_preview(previews)
        if regex_starts:
            logger.info(
                "Regex boundary batch pages %s-%s detected %d starts",
                previews[0]["page"], previews[-1]["page"], len(regex_starts)
            )
            all_starts.extend(regex_starts)
            continue

        prompt = f"""
Find REAL university course/module start headings in these PDF pages.

Return ONLY valid JSON array. No markdown, no prose.

Format:
[
  {{
    "course_title": "Advanced Security",
    "start_page": 12,
    "evidence": "Advanced Security: 2025-2026"
  }}
]

A course/module heading may look like:
- Advanced Security: 2025-2026
- Algorithms and Data Structures
- COMP3021 Machine Learning
- Module: Distributed Systems
- Artificial Intelligence | Hilary Term
- Computational Biology
- MSc Thesis Project
- Research Methods in AI
- Data Mining and Knowledge Discovery
- Course Unit — Computer Vision
- Paper B: Machine Learning
- Unit 4 - Database Systems

A real course/module usually has nearby words like:
ECTS, Credits, Lecturer, Instructor, Module leader, Learning outcomes,
Assessment, Overview, Syllabus, Semester, Term, Prerequisites, Coursework,
Hours, Aims, Objectives, Degree, Level, Language.

Do NOT return reading-list papers, bibliography entries, references, chapters,
table-of-contents items, page headers/footers, navigation text, or generic section
headings such as Assessment, Overview, Introduction, Syllabus.

If a page continues the previous course, do not create a new item.
Use the real page number from the provided page field.

Pages:
{json.dumps(previews, ensure_ascii=False)}
"""
        raw = _ollama_generate(prompt, model=model, temperature=0.0)
        starts = _extract_json_array(raw)

        if not starts:
            logger.warning(
                "LLM returned no starts for pages %s-%s. Raw response sample: %s",
                previews[0]["page"], previews[-1]["page"], raw[:1000]
            )

        logger.info(
            "LLM boundary batch pages %s-%s detected %d starts",
            previews[0]["page"], previews[-1]["page"], len(starts)
        )
        all_starts.extend(starts)

    deduped: Dict[tuple, Dict[str, Any]] = {}
    for s in all_starts:
        title = str(s.get("course_title") or "").strip()
        try:
            page = int(s.get("start_page"))
        except Exception:
            continue

        if not title or page < 1 or page > len(pages):
            continue

        key = (page, re.sub(r"\W+", " ", title.lower()).strip())
        deduped[key] = {"course_title": title, "start_page": page, "evidence": s.get("evidence")}

    return sorted(deduped.values(), key=lambda x: (int(x["start_page"]), x["course_title"].lower()))


def build_course_blocks_from_starts(
        pages: List[str],
        starts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    starts = sorted(
        [s for s in starts if s.get("course_title") and s.get("start_page")],
        key=lambda x: int(x["start_page"])
    )

    blocks = []

    for i, start in enumerate(starts):
        start_page = int(start["start_page"])
        end_page = (
            int(starts[i + 1]["start_page"]) - 1
            if i + 1 < len(starts)
            else len(pages)
        )

        text = "\n\n".join(pages[start_page - 1:end_page])

        blocks.append({
            "lesson_name_hint": start["course_title"],
            "start_page": start_page,
            "end_page": end_page,
            "text": text
        })

    return blocks
    
@app.post(
    "/pdf/upload_and_process",
    tags=["PDF", "CurricuNLP"],
    summary="Upload a PDF, run CurricuNLP, segment it into structured courses"
)
async def upload_pdf_and_process(
        file: UploadFile = File(...),
        university_name: Optional[str] = Form(None),
        country: Optional[str] = Form(None),
        website: Optional[str] = Form(None),
        save_to_db: bool = Form(False),
        translate: bool = Form(False),
        chunk_size: int = Form(2000),
        overlap: int = Form(250),
        max_chars: int = Form(80000),
        background_tasks: BackgroundTasks = None,
        split_mode: Literal[
            "full_text",
            "per_page",
            "force_chunk",
            "llm_course_boundaries"
        ] = Form(
            "full_text",
            description=(
                "Course splitting mode: "
                "full_text = split whole PDF into detected course blocks, "
                "per_page = split courses page by page, "
                "force_chunk = ignore course splitting and process fixed-size chunks, "
                "llm_course_boundaries = use Ollama to detect course start pages"
            ),
        )
        ):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(UPLOAD_PDF_DIR, exist_ok=True)

    safe_name = f"{uuid4()}_{_safe_filename(file.filename)}"
    pdf_path = os.path.join(UPLOAD_PDF_DIR, safe_name)

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    full_text = _ensure_text(extract_text_from_pdf_best(pdf_path))

    pages = extract_text_from_pdf(pdf_path)
    if not isinstance(pages, list):
        pages = []
    if not pages and full_text.strip():
        pages = [full_text]
        

    cleaned_text = _clean_pdf_text(full_text)



    if not cleaned_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No usable text could be extracted from the PDF."
        )

    detected_lang = _detect_lang(cleaned_text)

    if translate and detected_lang != "en":
        try:
            cleaned_text = _translate_hf(cleaned_text)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Translation failed: {e}")

    if university_name and university_name.strip():
        uni_guess = university_name.strip()
    else:
        uni_guess = re.sub(
            r"[_\W]+",
            " ",
            os.path.splitext(file.filename)[0]
        ).strip()

    meta = _find_uni_by_name(uni_guess)

    final_university_name = (
            university_name
            or meta.get("name")
            or uni_guess
            or "Unknown University"
    )

    final_country = (
            country
            or meta.get("country")
            or "Unknown"
    )

    domain = meta.get("domain")

    try:
        labels = []
        structured_courses = []

        if split_mode == "per_page" and pages:
            blocks = []

            for page_no, page_text in enumerate(pages, start=1):
                cleaned_page = _clean_pdf_text(page_text)

                for block in split_course_blocks(cleaned_page):
                    blocks.append(f"[PAGE {page_no}]\n{block}")

        elif split_mode == "per_page":
            blocks = split_course_blocks(cleaned_text)

        elif split_mode == "llm_course_boundaries":
            if not pages:
                logger.warning("No page list available for llm_course_boundaries; falling back to full_text splitter")
                blocks = split_course_blocks(cleaned_text)
            else:
                starts = detect_course_starts_with_ollama(
                    pages=pages,
                    model=os.getenv("OLLAMA_MODEL", "gemma3:4b")
                )
                logger.info("LLM/regex detected %d course starts", len(starts))

                course_blocks = build_course_blocks_from_starts(pages, starts)

                blocks = [
                    f"[COURSE_HINT: {b['lesson_name_hint']}]\n"
                    f"[PAGES: {b['start_page']}-{b['end_page']}]\n\n"
                    f"{b['text']}"
                    for b in course_blocks
                ]

                if not blocks:
                    logger.warning("llm_course_boundaries produced 0 blocks; falling back to full_text splitter")
                    blocks = split_course_blocks(cleaned_text)

        elif split_mode == "force_chunk":
            blocks = _force_fixed_chunks(
                full_text,
                chunk_size=chunk_size,
                overlap=overlap,
                limit=max_chars
            )

            logger.info(
                "FORCE_CHUNK sizes: %s",
                [len(b) for b in blocks[:20]]
            )

        else:
            blocks = split_course_blocks(cleaned_text)

        # Hard fallback: never let full_text/per_page/llm modes silently return zero.
        # If deterministic course splitting fails for a new university format, process
        # fixed-size chunks so extraction still returns courses instead of an empty payload.
        if not blocks and split_mode != "force_chunk":
            logger.warning(
                "%s produced 0 course blocks; falling back to fixed chunks chunk_size=%s overlap=%s",
                split_mode,
                chunk_size,
                overlap,
            )
            blocks = _force_fixed_chunks(
                cleaned_text,
                chunk_size=chunk_size or 8000,
                overlap=overlap or 800,
                limit=max_chars,
            )
            logger.info("Fallback fixed chunks created %d blocks", len(blocks))

        if split_mode == "force_chunk":
            logger.info("Created %d forced fixed chunks", len(blocks))
        elif split_mode == "llm_course_boundaries":
            logger.info("Created %d LLM/regex course boundary blocks", len(blocks))
        else:
            logger.info("Detected %d possible course blocks", len(blocks))

        for i, block in enumerate(blocks):
            logger.info(
                "Extracting block %d/%d split_mode=%s chars=%d",
                i + 1,
                len(blocks),
                split_mode,
                len(block)
            )

            try:
                course = extract_one_course_with_llm(
                    block=block,
                    university_name=final_university_name,
                    source_file=os.path.basename(pdf_path),
                    website=website,
                    model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
                    allow_missing_lesson_name=True
                )

                if course and course.get("lesson_name"):
                    structured_courses.append(course)

                elif structured_courses:
                    structured_courses[-1] = _append_continuation_to_course(
                        structured_courses[-1],
                        block
                    )

                else:
                    logger.warning(
                        "No lesson name in first forced chunk/block. Skipping sample:\n%s",
                        block[:1000]
                    )

            except Exception as e:
                logger.exception("Failed block %d/%d: %s", i + 1, len(blocks), e)

        structured_courses = [
            _sanitize_llm_course(c)
            for c in structured_courses
            if c and c.get("lesson_name")
        ]

        structured_courses = _prepare_and_merge_courses(
            structured_courses,
            final_university_name,
            file_hint=os.path.basename(pdf_path),
            fuzzy_threshold=88
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"CurricuNLP pipeline failed: {e}")


    payload = {
        "university_name": final_university_name,
        "country": final_country,
        "university_meta": {
            "name": final_university_name,
            "country": final_country,
            "domain": domain
        },
        "source_file": os.path.basename(pdf_path),
        "original_filename": file.filename,
        "language": {
            "detected": detected_lang,
            "translated": bool(translate and detected_lang != "en")
        },
        "text_stats": {
            "characters": len(cleaned_text),
            "max_chars_used": max_chars,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "split_mode": split_mode,
            "pages_detected": len(pages) if 'pages' in locals() else 0,
            "blocks_detected": len(blocks) if 'blocks' in locals() else 0
        },
        "labels_count": len(labels or []),
        "lesson_count": len(structured_courses),
        "courses": structured_courses
    }

    payload = _json_safe(payload)
    safe_payload = _json_safe(payload)
    logger.info("SAFE PAYLOAD: %s", json.dumps(safe_payload)[:5000])

    if save_to_db:
        if not is_database_connected(DB_CONFIG):
            raise HTTPException(status_code=500, detail="Database connection failed.")

        task_id = str(uuid4())
        TASKS[task_id] = {
            "status": "queued",
            "queued_at": time.time(),
            "type": "upload_pdf_and_process_save",
            "source_file": os.path.basename(pdf_path),
            "university_name": final_university_name
        }

        if background_tasks is None:
            raise HTTPException(status_code=500, detail="BackgroundTasks is not available.")
        background_tasks.add_task(_save_payload_task, task_id, payload)

        payload["db_save"] = {
            "status": "queued",
            "task_id": task_id,
            "note": "Poll /curriculum-skills/db/tasks/{task_id} for progress."
        }

    return JSONResponse(content=_json_safe(payload))



@app.get("/db/ping", tags=["Debug"])
def db_ping():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.fetchall()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB ping failed: {e}")
    finally:
        try:
            cur.close();
            conn.close()
        except Exception:
            pass


from fastapi.routing import APIRoute

def add_prefixed_routes(app: FastAPI, prefix: str = "/curriculum-skills"):
    existing = {
        (tuple(sorted(route.methods)), route.path)
        for route in app.router.routes
        if isinstance(route, APIRoute)
    }

    for route in list(app.router.routes):
        if not isinstance(route, APIRoute):
            continue

        if route.path.startswith(prefix):
            continue

        new_path = prefix + route.path
        key = (tuple(sorted(route.methods)), new_path)

        if key in existing:
            continue

        app.router.add_api_route(
            new_path,
            route.endpoint,
            response_model=route.response_model,
            status_code=route.status_code,
            tags=route.tags,
            dependencies=route.dependencies,
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            responses=route.responses,
            deprecated=route.deprecated,
            methods=list(route.methods),
            include_in_schema=False,
            response_class=route.response_class,
            name=f"{route.name}_prefixed",
            callbacks=route.callbacks,
            openapi_extra=route.openapi_extra,
        )

add_prefixed_routes(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

