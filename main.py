from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
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
from threading import Lock

try:
    import orjson as _jsonlib
    def _json_load(f): return _jsonlib.loads(f.read())
except Exception:
    import json as _jsonlib
    def _json_load(f): return _jsonlib.load(f)

import logging
import asyncio
import os
import json
import re
import requests
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



logger = logging.getLogger("db_saver")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


TASKS: Dict[str, Dict[str, Any]] = globals().get("TASKS", {})
globals()["TASKS"] = TASKS

load_dotenv()

CURRICU_CLIENT = Client("marfoli/CurricuNLP")

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

def call_curriculnlp_on_text(full_text: str, max_chars=40000, chunk_size=2000, overlap=250, pause=0.2, retries=2):
    client = Client("marfoli/CurricuNLP")
    chunks = _chunk_text(_clean_pdf_text(full_text), chunk_size=chunk_size, overlap=overlap, limit=max_chars)
    out = []
    for ch in chunks:
        attempt = 0
        while True:
            try:
                res = client.predict(text=ch, api_name="/predict")
                out.extend(res or [])
                break
            except Exception:
                attempt += 1
                if attempt > retries:
                    break
                time.sleep(0.5 * attempt)
        time.sleep(pause)
    return _merge_ner(out)


WORLD_UNI_PATH = os.environ.get(
    "WORLD_UNI_PATH",
    "/mnt/data/world_universities_and_domains.json"
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
    version="1.3.0",   
    description="API for skill extraction and course search (DB + domains JSON).",
    root_path="/curriculum-skills"
)

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
    class_or_confidence: str = Field(..., description="NER label name (e.g., 'lesson_name', 'ects', 'language', 'professor').")
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

class SaveJSONRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON to store (must contain 'university_name' unless normalize_university=true).")
    normalize_university: bool = Field(False, description="If true, try to fill university_name/country from a guessed name via _find_uni_by_name().")

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
    normalize_university: bool = Field(True, description="Fill university_name/country via _find_uni_by_name if possible")
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
        uni_name = (payload.get("university_name") if isinstance(payload, dict) else None) or meta.get("name") or uni_guess
        country = (
            (payload.get("country") if isinstance(payload, dict) else None)
            or (payload.get("university_country") if isinstance(payload, dict) else None)
            or meta.get("country")
            or "Unknown"
        )
    else:
        meta = {"name": uni_guess, "country": (payload.get("country") if isinstance(payload, dict) else None)}
        uni_name = (payload.get("university_name") if isinstance(payload, dict) else None) or uni_guess
        country  = (payload.get("country") if isinstance(payload, dict) else None) or "Unknown"

    if isinstance(payload, dict) and isinstance(payload.get("courses"), list) and payload["courses"]:
        courses = payload["courses"]
        source  = "payload.courses"
    else:
        flat_course = None
        if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
            flat_course = _course_from_curriculnlp_labels_payload(payload, filename_hint=file_base_hint)
            source = "labels->course"
        elif isinstance(payload, list) and payload and all(isinstance(x, dict) and "class_or_confidence" in x for x in payload):
            flat_course = _course_from_curriculnlp_labels_payload({"labels": payload}, filename_hint=file_base_hint)
            source = "labels(list)->course"
        elif isinstance(payload, dict):
            course_like = {
                "lesson_name","title","name","website","url","description","objectives","learning_outcomes",
                "course_content","assessment","exam","prerequisites","general_competences","educational_material",
                "ects","language","professor","professors","hours","msc_bsc","msc_bsc_list","degree_title","degree_titles",
                "semester_number","semester_label","mand_opt","fee_list","extras","year","attendance_type","attendence_type"
            }
            if course_like & set(payload.keys()):
                flat_course = dict(payload)
                if not isinstance(flat_course.get("lesson_name"), str) or not flat_course["lesson_name"].strip():
                    flat_course["lesson_name"] = (flat_course.get("title") or flat_course.get("name") or file_base_hint or "Untitled Course")[:255]
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

        pattern   = req_data.get("filename_pattern") or "*.json"
        recursive = bool(req_data.get("recursive"))
        glob_pat  = os.path.join(base_dir, "**", pattern) if recursive else os.path.join(base_dir, pattern)
        files     = [f for f in glob.glob(glob_pat, recursive=recursive) if os.path.isfile(f)]
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
        stop_on_error        = bool(req_data.get("stop_on_error"))
        use_process_pool     = bool(req_data.get("use_process_pool"))
        workers              = int(req_data.get("workers") or max(2, (os.cpu_count() or 4)))
        db_workers           = int(req_data.get("db_workers") or 2)

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
                fp  = futures[fut]
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
    filename: Optional[str] = Query(None, description="Specific JSON file to import. If omitted, imports ALL *.json in directory")
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
        payload["country"] = payload.get("country") or payload.get("university_country") or meta.get("country") or "Unknown"

        try:
            write_json_to_database(payload, DB_CONFIG)
            return {"file": os.path.basename(path), "status": "imported", "university": payload["university_name"], "country": payload["country"]}
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
            payload["country"] = payload.get("country") or payload.get("university_country") or meta.get("country") or "Unknown"

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
    if os.path.isabs(request.pdf_name) and os.path.exists(request.pdf_name):
        pdf_path = request.pdf_name
    else:
        curriculum_folder = "curriculum"
        os.makedirs(curriculum_folder, exist_ok=True)
        matches = [f for f in os.listdir(curriculum_folder) if f.endswith(".pdf") and request.pdf_name in f]
        if not matches:
            raise HTTPException(status_code=404, detail="PDF not found")
        pdf_path = os.path.join(curriculum_folder, matches[0])

    full_text = extract_text_from_pdf_best(pdf_path)

    university_name_guess = re.sub(r"[_\W]+", " ", os.path.basename(pdf_path).replace(".pdf", "")).strip()
    meta = _find_uni_by_name(university_name_guess)
    university_name = meta.get("name") or university_name_guess
    university_country = meta.get("country")
    domain = meta.get("domain")

    labels = call_curriculnlp_on_text(full_text, chunk_size=2000, overlap=150)

    return {
        "file": os.path.basename(pdf_path),
        "university_meta": {"name": university_name, "country": university_country, "domain": domain},
        "labels": labels
    }



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
    return task


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
        try:
            extractor_url = os.environ["API_SKILL_EXTRACTOR_BASE_URL"].rstrip("/") + "/extract-skills"
            resp = requests.post(
                extractor_url,
                headers={"Content-Type": "application/json", "accept": "application/json"},
                json=[t] if isinstance(t, str) else t,
                verify=False,
                timeout=60
            )
            urls: Set[str] = set()
            if resp.ok:
                data = resp.json()
                if isinstance(data, list):
                    for group in data:
                        if isinstance(group, list):
                            for su in group:
                                if isinstance(su, str):
                                    urls.add(su)
                elif isinstance(data, dict) and "items" in data:
                    for it in data["items"]:
                        ids = it.get("id") or []
                        if isinstance(ids, list):
                            for su in ids:
                                if isinstance(su, str):
                                    urls.add(su)
            return urls
        except Exception:
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
    match: Literal["exact","like"],
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

@app.post("/calculate_skillnames", tags=["Skills"], summary="Extract and upsert skills (background task)", status_code=202)
def calculate_skillnames(
    university_name: Optional[str] = "",
    lesson_name: Optional[str] = None,
    match: Literal["exact","like"] = "like",
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



def _calc_all_skillnames_task(master_task_id: str, lesson_name: Optional[str], match: Literal["exact","like"]):
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
    match: Literal["exact","like"],
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
    DB_WORKERS     = int(os.getenv("SKILL_DB_WORKERS", "6"))

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
            set_status(status="failed", finished_at=time.time(), error=f"University not found in DB: '{university_name}'")
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
                results.setdefault(title, defaultdict(list))
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
        nonempty_pages = sum(1 for p in (pages or []) if isinstance(p, str) and p.strip()) if isinstance(pages, list) else None
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

        return {
            "skill_query": skill,
            "universities": out,
            "university_frequency": dict(freq)
        }
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()
        
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

        return {"skill_url_query": skill_url, "results": grouped}
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
            payload["country"] = payload.get("country") or payload.get("university_country") or meta.get("country") or "Unknown"

    if not payload.get("university_name"):
        raise HTTPException(status_code=400, detail="payload.university_name is required (or enable normalize_university).")

    task_id = str(uuid4())
    TASKS[task_id] = {"status": "queued", "queued_at": time.time()}
    background_tasks.add_task(_save_payload_task, task_id, payload)
    return {
        "status": "queued",
        "task_id": task_id,
        "queued_courses": len((payload or {}).get("courses", []))
    }




@app.get("/descriptive/location", response_model=List[CountryUniversities], tags=["Descriptive"], summary="Universities per Country from DB")
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
        occ_id   = occ.get("id")
        label    = occ.get("label")
        parent   = occ.get("occupation_group")
        sector   = occ.get("top_level_sector")

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
            cursor.close(); conn.close()

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


@app.get("/theme_search", tags=["Queries"])
def theme_search(
    theme: str = Query(..., description="Thematic keyword, e.g. 'biology' or 'computer science'"),
    threshold: int = Query(70, ge=0, le=100, description="Fuzzy matching threshold (0-100)")
):
    """
    Fuzzy semantic search based on:
      - course title
      - degree titles
      - skill names
      - occupation labels / parent_label / top_sector (linked to each skill)

    Returns a flat list of matches and a grouping by university.
    """
    if not is_database_connected(DB_CONFIG):
        raise HTTPException(status_code=500, detail="Database connection failed.")

    theme_norm = theme.strip().lower()
    if not theme_norm:
        raise HTTPException(status_code=400, detail="theme must not be empty")

    sql = f"""
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
        JOIN Skill s              ON cs.skill_id   = s.skill_id
        LEFT JOIN SkillOccupation so ON s.skill_id   = so.skill_id
        LEFT JOIN Occupation o    ON so.occupation_id = o.occupation_id
        JOIN Course c             ON cs.course_id  = c.course_id
        JOIN University u         ON c.university_id = u.university_id
        WHERE s.skill_name IS NOT NULL AND s.skill_name != ''
    """

    try:
        conn   = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql)
        rows = cursor.fetchall()
    except mysql.connector.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        try: cursor.close(); conn.close()
        except Exception: pass

    course_map = defaultdict(lambda: {
        "university": "",
        "degree_titles": set(),
        "skills": set(),
        "occ_labels": set(),
        "occ_parents": set(),
        "occ_sectors": set()
    })

    for r in rows:
        uni   = r["university_name"]
        title = r["lesson_name"]
        key   = (uni, title)

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

    flat_results = []
    grouped      = defaultdict(list)

    for (uni, title), meta in course_map.items():
        thematic_items = [title]
        thematic_items += list(meta["degree_titles"])
        thematic_items += list(meta["skills"])
        thematic_items += list(meta["occ_labels"])
        thematic_items += list(meta["occ_parents"])
        thematic_items += list(meta["occ_sectors"])

        thematic_str = " ".join(thematic_items).lower()
        score = fuzz.partial_ratio(theme_norm, thematic_str)

        if score >= threshold:
            flat_results.append({
                "university": uni,
                "course": title,
                "score": score
            })
            grouped[uni].append(title)

    flat_results.sort(key=lambda x: (-x["score"], x["university"], x["course"]))

    return {
        "theme_query": theme,
        "threshold": threshold,
        "matches": flat_results,
        "grouped_by_university": grouped
    }


@app.get("/exploratory/skills_location", response_model=List[SkillsByCountry], tags=["Exploratory"], summary="Skills per Country from DB")
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

@app.get("/trend/location", response_model=List[CountryTrend], tags=["Trend"], summary="University Join Trend per Country from DB")
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

@app.get("/cluster/universities/{k}", response_model=List[ClusterResult], tags=["Clustering"], summary="University Clustering by Skills from DB")
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

@app.get("/descriptive/skills_frequency", response_model=List[SkillFrequency], tags=["Descriptive"], summary="Skill frequency across courses from DB")
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

@app.get("/bilateral/biodiversity_analysis", tags=["Bilateral"], summary="Degree/Department biodiversity of Level-4 skills")
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

@app.get("/bilateral/labor_market_export", response_model=ExportResponse, tags=["Bilateral"], summary="List of all degrees with their Level 4 skills")
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
            lessons_data[key]["upload_date"] = row["created_at"].strftime("%Y-%m-%d") if row.get("created_at") else datetime.today().strftime("%Y-%m-%d")
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

        db_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name=f"pool_{task_id}", pool_size=workers, **DB_CONFIG)

        conn = db_pool.get_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT skill_id, skill_url FROM Skill WHERE esco_level IS NULL AND skill_url IS NOT NULL AND skill_url != ''")
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
                    logger.warning(f"[Task {task_id}] API call for URL '{url_value}' failed with status {resp.status_code}.")
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
                    logger.warning(f"[Task {task_id}] No level information found for skill_id={skill_id} (URL: {url_value}).")
       
                level_val_str = str(level_val)
            
            except (requests.exceptions.RequestException, json.JSONDecodeError, IndexError) as e:
                logger.error(f"[Task {task_id}] API/Network error for URL '{url_value}': {e}")
                return False

            try:
                db_conn = db_pool.get_connection()
                cursor = db_conn.cursor()

                logger.info(f"[Task {task_id}] ATTEMPTING DB WRITE: SET esco_level = '{level_val_str}' WHERE skill_id = {skill_id}")

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
            cur.close(); conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

