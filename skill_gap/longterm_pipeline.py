# ==============================================================================
# TITLE-BASED GAP (policies/by-title → skills with ESCO url → vs curricula → SAVE)
# ==============================================================================
import os
import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List

import requests
import mysql.connector
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import uuid
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, TIMESTAMP, text as sa_text
# Σημείωση: Αν χρησιμοποιείτε παλαιότερη έκδοση της SQLAlchemy, ίσως χρειαστεί το:
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# --- Εισαγωγή ρυθμίσεων από το config.py του project σας ---
from config import DB_CONFIG

# --- Αρχικοποίηση Logger ---
logger = logging.getLogger(__name__)

# --- Αρχικοποίηση των Routers (Απαραίτητα για το main.py / __init__.py) ---
curriculum_router = APIRouter()
full_pipeline_router = APIRouter()

# --- Ορισμός του TRENDS_BASE_URL (π.χ. από περιβαλλοντικές μεταβλητές) ---
TRENDS_BASE_URL = os.getenv("TRENDS_API_BASE_URL", "https://portal.skillab-project.eu/future-technology-trends-identifier")

# --- DB setup for saving title gap results ---
_TITLE_DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','root')}"
    f"@{os.getenv('DB_HOST','mysql-curriculum-skill')}:{os.getenv('DB_PORT','3306')}"
    f"/{os.getenv('DB_NAME','skillcrawl')}"
)
_title_engine = create_engine(_TITLE_DB_URL, echo=False, pool_pre_ping=True)
_TitleSession = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=_title_engine))
_TitleBase = declarative_base()


class TitleGapResult(_TitleBase):
    __tablename__ = "title_gap_results"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), nullable=False, index=True)
    title = Column(String(512), nullable=False, index=True)
    skill_name = Column(String(512), nullable=True)
    skill_id = Column(String(512), nullable=True, index=True)   # ESCO url
    technologies = Column(JSON, nullable=True)                  # which technologies asked for this skill
    job_ids = Column(JSON, nullable=True)                       # which jobs contained this skill
    in_curriculum = Column(Boolean, nullable=True)              # taught anywhere in DB
    curriculum_courses = Column(JSON, nullable=True)            # in which courses
    esco_threshold = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, server_default=sa_text("CURRENT_TIMESTAMP"))

# --- Tsouk policies/by-title fetch ---
def _fetch_policies_by_title(title: str) -> List[Dict[str, Any]]:
    """
    GET /policies/by-title/{title}?include_content=true
    Returns the list of policy jobs for that title.
    """
    try:
        resp = requests.get(
            f"{TRENDS_BASE_URL}/policies/by-title/{requests.utils.quote(title)}",
            params={"include_content": "true"},
            timeout=60, verify=False
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"policies/by-title for '{title}' failed: {e}")
    return data if isinstance(data, list) else []


def _extract_skills_with_url_from_policy_jobs(jobs: List[Dict[str, Any]], threshold: float) -> Dict[str, Dict[str, Any]]:
    """
    From the list of policy jobs, build a DISTINCT skill pool keyed by ESCO url (id).
    Walks content.mapping_evidence.skills[].matches[] and keeps id/label/technology/job_id.

    Returns: { skill_url: {"skill_id": url, "skill_name": label,
                           "technologies": [...], "job_ids": [...]} }
    """
    pool: Dict[str, Dict[str, Any]] = {}

    for job in jobs:
        job_id = job.get("job_id")
        content = job.get("content") or {}
        mapping = content.get("mapping_evidence") or {}
        skills_sections = mapping.get("skills") or []

        for section in skills_sections:
            tech = section.get("technology", "")
            for m in section.get("matches", []):
                url = m.get("id")
                label = m.get("label")
                score = m.get("score", 0.0)
                if not url or score < threshold:
                    continue
                url = url.strip()
                if url not in pool:
                    pool[url] = {
                        "skill_id": url,
                        "skill_name": (label or "").strip(),
                        "technologies": set(),
                        "job_ids": set(),
                    }
                if tech:
                    pool[url]["technologies"].add(tech)
                if job_id:
                    pool[url]["job_ids"].add(job_id)

    # convert sets to sorted lists for JSON
    for v in pool.values():
        v["technologies"] = sorted(v["technologies"])
        v["job_ids"] = sorted(v["job_ids"])

    return pool


def _check_skill_urls_in_curriculum(skill_urls: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    URL-based curriculum check (like policy/skill_gap): match ESCO url against
    Skill.skill_url. Returns { url: {"in_curriculum": bool, "courses": [...]} }.
    """
    result = {u: {"in_curriculum": False, "courses": []} for u in skill_urls if u}
    if not skill_urls:
        return result

    conn = None
    BATCH = 50
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        ids = [u for u in skill_urls if u]
        for i in range(0, len(ids), BATCH):
            batch = ids[i:i + BATCH]
            placeholders = ", ".join(["%s"] * len(batch))
            cursor.execute(f"""
                SELECT s.skill_url, c.lesson_name, u.university_name, u.country
                FROM Skill s
                JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                JOIN Course c ON cs.course_id = c.course_id
                JOIN University u ON c.university_id = u.university_id
                WHERE s.skill_url IN ({placeholders})
                LIMIT 3000
            """, batch)
            for r in cursor.fetchall():
                u = r["skill_url"].strip() if r.get("skill_url") else None
                if not u or u not in result:
                    continue
                result[u]["in_curriculum"] = True
                entry = f"{r['lesson_name']} ({r['university_name']}) - [{r['country']}]"
                if entry not in result[u]["courses"]:
                    result[u]["courses"].append(entry)
    except Exception as e:
        logger.error(f"DB error in _check_skill_urls_in_curriculum: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return result


class TitleGapRequest(BaseModel):
    title: str
    esco_threshold: float = 0.4


@full_pipeline_router.post(
    "/gap-by-title",
    summary="Title → policy jobs → skills (ESCO url) → gap vs curricula DB → SAVE"
)
def gap_by_title(req: TitleGapRequest):
    """
    1. GET /policies/by-title/{title} → list of policy jobs.
    2. Union of all jobs' skills (mapping_evidence.skills), distinct by ESCO url.
    3. URL-based match against the universities DB (taught? in which courses?).
    4. Persist per skill under a run_id, keyed by title.
    """
    jobs = _fetch_policies_by_title(req.title)
    if not jobs:
        raise HTTPException(status_code=404, detail=f"No policy jobs found for title '{req.title}'.")

    pool = _extract_skills_with_url_from_policy_jobs(jobs, threshold=req.esco_threshold)
    if not pool:
        raise HTTPException(status_code=400, detail="No skills with ESCO url found for this title.")

    curric = _check_skill_urls_in_curriculum(list(pool.keys()))

    run_id = str(uuid.uuid4())
    _TitleBase.metadata.create_all(bind=_title_engine)

    db = _TitleSession()
    saved = 0
    try:
        for url, info in pool.items():
            c = curric.get(url, {"in_curriculum": False, "courses": []})
            db.add(TitleGapResult(
                run_id=run_id,
                title=req.title,
                skill_name=info["skill_name"],
                skill_id=url,
                technologies=info["technologies"],
                job_ids=info["job_ids"],
                in_curriculum=c["in_curriculum"],
                curriculum_courses=c["courses"],
                esco_threshold=req.esco_threshold,
            ))
            saved += 1
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB save error: {e}")
    finally:
        db.close()

    skills_out = []
    for url, info in pool.items():
        c = curric.get(url, {"in_curriculum": False, "courses": []})
        skills_out.append({
            "skill_name": info["skill_name"],
            "skill_id": url,
            "technologies": info["technologies"],
            "job_ids": info["job_ids"],
            "in_curriculum": c["in_curriculum"],
            "curriculum_courses": c["courses"],
        })

    covered = sum(1 for s in skills_out if s["in_curriculum"])
    return {
        "run_id": run_id,
        "title": req.title,
        "jobs_found": len(jobs),
        "total_unique_skills": len(skills_out),
        "covered_count": covered,
        "missing_count": len(skills_out) - covered,
        "coverage_pct": round(covered / len(skills_out) * 100, 2) if skills_out else 0.0,
        "skills": skills_out,
    }


@full_pipeline_router.get(
    "/gap-by-title/results",
    summary="Read saved title-gap results"
)
def gap_by_title_results(
    run_id: str = None,
    title: str = None,
    in_curriculum: bool = None,
):
    """Read back saved title-gap results, filtered by run_id / title / in_curriculum."""
    db = _TitleSession()
    try:
        q = db.query(TitleGapResult)
        if run_id:
            q = q.filter(TitleGapResult.run_id == run_id)
        if title:
            q = q.filter(TitleGapResult.title.ilike(f"%{title}%"))
        if in_curriculum is not None:
            q = q.filter(TitleGapResult.in_curriculum == in_curriculum)
        rows = q.all()
        if not rows:
            return {"message": "No results found.", "data": []}
        return rows
    finally:
        db.close()


@full_pipeline_router.get(
    "/gap-by-title/runs",
    summary="List all title-gap runs"
)
def gap_by_title_runs():
    """List all title-gap analysis runs with their title and skill counts."""
    db = _TitleSession()
    try:
        rows = db.query(TitleGapResult).order_by(TitleGapResult.created_at.desc()).all()
        seen = {}
        for r in rows:
            if r.run_id not in seen:
                seen[r.run_id] = {
                    "run_id": r.run_id,
                    "title": r.title,
                    "created_at": r.created_at,
                    "skills_count": 0,
                }
            seen[r.run_id]["skills_count"] += 1
        return {"runs": list(seen.values())}
    finally:
        db.close()