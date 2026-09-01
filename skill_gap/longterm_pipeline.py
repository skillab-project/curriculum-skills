# ==============================================================================
# TITLE-BASED GAP (policies/by-title → skills with ESCO url → vs curricula → SAVE)
# ==============================================================================
import os
import logging
from typing import Any, Dict, List, Optional

import requests
import mysql.connector
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import uuid
from datetime import date
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, TIMESTAMP, Date, text as sa_text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

from config import DB_CONFIG

logger = logging.getLogger(__name__)

full_pipeline_router = APIRouter(prefix="/longterm", tags=["Long-Term Gap by Title"])

TRENDS_BASE_URL = os.getenv(
    "TRENDS_API_BASE_URL",
    "https://portal.skillab-project.eu/future-technology-trends-identifier"
)

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
    title = Column(String(512), nullable=False, index=True)      # unique save key
    source_title = Column(String(512), nullable=True, index=True) # FTTI analysis title used as source
    description = Column(String(2048), nullable=True)
    analysis_date = Column(Date, nullable=True)
    country = Column(String(255), nullable=True)                # analysis filter
    skill_name = Column(String(512), nullable=True)
    skill_id = Column(String(512), nullable=True, index=True)   # ESCO url
    technologies = Column(JSON, nullable=True)
    job_ids = Column(JSON, nullable=True)
    in_curriculum = Column(Boolean, nullable=True)
    curriculum_courses = Column(JSON, nullable=True)
    esco_threshold = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, server_default=sa_text("CURRENT_TIMESTAMP"))


# ==========================================
# SCHEMA SELF-MIGRATION
# ==========================================
_TITLE_GAP_COLUMNS = {
    "source_title": "VARCHAR(512) NULL",
}


def _ensure_title_schema():
    """Create the table if missing, then add any columns the model has that the
    existing table lacks (e.g. source_title on older DBs). Never raises."""
    try:
        _TitleBase.metadata.create_all(bind=_title_engine)
    except Exception as e:
        logger.error(f"title_gap create_all failed: {e}")
        return
    try:
        with _title_engine.begin() as conn:
            existing = {
                row[0]
                for row in conn.execute(sa_text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() "
                    "AND table_name = 'title_gap_results'"
                ))
            }
            if not existing:
                return
            for col, ddl in _TITLE_GAP_COLUMNS.items():
                if col not in existing:
                    logger.warning("Adding missing column title_gap_results.%s", col)
                    conn.execute(sa_text(f"ALTER TABLE title_gap_results ADD COLUMN {col} {ddl}"))
                    try:
                        conn.execute(sa_text(
                            f"CREATE INDEX idx_title_gap_{col} ON title_gap_results ({col})"
                        ))
                    except Exception:
                        pass
    except Exception as e:
        logger.error(f"title_gap column migration failed: {e}")


# --- Tsouk policies/by-title fetch ---
def _fetch_policies_by_title(title: str) -> List[Dict[str, Any]]:
    """GET /policies/by-title/{title}?include_content=true"""
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
    """Distinct skill pool keyed by ESCO url (id) across all jobs."""
    pool: Dict[str, Dict[str, Any]] = {}
    for job in jobs:
        job_id = job.get("job_id")
        content = job.get("content") or {}
        mapping = content.get("mapping_evidence") or {}
        for section in mapping.get("skills") or []:
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
    for v in pool.values():
        v["technologies"] = sorted(v["technologies"])
        v["job_ids"] = sorted(v["job_ids"])
    return pool


def _check_skill_urls_in_curriculum(skill_urls: List[str], country: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    URL-based curriculum check. If country is given, only universities of that
    country are considered (curriculum coverage is scoped to that country).
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
            params = list(batch)
            country_clause = ""
            if country and country.strip():
                country_clause = " AND LOWER(u.country) LIKE LOWER(%s)"
                params.append(f"%{country.strip()}%")
            cursor.execute(f"""
                SELECT s.skill_url, c.lesson_name, u.university_name, u.country
                FROM Skill s
                JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                JOIN Course c ON cs.course_id = c.course_id
                JOIN University u ON c.university_id = u.university_id
                WHERE s.skill_url IN ({placeholders}){country_clause}
                LIMIT 3000
            """, params)
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


def _title_exists(title: str) -> bool:
    """True if an analysis with this exact title is already saved."""
    db = _TitleSession()
    try:
        return db.query(TitleGapResult).filter(TitleGapResult.title == title).first() is not None
    finally:
        db.close()


class TitleGapRequest(BaseModel):
    title: str                                  # unique save key for this long-term analysis
    source_title: Optional[str] = None          # FTTI analysis title to pull skills from
    description: Optional[str] = None
    day: Optional[int] = None
    month: Optional[int] = None
    year: Optional[int] = None
    country: Optional[str] = None
    esco_threshold: float = 0.4


@full_pipeline_router.post(
    "/gap-by-title",
    summary="Title → policy jobs → skills (ESCO url) → gap vs curricula DB → SAVE"
)
def gap_by_title(req: TitleGapRequest):
    """
    Start a title-gap analysis. `title` is the unique save key: if one already
    exists, the request is rejected. `source_title` is the Future Technology
    Trends analysis whose policy jobs supply the skills — it defaults to `title`
    for backward compatibility. This lets several long-term analyses (e.g. one
    per country) be saved for the SAME source analysis under distinct titles.
    Skills come from all policy jobs of the source (union, distinct by ESCO url);
    curriculum coverage is scoped to `country` if provided; results are saved
    under a run_id.
    """
    _ensure_title_schema()

    source_title = req.source_title or req.title

    if _title_exists(req.title):
        raise HTTPException(
            status_code=409,
            detail=f"An analysis with title '{req.title}' already exists. Use a different title."
        )

    analysis_date = None
    if req.year and req.month and req.day:
        try:
            analysis_date = date(req.year, req.month, req.day)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date (day/month/year).")

    jobs = _fetch_policies_by_title(source_title)
    if not jobs:
        raise HTTPException(status_code=404, detail=f"No policy jobs found for source title '{source_title}'.")

    pool = _extract_skills_with_url_from_policy_jobs(jobs, threshold=req.esco_threshold)
    if not pool:
        raise HTTPException(status_code=400, detail="No skills with ESCO url found for this title.")

    curric = _check_skill_urls_in_curriculum(list(pool.keys()), country=req.country)

    run_id = str(uuid.uuid4())
    db = _TitleSession()
    try:
        for url, info in pool.items():
            c = curric.get(url, {"in_curriculum": False, "courses": []})
            db.add(TitleGapResult(
                run_id=run_id,
                title=req.title,
                source_title=source_title,
                description=req.description,
                analysis_date=analysis_date,
                country=req.country,
                skill_name=info["skill_name"],
                skill_id=url,
                technologies=info["technologies"],
                job_ids=info["job_ids"],
                in_curriculum=c["in_curriculum"],
                curriculum_courses=c["courses"],
                esco_threshold=req.esco_threshold,
            ))
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
        "source_title": source_title,
        "description": req.description,
        "date": analysis_date.isoformat() if analysis_date else None,
        "country": req.country,
        "jobs_found": len(jobs),
        "total_unique_skills": len(skills_out),
        "covered_count": covered,
        "missing_count": len(skills_out) - covered,
        "coverage_pct": round(covered / len(skills_out) * 100, 2) if skills_out else 0.0,
        "skills": skills_out,
    }


@full_pipeline_router.get(
    "/gap-by-title/runs",
    summary="List all title-gap analyses with their filters"
)
def gap_by_title_runs():
    """
    List every saved analysis with its title, description, date, and filters
    (country, esco_threshold), plus how many skills it produced.
    """
    _ensure_title_schema()
    db = _TitleSession()
    try:
        rows = db.query(TitleGapResult).order_by(TitleGapResult.created_at.desc()).all()
        seen: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            if r.run_id not in seen:
                seen[r.run_id] = {
                    "run_id": r.run_id,
                    "title": r.title,
                    "source_title": r.source_title,
                    "description": r.description,
                    "date": r.analysis_date.isoformat() if r.analysis_date else None,
                    "created_at": r.created_at,
                    "filters": {
                        "country": r.country,
                        "esco_threshold": r.esco_threshold,
                    },
                    "skills_count": 0,
                }
            seen[r.run_id]["skills_count"] += 1
        return {"runs": list(seen.values())}
    finally:
        db.close()


@full_pipeline_router.get(
    "/gap-by-title/results",
    summary="Read saved title-gap results by title"
)
def gap_by_title_results(
    title: str = Query(..., description="Title of the analysis to fetch results for"),
    in_curriculum: Optional[bool] = Query(None),
):
    """
    Read back the saved results of an analysis, identified by its title.
    Optionally filter the skills by in_curriculum.
    """
    _ensure_title_schema()
    db = _TitleSession()
    try:
        q = db.query(TitleGapResult).filter(TitleGapResult.title == title)
        if in_curriculum is not None:
            q = q.filter(TitleGapResult.in_curriculum == in_curriculum)
        rows = q.all()
        if not rows:
            return {"message": f"No results found for title '{title}'.", "data": []}

        first = rows[0]
        skills = [{
            "skill_name": r.skill_name,
            "skill_id": r.skill_id,
            "technologies": r.technologies,
            "job_ids": r.job_ids,
            "in_curriculum": r.in_curriculum,
            "curriculum_courses": r.curriculum_courses,
        } for r in rows]
        covered = sum(1 for s in skills if s["in_curriculum"])

        return {
            "run_id": first.run_id,
            "title": first.title,
            "source_title": first.source_title,
            "description": first.description,
            "date": first.analysis_date.isoformat() if first.analysis_date else None,
            "filters": {"country": first.country, "esco_threshold": first.esco_threshold},
            "total_skills": len(skills),
            "covered_count": covered,
            "missing_count": len(skills) - covered,
            "coverage_pct": round(covered / len(skills) * 100, 2) if skills else 0.0,
            "skills": skills,
        }
    finally:
        db.close()