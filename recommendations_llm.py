# ==============================================================================
# LLM RECOMMENDATIONS
# Reads saved analysis results (short-term / long-term / policy) from the DB and
# can also fetch FTTI (Tsouk) trends live by title. Builds a compact evidence
# summary, asks the Mistral LLM for curriculum recommendations (Markdown), and
# CACHES the result in the DB keyed by the exact combination of titles (+ focus).
# Same combo -> cached, no new LLM call.
# ==============================================================================
import os
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

import requests
import mysql.connector
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, TIMESTAMP, text as sa_text
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

from config import DB_CONFIG
from llm_client import chat_generate

logger = logging.getLogger(__name__)

TRENDS_BASE_URL = os.getenv(
    "TRENDS_API_BASE_URL",
    "https://portal.skillab-project.eu/future-technology-trends-identifier"
)

router = APIRouter(prefix="/recommendations", tags=["LLM Recommendations"])


# ==========================================
# DB setup for caching recommendations
# ==========================================
_REC_DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','root')}"
    f"@{os.getenv('DB_HOST','mysql-curriculum-skill')}:{os.getenv('DB_PORT','3306')}"
    f"/{os.getenv('DB_NAME','skillcrawl')}"
)
_rec_engine = create_engine(_REC_DB_URL, echo=False, pool_pre_ping=True)
_RecSession = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=_rec_engine))
_RecBase = declarative_base()


class LLMRecommendation(_RecBase):
    __tablename__ = "llm_recommendations"
    id = Column(Integer, primary_key=True, index=True)
    combo_key = Column(String(64), nullable=False, unique=True, index=True)  # hash of titles+focus
    shortterm_title = Column(String(512), nullable=True)
    longterm_title = Column(String(512), nullable=True)
    policy_title = Column(String(512), nullable=True)
    tsouk_title = Column(String(512), nullable=True)
    focus = Column(String(1024), nullable=True)
    recommendations_md = Column(Text, nullable=True)   # the Markdown result
    created_at = Column(TIMESTAMP, server_default=sa_text("CURRENT_TIMESTAMP"))


def _ensure_rec_schema():
    try:
        _RecBase.metadata.create_all(bind=_rec_engine)
    except Exception as e:
        logger.error(f"llm_recommendations create_all failed: {e}")


def _combo_key(st: Optional[str], lt: Optional[str], pol: Optional[str],
               tsouk: Optional[str], focus: Optional[str]) -> str:
    raw = json.dumps({
        "st": st or "", "lt": lt or "", "pol": pol or "",
        "tsouk": tsouk or "", "focus": (focus or "").strip()
    }, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ==========================================
# DB READERS (raw SQL, no ORM dependency)
# ==========================================
def _conn():
    return mysql.connector.connect(**DB_CONFIG)


def _json_or_raw(v):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except Exception:
        return v


def _read_shortterm(title: str) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = _conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT title, description, analysis_date, country, university,
                   threshold, top_n, occupations,
                   skill_name, skill_id, gap_score, demand_score, supply_score,
                   in_curriculum, curriculum_courses
            FROM skill_gap_results
            WHERE title = %s
        """, (title,))
        rows = cur.fetchall() or []
        if not rows:
            return None
        first = rows[0]
        skills = [{
            "skill": r["skill_name"],
            "gap_score": r["gap_score"],
            "in_curriculum": bool(r["in_curriculum"]),
        } for r in rows]
        hot = sorted([s for s in skills if (s["gap_score"] or 0) > 0],
                     key=lambda x: x["gap_score"], reverse=True)
        oversupplied = sorted([s for s in skills if (s["gap_score"] or 0) < 0],
                              key=lambda x: x["gap_score"])
        return {
            "type": "short-term",
            "title": first["title"],
            "description": first["description"],
            "date": str(first["analysis_date"]) if first["analysis_date"] else None,
            "filters": {
                "country": first["country"], "university": first["university"],
                "threshold": first["threshold"], "top_n": first["top_n"],
                "occupations": _json_or_raw(first["occupations"]),
            },
            "total_skills": len(skills),
            "hot_skills": hot,
            "oversupplied_skills": oversupplied,
        }
    finally:
        if conn and conn.is_connected():
            conn.close()


def _read_longterm(title: str) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = _conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT title, source_title, description, analysis_date, country,
                   esco_threshold, skill_name, skill_id, technologies,
                   in_curriculum, curriculum_courses
            FROM title_gap_results
            WHERE title = %s
        """, (title,))
        rows = cur.fetchall() or []
        if not rows:
            return None
        first = rows[0]
        skills = [{
            "skill": r["skill_name"],
            "technologies": _json_or_raw(r["technologies"]),
            "in_curriculum": bool(r["in_curriculum"]),
        } for r in rows]
        covered = [s for s in skills if s["in_curriculum"]]
        missing = [s for s in skills if not s["in_curriculum"]]
        return {
            "type": "long-term",
            "title": first["title"],
            "source_title": first["source_title"],
            "description": first["description"],
            "date": str(first["analysis_date"]) if first["analysis_date"] else None,
            "filters": {"country": first["country"], "esco_threshold": first["esco_threshold"]},
            "total_skills": len(skills),
            "covered_skills": [s["skill"] for s in covered],
            "missing_skills": [s["skill"] for s in missing],
        }
    finally:
        if conn and conn.is_connected():
            conn.close()


def _read_policy(title: str) -> Optional[Dict[str, Any]]:
    conn = None
    try:
        conn = _conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT title, description, analysis_date, filter_country,
                   threshold, top_n, occupations,
                   university_name, country, coverage_score,
                   present_skills_count, missing_skills_count
            FROM policy_recommendations
            WHERE title = %s
            ORDER BY coverage_score DESC
        """, (title,))
        rows = cur.fetchall() or []
        if not rows:
            return None
        first = rows[0]
        universities = [{
            "university": r["university_name"],
            "country": r["country"],
            "coverage_score": r["coverage_score"],
            "present": r["present_skills_count"],
            "missing": r["missing_skills_count"],
        } for r in rows]
        agg: Dict[str, List[float]] = {}
        for u in universities:
            agg.setdefault(u["country"] or "Unknown", []).append(u["coverage_score"] or 0.0)
        countries = [{
            "country": c,
            "avg_coverage": round(sum(v) / len(v), 2) if v else 0.0,
            "universities_count": len(v),
        } for c, v in agg.items()]
        countries.sort(key=lambda x: x["avg_coverage"], reverse=True)
        return {
            "type": "policy",
            "title": first["title"],
            "description": first["description"],
            "date": str(first["analysis_date"]) if first["analysis_date"] else None,
            "filters": {
                "country": first["filter_country"],
                "threshold": first["threshold"], "top_n": first["top_n"],
                "occupations": _json_or_raw(first["occupations"]),
            },
            "universities": universities[:25],
            "countries": countries,
        }
    finally:
        if conn and conn.is_connected():
            conn.close()


def _read_tsouk_trends(title: str) -> Optional[Dict[str, Any]]:
    """
    Live fetch from the FTTI (Tsouk) API: GET /policies/by-title/{title}.
    Extracts the distinct skills + technologies for that trends analysis.
    No curriculum comparison — pure trends/skills evidence.
    """
    try:
        resp = requests.get(
            f"{TRENDS_BASE_URL}/policies/by-title/{requests.utils.quote(title)}",
            params={"include_content": "true"},
            timeout=60, verify=False
        )
        resp.raise_for_status()
        jobs = resp.json()
    except Exception as e:
        logger.error(f"FTTI fetch for '{title}' failed: {e}")
        return None

    if not isinstance(jobs, list) or not jobs:
        return None

    skills_by_tech: Dict[str, set] = {}
    all_skills: set = set()
    for job in jobs:
        content = job.get("content") or {}
        mapping = content.get("mapping_evidence") or {}
        for section in mapping.get("skills") or []:
            tech = section.get("technology", "") or "Unknown"
            skills_by_tech.setdefault(tech, set())
            for m in section.get("matches", []):
                label = (m.get("label") or "").strip()
                if label:
                    skills_by_tech[tech].add(label)
                    all_skills.add(label)

    if not all_skills:
        return None

    return {
        "type": "tsouk-trends",
        "title": title,
        "jobs_found": len(jobs),
        "technologies": sorted(skills_by_tech.keys()),
        "skills_by_technology": {t: sorted(sk) for t, sk in skills_by_tech.items()},
        "total_skills": len(all_skills),
    }


# ==========================================
# PROMPT BUILDING
# ==========================================
def _trim(items: List, n: int) -> List:
    return items[:n] if items else []


def _build_evidence(sources: List[Dict[str, Any]]) -> str:
    parts = []
    for s in sources:
        if s["type"] == "short-term":
            hot = [x["skill"] for x in _trim(s["hot_skills"], 15)]
            over = [x["skill"] for x in _trim(s["oversupplied_skills"], 15)]
            missing_hot = [x["skill"] for x in s["hot_skills"] if not x["in_curriculum"]][:15]
            parts.append(
                f"[SHORT-TERM] title='{s['title']}' country={s['filters'].get('country')} "
                f"occupations={s['filters'].get('occupations')}\n"
                f"  hot_skills (high demand vs supply): {hot}\n"
                f"  oversupplied_skills: {over}\n"
                f"  hot_skills NOT in curriculum: {missing_hot}"
            )
        elif s["type"] == "long-term":
            parts.append(
                f"[LONG-TERM] title='{s['title']}' source='{s.get('source_title')}' "
                f"country={s['filters'].get('country')}\n"
                f"  future skills covered by curricula: {_trim(s['covered_skills'], 20)}\n"
                f"  future skills MISSING from curricula: {_trim(s['missing_skills'], 20)}"
            )
        elif s["type"] == "policy":
            top_countries = [f"{c['country']} ({c['avg_coverage']}%)" for c in _trim(s["countries"], 10)]
            low_unis = sorted(s["universities"], key=lambda x: x["coverage_score"] or 0)[:10]
            low = [f"{u['university']} {u['coverage_score']}%" for u in low_unis]
            parts.append(
                f"[POLICY] title='{s['title']}' occupations={s['filters'].get('occupations')}\n"
                f"  coverage by country (avg): {top_countries}\n"
                f"  lowest-coverage universities: {low}"
            )
        elif s["type"] == "tsouk-trends":
            tech_lines = []
            for tech, skills in list(s["skills_by_technology"].items())[:12]:
                tech_lines.append(f"    - {tech}: {skills[:15]}")
            tech_block = "\n".join(tech_lines)
            parts.append(
                f"[FUTURE TRENDS (FTTI)] title='{s['title']}' "
                f"technologies={s['technologies']}\n"
                f"  future-relevant skills by technology:\n{tech_block}"
            )
    return "\n\n".join(parts)


_SYSTEM_INSTRUCTIONS = (
    "You are an education-policy analyst for the SKILLAB project. You are given the "
    "results of skill-gap analyses comparing labour-market demand and future technology "
    "trends against university curricula. Produce concrete, actionable recommendations "
    "for curriculum development. Base every recommendation ONLY on the evidence provided; "
    "do not invent skills, universities, or numbers. Be specific and concise."
)


def _build_prompt(evidence: str, focus: Optional[str]) -> str:
    focus_line = f"\nParticular focus requested: {focus}\n" if focus else ""
    return f"""{_SYSTEM_INSTRUCTIONS}

EVIDENCE FROM SAVED ANALYSES:
{evidence}
{focus_line}
Write the recommendations as Markdown with these sections:
1. **Summary** — 2-3 sentences on the overall picture.
2. **Priority skills to add** — skills in demand / future-relevant but missing from curricula, with a one-line justification each.
3. **Skills to de-emphasise** — oversupplied skills, if any.
4. **University / country actions** — where coverage is weakest and what to do.
5. **Concrete next steps** — 3-5 bullet actions.

Keep it grounded strictly in the evidence above. Output valid Markdown only."""


# ==========================================
# SHARED COLLECTOR
# ==========================================
def _collect_sources(st: Optional[str], lt: Optional[str], pol: Optional[str], tsouk: Optional[str]):
    sources: List[Dict[str, Any]] = []
    not_found: List[str] = []
    if st:
        s = _read_shortterm(st)
        (sources.append(s) if s else not_found.append(f"short-term '{st}'"))
    if lt:
        s = _read_longterm(lt)
        (sources.append(s) if s else not_found.append(f"long-term '{lt}'"))
    if pol:
        s = _read_policy(pol)
        (sources.append(s) if s else not_found.append(f"policy '{pol}'"))
    if tsouk:
        s = _read_tsouk_trends(tsouk)
        (sources.append(s) if s else not_found.append(f"FTTI trends '{tsouk}'"))
    return sources, not_found


# ==========================================
# REQUEST SCHEMA
# ==========================================
class RecommendRequest(BaseModel):
    shortterm_title: Optional[str] = Field(None, description="Title of a saved short-term analysis.")
    longterm_title: Optional[str] = Field(None, description="Title of a saved long-term analysis.")
    policy_title: Optional[str] = Field(None, description="Title of a saved policy analysis.")
    tsouk_title: Optional[str] = Field(None, description="FTTI trends analysis title (fetched live from the Tsouk API, no curricula).")
    focus: Optional[str] = Field(None, description="Optional extra instruction (e.g. 'focus on Greece').")
    force_refresh: bool = Field(False, description="Ignore cache and regenerate with a new LLM call.")


# ==========================================
# ENDPOINTS
# ==========================================
@router.post("/generate", summary="Generate (or return cached) LLM recommendations from saved analyses / FTTI trends")
def generate_recommendations(req: RecommendRequest):
    """
    Give one or more titles: short-term / long-term / policy (read from the DB),
    and/or an FTTI trends title (fetched live from the Tsouk API). The evidence is
    summarised and sent to the Mistral LLM, which returns curriculum recommendations
    in Markdown. The Markdown is CACHED in the DB keyed by the exact title
    combination (+ focus): the same combo returns the cached Markdown with NO new
    LLM call. Set force_refresh=true to regenerate.
    """
    _ensure_rec_schema()

    if not (req.shortterm_title or req.longterm_title or req.policy_title or req.tsouk_title):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: shortterm_title, longterm_title, policy_title, tsouk_title."
        )

    key = _combo_key(req.shortterm_title, req.longterm_title, req.policy_title, req.tsouk_title, req.focus)

    # 1) Cache hit?
    if not req.force_refresh:
        db = _RecSession()
        try:
            cached = db.query(LLMRecommendation).filter(LLMRecommendation.combo_key == key).first()
            if cached:
                return {
                    "cached": True,
                    "used_analyses": {
                        "shortterm_title": cached.shortterm_title,
                        "longterm_title": cached.longterm_title,
                        "policy_title": cached.policy_title,
                        "tsouk_title": cached.tsouk_title,
                    },
                    "focus": cached.focus,
                    "created_at": cached.created_at,
                    "recommendations": cached.recommendations_md,
                }
        finally:
            db.close()

    # 2) Collect evidence
    sources, not_found = _collect_sources(
        req.shortterm_title, req.longterm_title, req.policy_title, req.tsouk_title
    )
    if not sources:
        raise HTTPException(status_code=404, detail=f"No saved analyses found for: {', '.join(not_found)}.")

    evidence = _build_evidence(sources)
    prompt = _build_prompt(evidence, req.focus)

    # 3) LLM call
    try:
        recommendations_md = chat_generate(prompt, temperature=0.2)
    except Exception as e:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

    # 4) Save/refresh cache
    db = _RecSession()
    try:
        existing = db.query(LLMRecommendation).filter(LLMRecommendation.combo_key == key).first()
        if existing:
            existing.recommendations_md = recommendations_md
            existing.shortterm_title = req.shortterm_title
            existing.longterm_title = req.longterm_title
            existing.policy_title = req.policy_title
            existing.tsouk_title = req.tsouk_title
            existing.focus = req.focus
        else:
            db.add(LLMRecommendation(
                combo_key=key,
                shortterm_title=req.shortterm_title,
                longterm_title=req.longterm_title,
                policy_title=req.policy_title,
                tsouk_title=req.tsouk_title,
                focus=req.focus,
                recommendations_md=recommendations_md,
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to cache recommendation: {e}")
    finally:
        db.close()

    return {
        "cached": False,
        "used_analyses": [{"type": s["type"], "title": s["title"]} for s in sources],
        "not_found": not_found,
        "focus": req.focus,
        "recommendations": recommendations_md,
    }


@router.get("/list", summary="List all cached recommendations")
def list_recommendations():
    """List saved recommendations with their title combinations and dates."""
    _ensure_rec_schema()
    db = _RecSession()
    try:
        rows = db.query(LLMRecommendation).order_by(LLMRecommendation.created_at.desc()).all()
        return {"recommendations": [{
            "id": r.id,
            "shortterm_title": r.shortterm_title,
            "longterm_title": r.longterm_title,
            "policy_title": r.policy_title,
            "tsouk_title": r.tsouk_title,
            "focus": r.focus,
            "created_at": r.created_at,
        } for r in rows]}
    finally:
        db.close()


@router.get("/preview", summary="Preview the evidence that would be sent to the LLM (no LLM call)")
def preview_evidence(
    shortterm_title: Optional[str] = Query(None),
    longterm_title: Optional[str] = Query(None),
    policy_title: Optional[str] = Query(None),
    tsouk_title: Optional[str] = Query(None),
):
    """Debug helper: shows the evidence block without calling the LLM."""
    sources, not_found = _collect_sources(shortterm_title, longterm_title, policy_title, tsouk_title)
    if not sources:
        raise HTTPException(status_code=404, detail=f"No saved analyses found for: {', '.join(not_found)}.")
    return {
        "used_analyses": [{"type": s["type"], "title": s["title"]} for s in sources],
        "not_found": not_found,
        "evidence": _build_evidence(sources),
    }