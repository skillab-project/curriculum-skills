# ==============================================================================
# LLM RECOMMENDATIONS
# Background generation: /generate starts a task and returns immediately. The
# LLM runs in a background thread (so nginx/gateway timeouts don't kill it).
# Status is tracked in the DB (running / completed / failed). Asking for the same
# combo while it runs returns "running"; once done it is stored and served from
# cache. Keyed by titles + focus + suggest_universities + policy_country.
# ==============================================================================
import os
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional

import requests
import mysql.connector
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
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
    combo_key = Column(String(64), nullable=False, unique=True, index=True)  # hash of titles+focus+suggest+country
    status = Column(String(20), nullable=True, index=True)   # running | completed | failed
    error = Column(String(2048), nullable=True)
    shortterm_title = Column(String(512), nullable=True)
    longterm_title = Column(String(512), nullable=True)
    policy_title = Column(String(512), nullable=True)
    tsouk_title = Column(String(512), nullable=True)
    policy_country = Column(String(255), nullable=True)
    focus = Column(String(1024), nullable=True)
    recommendations_md = Column(Text, nullable=True)   # the Markdown result
    created_at = Column(TIMESTAMP, server_default=sa_text("CURRENT_TIMESTAMP"))


# ---- schema self-migration (adds status/error/policy_country on old DBs) ----
_REC_COLUMNS = {
    "status": "VARCHAR(20) NULL",
    "error": "VARCHAR(2048) NULL",
    "policy_country": "VARCHAR(255) NULL",
}


def _ensure_rec_schema():
    try:
        _RecBase.metadata.create_all(bind=_rec_engine)
    except Exception as e:
        logger.error(f"llm_recommendations create_all failed: {e}")
        return
    try:
        with _rec_engine.begin() as conn:
            existing = {
                row[0]
                for row in conn.execute(sa_text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() "
                    "AND table_name = 'llm_recommendations'"
                ))
            }
            if not existing:
                return
            for col, ddl in _REC_COLUMNS.items():
                if col not in existing:
                    logger.warning("Adding missing column llm_recommendations.%s", col)
                    conn.execute(sa_text(f"ALTER TABLE llm_recommendations ADD COLUMN {col} {ddl}"))
                    if col == "status":
                        try:
                            conn.execute(sa_text(
                                "CREATE INDEX idx_rec_status ON llm_recommendations (status)"
                            ))
                        except Exception:
                            pass
    except Exception as e:
        logger.error(f"llm_recommendations column migration failed: {e}")


def _combo_key(st: Optional[str], lt: Optional[str], pol: Optional[str],
               tsouk: Optional[str], focus: Optional[str],
               suggest_unis: bool = False, policy_country: Optional[str] = None) -> str:
    raw = json.dumps({
        "st": st or "", "lt": lt or "", "pol": pol or "",
        "tsouk": tsouk or "", "focus": (focus or "").strip(),
        "suggest_unis": bool(suggest_unis),
        "policy_country": (policy_country or "").strip(),
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
            "skill_id": r["skill_id"],
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
            "missing_urls": [s["skill_id"] for s in hot if not s["in_curriculum"] and s["skill_id"]],
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
            "skill_id": r["skill_id"],
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
            "missing_urls": [s["skill_id"] for s in missing if s["skill_id"]],
        }
    finally:
        if conn and conn.is_connected():
            conn.close()


def _read_policy(title: str, country_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Read a policy analysis by title. Country scope: country_override (request) if
    given, else stored filter_country. Once a country is in effect, ONLY that
    country's universities are kept — even if empty (no leaking of other countries).
    """
    conn = None
    try:
        conn = _conn()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT title, description, analysis_date, filter_country,
                   threshold, top_n, occupations,
                   university_name, country, coverage_score,
                   present_skills_count, missing_skills_count,
                   missing_courses
            FROM policy_recommendations
            WHERE title = %s
            ORDER BY coverage_score DESC
        """, (title,))
        rows = cur.fetchall() or []
        if not rows:
            return None
        first = rows[0]

        effective_country = (country_override or first["filter_country"] or "").strip()

        if effective_country:
            fc = effective_country.lower()
            rows = [r for r in rows if fc in (r["country"] or "").strip().lower()]

        if not rows:
            return {
                "type": "policy",
                "title": first["title"],
                "description": first["description"],
                "date": str(first["analysis_date"]) if first["analysis_date"] else None,
                "filters": {
                    "country": effective_country or None,
                    "threshold": first["threshold"], "top_n": first["top_n"],
                    "occupations": _json_or_raw(first["occupations"]),
                },
                "universities": [],
                "countries": [],
                "empty_reason": f"No universities found for country '{effective_country}' in this analysis.",
            }

        universities = [{
            "university": r["university_name"],
            "country": r["country"],
            "coverage_score": r["coverage_score"],
            "present": r["present_skills_count"],
            "missing": r["missing_skills_count"],
            "missing_courses": _json_or_raw(r["missing_courses"]),
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
                "country": effective_country or None,
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
    """Live fetch from the FTTI (Tsouk) API: GET /policies/by-title/{title}."""
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


def _universities_teaching_skills(skill_urls: List[str],
                                  exclude_country: Optional[str] = None,
                                  limit_per_skill: int = 5) -> Dict[str, List[str]]:
    """For ESCO skill urls, find which universities teach them; optionally exclude a country."""
    result: Dict[str, List[str]] = {}
    if not skill_urls:
        return result

    conn = None
    BATCH = 50
    try:
        conn = _conn()
        cur = conn.cursor(dictionary=True)
        ids = [u for u in skill_urls if u]
        for i in range(0, len(ids), BATCH):
            batch = ids[i:i + BATCH]
            placeholders = ", ".join(["%s"] * len(batch))
            params = list(batch)
            country_clause = ""
            if exclude_country and exclude_country.strip():
                country_clause = " AND LOWER(u.country) NOT LIKE LOWER(%s)"
                params.append(f"%{exclude_country.strip()}%")
            cur.execute(f"""
                SELECT s.skill_url, s.skill_name, c.lesson_name,
                       u.university_name, u.country
                FROM Skill s
                JOIN CourseSkill cs ON s.skill_id = cs.skill_id
                JOIN Course c ON cs.course_id = c.course_id
                JOIN University u ON c.university_id = u.university_id
                WHERE s.skill_url IN ({placeholders}){country_clause}
                LIMIT 3000
            """, params)
            for r in cur.fetchall():
                su = r["skill_url"].strip() if r.get("skill_url") else None
                if not su:
                    continue
                entry = f"{r['lesson_name']} ({r['university_name']}) - [{r['country']}]"
                lst = result.setdefault(su, [])
                if entry not in lst and len(lst) < limit_per_skill:
                    lst.append(entry)
    except Exception as e:
        logger.error(f"DB error in _universities_teaching_skills: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()
    return result


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
            country = s["filters"].get("country")
            if not s.get("universities"):
                parts.append(
                    f"[POLICY] title='{s['title']}' country={country}\n"
                    f"  {s.get('empty_reason', 'No universities for this country in the analysis.')}"
                )
                continue
            top_countries = [f"{c['country']} ({c['avg_coverage']}%)" for c in _trim(s["countries"], 5)]
            low_unis = sorted(s["universities"], key=lambda x: x["coverage_score"] or 0)[:5]
            low = [f"{u['university']} {u['coverage_score']}%" for u in low_unis]

            suggest_lines = []
            for u in low_unis[:3]:
                mc = u.get("missing_courses") or {}
                if isinstance(mc, dict) and mc:
                    for skill, courses in list(mc.items())[:3]:
                        clist = courses[:2] if isinstance(courses, list) else []
                        if clist:
                            suggest_lines.append(f"    - '{skill}' taught at: {clist}")
            suggest_block = ("\n  gap skills taught at OTHER (model) universities:\n" +
                             "\n".join(suggest_lines)) if suggest_lines else ""

            parts.append(
                f"[POLICY] TARGET_COUNTRY={country} title='{s['title']}' "
                f"occupations={s['filters'].get('occupations')}\n"
                f"  coverage in TARGET country (avg): {top_countries}\n"
                f"  lowest-coverage universities IN {country}: {low}"
                f"{suggest_block}"
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


def _build_university_suggestions(sources: List[Dict[str, Any]]) -> str:
    blocks = []
    for s in sources:
        if s["type"] not in ("short-term", "long-term"):
            continue
        missing_urls = s.get("missing_urls", [])
        exclude_country = s["filters"].get("country")
        if not missing_urls:
            continue
        teaching = _universities_teaching_skills(missing_urls, exclude_country=exclude_country)
        if not teaching:
            continue
        lines = []
        for url, courses in list(teaching.items())[:20]:
            lines.append(f"    - {courses}")
        block = "\n".join(lines)
        blocks.append(
            f"[SUGGESTED UNIVERSITIES for gap of '{s['title']}' "
            f"(excluding {exclude_country})]\n{block}"
        )
    return "\n\n".join(blocks)


_SYSTEM_INSTRUCTIONS = (
    "You are an education-policy analyst for the SKILLAB project. You are given the "
    "results of skill-gap analyses comparing labour-market demand and future technology "
    "trends against university curricula. Produce concrete, actionable recommendations "
    "for curriculum development. Base every recommendation ONLY on the evidence provided; "
    "do not invent skills, universities, or numbers. Be specific and concise."
)


def _build_prompt(evidence: str, focus: Optional[str], uni_suggestions: str = "") -> str:
    focus_line = f"\nParticular focus requested: {focus}\n" if focus else ""
    suggestions_block = ""
    extra_section = ""
    if uni_suggestions.strip():
        suggestions_block = (
            f"\nUNIVERSITIES TEACHING THE GAP SKILLS (candidates to learn from):\n"
            f"{uni_suggestions}\n"
        )
        extra_section = (
            "6. **Universities to look at** — based on the candidates above (and any "
            "'gap skills taught at OTHER (model) universities' in the policy block), which "
            "universities/countries already teach the missing skills and could be models "
            "or partners. Only use universities explicitly listed.\n"
        )
    return f"""{_SYSTEM_INSTRUCTIONS}

EVIDENCE FROM SAVED ANALYSES:
{evidence}
{suggestions_block}{focus_line}
Write the recommendations as Markdown with these sections:
1. **Summary** — 2-3 sentences on the overall picture.
2. **Priority skills to add** — skills in demand / future-relevant but missing from curricula, with a one-line justification each.
3. **Skills to de-emphasise** — oversupplied skills, if any.
4. **University / country actions** — where coverage is weakest and what to do.
5. **Concrete next steps** — 3-5 bullet actions.
{extra_section}
Keep it grounded strictly in the evidence above. Output valid Markdown only."""


def _build_policy_prompt(evidence: str, focus: Optional[str], country: Optional[str]) -> str:
    focus_line = f"\nParticular focus requested: {focus}\n" if focus else ""
    target = country or "the target country"
    return f"""{_SYSTEM_INSTRUCTIONS}

CRITICAL CONTEXT:
- The TARGET country of this analysis is: {target}.
- ALL recommendations must be about universities IN {target} ONLY.
- The evidence may mention universities in OTHER countries. Those appear ONLY
  because they already teach a gap skill — they are MODELS to learn from, NOT
  the subject of the recommendations. NEVER tell a university outside {target}
  to change anything.

EVIDENCE FROM THE POLICY ANALYSIS:
{evidence}
{focus_line}
Write the recommendations as Markdown with EXACTLY these two sections:
1. **Summary** — 2-3 sentences on the coverage picture for {target}'s universities.
2. **Universities to strengthen in {target}** — for each low-coverage university IN {target}, list the specific gap skills it should add. For each gap skill, if the evidence's 'gap skills taught at OTHER (model) universities' names a university (in any country) that already teaches it, cite that university as a model to learn from. Only recommend CHANGES to universities in {target}; other universities are cited only as models.

Do NOT add any other sections. Do NOT recommend changes to universities outside {target}. Keep it grounded strictly in the evidence above. Output valid Markdown only."""


# ==========================================
# SHARED COLLECTOR
# ==========================================
def _collect_sources(st: Optional[str], lt: Optional[str], pol: Optional[str],
                     tsouk: Optional[str], policy_country: Optional[str] = None):
    sources: List[Dict[str, Any]] = []
    not_found: List[str] = []
    if st:
        s = _read_shortterm(st)
        (sources.append(s) if s else not_found.append(f"short-term '{st}'"))
    if lt:
        s = _read_longterm(lt)
        (sources.append(s) if s else not_found.append(f"long-term '{lt}'"))
    if pol:
        s = _read_policy(pol, country_override=policy_country)
        (sources.append(s) if s else not_found.append(f"policy '{pol}'"))
    if tsouk:
        s = _read_tsouk_trends(tsouk)
        (sources.append(s) if s else not_found.append(f"FTTI trends '{tsouk}'"))
    return sources, not_found


# ==========================================
# BACKGROUND WORKER
# ==========================================
def _run_generation(combo_key: str, req_data: Dict[str, Any]):
    """Runs in a background thread: build prompt, call LLM, store result."""
    try:
        sources, not_found = _collect_sources(
            req_data["shortterm_title"], req_data["longterm_title"],
            req_data["policy_title"], req_data["tsouk_title"],
            policy_country=req_data["policy_country"]
        )
        if not sources:
            _mark_failed(combo_key, f"No saved analyses found for: {', '.join(not_found)}.")
            return

        evidence = _build_evidence(sources)
        is_policy_only = (len(sources) == 1 and sources[0]["type"] == "policy")
        if is_policy_only:
            policy_country = sources[0]["filters"].get("country")
            prompt = _build_policy_prompt(evidence, req_data["focus"], policy_country)
        else:
            uni_suggestions = (_build_university_suggestions(sources)
                               if req_data["suggest_universities"] else "")
            prompt = _build_prompt(evidence, req_data["focus"], uni_suggestions)

        recommendations_md = chat_generate(prompt, temperature=0.2)
        _mark_completed(combo_key, recommendations_md)
        logger.info(f"Recommendation completed for combo_key={combo_key[:12]}...")
    except Exception as e:
        logger.exception("Background generation failed")
        _mark_failed(combo_key, str(e)[:2000])


def _mark_completed(combo_key: str, md: str):
    db = _RecSession()
    try:
        row = db.query(LLMRecommendation).filter(LLMRecommendation.combo_key == combo_key).first()
        if row:
            row.status = "completed"
            row.error = None
            row.recommendations_md = md
            db.commit()
    finally:
        db.close()


def _mark_failed(combo_key: str, error: str):
    db = _RecSession()
    try:
        row = db.query(LLMRecommendation).filter(LLMRecommendation.combo_key == combo_key).first()
        if row:
            row.status = "failed"
            row.error = error
            db.commit()
    finally:
        db.close()


# ==========================================
# REQUEST SCHEMA
# ==========================================
class RecommendRequest(BaseModel):
    shortterm_title: Optional[str] = Field(None, description="Title of a saved short-term analysis.")
    longterm_title: Optional[str] = Field(None, description="Title of a saved long-term analysis.")
    policy_title: Optional[str] = Field(None, description="Title of a saved policy analysis.")
    tsouk_title: Optional[str] = Field(None, description="FTTI trends analysis title (fetched live from the Tsouk API, no curricula).")
    focus: Optional[str] = Field(None, description="Optional extra instruction (e.g. 'focus on Greece').")
    suggest_universities: bool = Field(False, description="For short-term/long-term: also suggest OTHER universities/countries that teach the gap (missing) skills. Policy always includes this from its stored data.")
    policy_country: Optional[str] = Field(None, description="Override the country scope for the policy analysis (keeps only universities of this country).")
    force_refresh: bool = Field(False, description="Ignore cache and regenerate.")


# ==========================================
# ENDPOINTS
# ==========================================
@router.post("/generate", summary="Start (or fetch) LLM recommendations — runs in the background")
def generate_recommendations(req: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Starts generation in the BACKGROUND and returns immediately.
    - If this exact combination is already COMPLETED (and not force_refresh),
      the stored Markdown is returned right away (status=completed).
    - If it is already RUNNING, returns status=running ("your analysis is running").
    - Otherwise it is queued: returns status=running; poll /status or call
      /generate again later to get the result once completed.
    """
    _ensure_rec_schema()

    if not (req.shortterm_title or req.longterm_title or req.policy_title or req.tsouk_title):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: shortterm_title, longterm_title, policy_title, tsouk_title."
        )

    key = _combo_key(req.shortterm_title, req.longterm_title, req.policy_title,
                     req.tsouk_title, req.focus, req.suggest_universities, req.policy_country)

    db = _RecSession()
    try:
        existing = db.query(LLMRecommendation).filter(LLMRecommendation.combo_key == key).first()

        if existing and not req.force_refresh:
            if existing.status == "completed":
                return {
                    "status": "completed",
                    "cached": True,
                    "created_at": existing.created_at,
                    "recommendations": existing.recommendations_md,
                }
            if existing.status == "running":
                return {
                    "status": "running",
                    "message": "Your analysis is already running. Check back shortly.",
                }
            # failed -> allow a retry below

        # Create or reset the row to 'running'
        if existing:
            existing.status = "running"
            existing.error = None
            existing.recommendations_md = None
            existing.shortterm_title = req.shortterm_title
            existing.longterm_title = req.longterm_title
            existing.policy_title = req.policy_title
            existing.tsouk_title = req.tsouk_title
            existing.policy_country = req.policy_country
            existing.focus = req.focus
        else:
            db.add(LLMRecommendation(
                combo_key=key,
                status="running",
                shortterm_title=req.shortterm_title,
                longterm_title=req.longterm_title,
                policy_title=req.policy_title,
                tsouk_title=req.tsouk_title,
                policy_country=req.policy_country,
                focus=req.focus,
                recommendations_md=None,
            ))
        db.commit()
    finally:
        db.close()

    # Kick off the background generation (runs after the response is sent).
    req_data = {
        "shortterm_title": req.shortterm_title,
        "longterm_title": req.longterm_title,
        "policy_title": req.policy_title,
        "tsouk_title": req.tsouk_title,
        "focus": req.focus,
        "suggest_universities": req.suggest_universities,
        "policy_country": req.policy_country,
    }
    background_tasks.add_task(_run_generation, key, req_data)

    return {
        "status": "running",
        "message": "Analysis started. It runs in the background; check back shortly for the result.",
    }


@router.post("/status", summary="Check the status/result of a recommendation (by the same inputs)")
def recommendation_status(req: RecommendRequest):
    """
    Returns the status for the SAME combination of inputs:
    - completed -> includes the recommendations Markdown
    - running   -> still generating
    - failed    -> includes the error
    - not_found -> nothing started for this combination
    """
    _ensure_rec_schema()
    key = _combo_key(req.shortterm_title, req.longterm_title, req.policy_title,
                     req.tsouk_title, req.focus, req.suggest_universities, req.policy_country)
    db = _RecSession()
    try:
        row = db.query(LLMRecommendation).filter(LLMRecommendation.combo_key == key).first()
        if not row:
            return {"status": "not_found", "message": "No recommendation started for these inputs."}
        if row.status == "completed":
            return {"status": "completed", "created_at": row.created_at,
                    "recommendations": row.recommendations_md}
        if row.status == "failed":
            return {"status": "failed", "error": row.error}
        return {"status": "running", "message": "Your analysis is running."}
    finally:
        db.close()


@router.get("/list", summary="List all recommendations with their status")
def list_recommendations():
    _ensure_rec_schema()
    db = _RecSession()
    try:
        rows = db.query(LLMRecommendation).order_by(LLMRecommendation.created_at.desc()).all()
        return {"recommendations": [{
            "id": r.id,
            "status": r.status,
            "shortterm_title": r.shortterm_title,
            "longterm_title": r.longterm_title,
            "policy_title": r.policy_title,
            "tsouk_title": r.tsouk_title,
            "policy_country": r.policy_country,
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
    suggest_universities: bool = Query(False),
    policy_country: Optional[str] = Query(None),
):
    """Debug helper: shows the evidence block without calling the LLM."""
    sources, not_found = _collect_sources(shortterm_title, longterm_title, policy_title,
                                          tsouk_title, policy_country=policy_country)
    if not sources:
        raise HTTPException(status_code=404, detail=f"No saved analyses found for: {', '.join(not_found)}.")
    return {
        "used_analyses": [{"type": s["type"], "title": s["title"]} for s in sources],
        "not_found": not_found,
        "evidence": _build_evidence(sources),
        "university_suggestions": _build_university_suggestions(sources) if suggest_universities else "",
    }