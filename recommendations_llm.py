# ==============================================================================
# LLM RECOMMENDATIONS
# Reads saved analysis results (short-term / long-term / policy) from the DB and
# can also fetch FTTI (Tsouk) trends live by title. Builds a compact evidence
# summary, asks the Mistral LLM for curriculum recommendations (Markdown), and
# CACHES the result in the DB keyed by the exact combination of titles (+ focus
# + suggest_universities + policy_country). Same combo -> cached, no new LLM call.
#
# University suggestions:
#  - short-term / long-term: opt-in via suggest_universities=true.
#  - policy: always included from stored missing_courses.
#
# Policy scoping:
#  - policy_country (request) overrides the analysis' stored filter_country.
#  - Once a country is in effect, ONLY that country's universities are kept
#    (no leaking of other countries even if the result is empty).
#
# Policy-only requests use a SHORTER prompt: Summary + universities for the
# target country. The prompt makes clear the TARGET country is the one being
# improved, and any other countries are only models where the gap skills are
# already taught.
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
    combo_key = Column(String(64), nullable=False, unique=True, index=True)  # hash of titles+focus+suggest+country
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
    Read a policy analysis by title. The country scope is: country_override (from
    the request) if given, else the stored filter_country. Once a country is in
    effect, ONLY that country's universities are kept — even if the result is
    empty (so other countries never leak into the evidence).
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


def _universities_teaching_skills(skill_urls: List[str],
                                  exclude_country: Optional[str] = None,
                                  limit_per_skill: int = 5) -> Dict[str, List[str]]:
    """
    For a list of ESCO skill urls, find which universities teach them (by
    Skill.skill_url). Optionally EXCLUDE a country (e.g. the analysis country),
    so it suggests OTHER universities/countries that cover the gap.
    Returns: { skill_url: ["Course (University) - [Country]", ...] }
    """
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
            top_countries = [f"{c['country']} ({c['avg_coverage']}%)" for c in _trim(s["countries"], 10)]
            low_unis = sorted(s["universities"], key=lambda x: x["coverage_score"] or 0)[:10]
            low = [f"{u['university']} {u['coverage_score']}%" for u in low_unis]

            suggest_lines = []
            for u in low_unis[:5]:
                mc = u.get("missing_courses") or {}
                if isinstance(mc, dict) and mc:
                    for skill, courses in list(mc.items())[:5]:
                        clist = courses[:3] if isinstance(courses, list) else []
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
    """
    For the missing (gap) skills of short-term / long-term analyses, find OTHER
    universities that teach them (excluding the analysis country) and format as
    evidence for the LLM. Policy already carries this info in its own block.
    """
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
    """
    Shorter prompt used when the ONLY source is a policy analysis. The TARGET
    country is the one to improve; any other countries in the evidence are ONLY
    models where the gap skills are already taught — never the subject of the
    recommendations.
    """
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
    force_refresh: bool = Field(False, description="Ignore cache and regenerate with a new LLM call.")


# ==========================================
# ENDPOINTS
# ==========================================
@router.post("/generate", summary="Generate (or return cached) LLM recommendations from saved analyses / FTTI trends")
def generate_recommendations(req: RecommendRequest):
    """
    Give one or more titles: short-term / long-term / policy (read from the DB),
    and/or an FTTI trends title (fetched live from the Tsouk API). Use policy_country
    to force the policy scope to one country. When ONLY policy_title is given, a
    shorter prompt is used: Summary + universities to strengthen in the target
    country (other countries are cited only as models). The Markdown is CACHED
    keyed by the exact title combination (+ focus + suggest_universities +
    policy_country). Set force_refresh=true to regenerate.
    """
    _ensure_rec_schema()

    if not (req.shortterm_title or req.longterm_title or req.policy_title or req.tsouk_title):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: shortterm_title, longterm_title, policy_title, tsouk_title."
        )

    key = _combo_key(req.shortterm_title, req.longterm_title, req.policy_title,
                     req.tsouk_title, req.focus, req.suggest_universities, req.policy_country)

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
        req.shortterm_title, req.longterm_title, req.policy_title, req.tsouk_title,
        policy_country=req.policy_country
    )
    if not sources:
        raise HTTPException(status_code=404, detail=f"No saved analyses found for: {', '.join(not_found)}.")

    evidence = _build_evidence(sources)

    # Policy-only analyses get a shorter, target-country-focused prompt.
    is_policy_only = (len(sources) == 1 and sources[0]["type"] == "policy")
    if is_policy_only:
        policy_country = sources[0]["filters"].get("country")
        prompt = _build_policy_prompt(evidence, req.focus, policy_country)
    else:
        uni_suggestions = _build_university_suggestions(sources) if req.suggest_universities else ""
        prompt = _build_prompt(evidence, req.focus, uni_suggestions)

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
        "suggest_universities": req.suggest_universities,
        "policy_country": req.policy_country,
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
    suggest_universities: bool = Query(False),
    policy_country: Optional[str] = Query(None),
):
    """Debug helper: shows the evidence block (and optional university suggestions) without calling the LLM."""
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