"""
router.py
=========
FastAPI endpoints (Short-Term Analysis):
- GET  /health
- GET  /sectors
- GET  /occupations
- POST /analyze          (title-keyed; stores description/date/country/university)
- POST /analyze_sync
- GET  /status/{run_id}
- GET  /runs             (each run: title, description, date, filters)
- GET  /results          (by title)
- GET  /results/summary
"""
import uuid
import logging
from typing import List, Optional
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Query, Body, HTTPException
from pydantic import BaseModel, Field

from skill_gap.database import SkillGapResult, SessionLocal, Base, engine
from skill_gap.services import (
    load_sectors,
    load_occupations,
    fetch_all_skills_parallel,
    build_distinct_skill_pool,
    fetch_counts_parallel,
    compute_rank_score,
    compute_gap,
    check_skills_in_curriculum,
    get_tracker_token,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ==========================================
# REQUEST SCHEMA
# ==========================================
class GapAnalyzeRequest(BaseModel):
    title: str = Field(..., description="Unique title for this analysis (used as key).")
    description: Optional[str] = Field(None, description="Optional description.")
    day: Optional[int] = Field(None, description="Day of analysis date.")
    month: Optional[int] = Field(None, description="Month of analysis date.")
    year: Optional[int] = Field(None, description="Year of analysis date.")
    country: Optional[str] = Field(None, description="Country filter (scopes curriculum coverage).")
    university: Optional[str] = Field(None, description="University filter (stored as analysis filter).")
    occupations: List[str] = Field(
        ..., min_items=1,
        description="List of occupations selected by the user.",
        example=["Web and multimedia developers", "Web technicians"]
    )
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum Importance threshold per skill.")
    top_n: int = Field(100, ge=1, le=500, description="Maximum number of skills across the distinct pool.")


def _title_exists(title: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(SkillGapResult).filter(SkillGapResult.title == title).first() is not None
    finally:
        db.close()


# ==========================================
# CORE ANALYSIS
# ==========================================
def _run_shortterm_gap(run_id: str, req: GapAnalyzeRequest) -> dict:
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)

        occupations = [o.strip() for o in req.occupations if o and o.strip()]
        if not occupations:
            return {"error": "No occupations provided", "run_id": run_id}

        analysis_date = None
        if req.year and req.month and req.day:
            try:
                analysis_date = date(req.year, req.month, req.day)
            except ValueError:
                return {"error": "Invalid date (day/month/year)", "run_id": run_id}

        logger.info(f"[{run_id}] Fetching skills for {len(occupations)} occupations...")
        occ_skills_map = fetch_all_skills_parallel(occupations, min_val=req.threshold)

        occupations_with_skills = list(occ_skills_map.keys())
        occupations_without_skills = [o for o in occupations if o not in occ_skills_map]

        if not occ_skills_map:
            return {
                "error": "No skills returned for the selected occupations",
                "occupations_without_skills": occupations_without_skills,
                "run_id": run_id
            }

        pool = build_distinct_skill_pool(occ_skills_map, top_n=req.top_n)

        token = get_tracker_token()
        if not token:
            return {"error": "No tracker token", "run_id": run_id}

        enriched = fetch_counts_parallel(pool, token)

        demand_ranked = compute_rank_score(
            [{"skill_id": s["skill_id"], "skill_name": s["skill_name"],
              "occupations": s.get("occupations", []), "demand_count": s["demand_count"]} for s in enriched],
            "demand_count"
        )
        supply_ranked = compute_rank_score(
            [{"skill_id": s["skill_id"], "skill_name": s["skill_name"],
              "occupations": s.get("occupations", []), "supply_count": s["supply_count"]} for s in enriched],
            "supply_count"
        )

        gap_list = compute_gap(demand_ranked, supply_ranked)

        # Curriculum cross-check, scoped to country if provided
        curric = check_skills_in_curriculum([g["skill_id"] for g in gap_list], country=req.country)

        saved = 0
        for item in gap_list:
            c = curric.get(item["skill_id"], {"in_curriculum": False, "courses": []})
            db.add(SkillGapResult(
                run_id=run_id,
                title=req.title,
                description=req.description,
                analysis_date=analysis_date,
                country=req.country,
                university=req.university,
                skill_name=item["skill_name"],
                skill_id=item["skill_id"],
                occupations=item.get("occupations", []),
                demand_count=item["demand_count"],
                supply_count=item["supply_count"],
                demand_score=item["demand_score"],
                supply_score=item["supply_score"],
                gap_score=item["gap_score"],
                in_curriculum=c["in_curriculum"],
                curriculum_courses=c["courses"],
                threshold=req.threshold,
                top_n=req.top_n
            ))
            saved += 1
        db.commit()
        logger.info(f"[{run_id}] Analysis complete. {saved} skills saved.")

        return {
            "run_id": run_id,
            "title": req.title,
            "description": req.description,
            "date": analysis_date.isoformat() if analysis_date else None,
            "filters": {"country": req.country, "university": req.university,
                        "threshold": req.threshold, "top_n": req.top_n},
            "occupations_with_skills": occupations_with_skills,
            "occupations_without_skills": occupations_without_skills,
            "total_unique_skills": len(gap_list),
            "results": [
                {
                    "skill_name": item["skill_name"],
                    "skill_id": item["skill_id"],
                    "occupations": item.get("occupations", []),
                    "demand_count": item["demand_count"],
                    "supply_count": item["supply_count"],
                    "demand_score": item["demand_score"],
                    "supply_score": item["supply_score"],
                    "gap_score": item["gap_score"],
                    "in_curriculum": curric.get(item["skill_id"], {}).get("in_curriculum", False),
                    "curriculum_courses": curric.get(item["skill_id"], {}).get("courses", [])
                }
                for item in gap_list
            ]
        }

    except Exception as e:
        logger.error(f"[{run_id}] Gap analysis error: {e}")
        db.rollback()
        return {"error": str(e), "run_id": run_id}
    finally:
        db.close()


def _background_wrapper(run_id: str, req: GapAnalyzeRequest):
    _run_shortterm_gap(run_id, req)


# ==========================================
# ENDPOINTS
# ==========================================
@router.get("/health", tags=["Meta"])
def health():
    return {"status": "running"}


@router.get("/sectors", summary="View available sectors", tags=["Occupations"])
def get_sectors(starts_with: str = Query(None, description="Filter sectors starting with these characters")):
    sectors = load_sectors()
    if starts_with:
        sectors = [s for s in sectors if s.lower().startswith(starts_with.lower())]
    return {"sectors": sectors, "count": len(sectors)}


@router.get("/occupations", summary="View available occupations", tags=["Occupations"])
def get_occupations(sector: str = Query(None, description="Filter by sector name")):
    occupations = load_occupations(sector_filter=sector)
    return {"occupations": sorted(occupations), "count": len(occupations)}


@router.post("/analyze", summary="[Short-Term] Multi-occupation gap analysis (Background)", tags=["Short-Term Analysis"])
def trigger_analysis(payload: GapAnalyzeRequest = Body(...), background_tasks: BackgroundTasks = None):
    """
    Start a short-term gap analysis in the background. Title is the unique key:
    if one already exists, the request is rejected. Country scopes curriculum
    coverage; country/university are stored as analysis filters.
    """
    Base.metadata.create_all(bind=engine)
    if _title_exists(payload.title):
        raise HTTPException(
            status_code=409,
            detail=f"An analysis with title '{payload.title}' already exists. Use a different title."
        )
    run_id = str(uuid.uuid4())
    background_tasks.add_task(_background_wrapper, run_id, payload)
    return {
        "message": "Gap analysis started in background.",
        "run_id": run_id,
        "title": payload.title,
        "parameters": {
            "occupations": payload.occupations,
            "threshold": payload.threshold,
            "top_n": payload.top_n,
            "country": payload.country,
            "university": payload.university,
        }
    }


@router.post("/analyze_sync", summary="[Short-Term] Multi-occupation gap analysis (Blocking)", tags=["Short-Term Analysis"])
def trigger_analysis_sync(payload: GapAnalyzeRequest = Body(...)):
    """Same as /analyze but blocking; returns the full result. Rejects duplicate titles."""
    Base.metadata.create_all(bind=engine)
    if _title_exists(payload.title):
        raise HTTPException(
            status_code=409,
            detail=f"An analysis with title '{payload.title}' already exists. Use a different title."
        )
    run_id = str(uuid.uuid4())
    return _run_shortterm_gap(run_id, payload)


@router.get("/status/{run_id}", summary="[Short-Term] Check if a run has finished", tags=["Short-Term Analysis"])
def get_status(run_id: str):
    db = SessionLocal()
    try:
        count = db.query(SkillGapResult).filter_by(run_id=run_id).count()
        return {
            "run_id": run_id,
            "status": "completed" if count > 0 else "pending",
            "skills_analyzed": count
        }
    finally:
        db.close()


@router.get("/runs", summary="[Short-Term] List all analysis runs with their filters", tags=["Short-Term Analysis"])
def list_runs():
    db = SessionLocal()
    try:
        rows = db.query(SkillGapResult).order_by(SkillGapResult.created_at.desc()).all()
        seen = {}
        for r in rows:
            if r.run_id not in seen:
                seen[r.run_id] = {
                    "run_id": r.run_id,
                    "title": r.title,
                    "description": r.description,
                    "date": r.analysis_date.isoformat() if r.analysis_date else None,
                    "created_at": r.created_at,
                    "filters": {
                        "occupations": set(),
                        "threshold": r.threshold,
                        "top_n": r.top_n,
                        "country": r.country,
                        "university": r.university,
                    },
                    "skills_count": 0,
                }
            seen[r.run_id]["skills_count"] += 1
            if r.occupations:
                seen[r.run_id]["filters"]["occupations"].update(r.occupations)
        runs = []
        for v in seen.values():
            v["filters"]["occupations"] = sorted(v["filters"]["occupations"])
            runs.append(v)
        return {"runs": runs}
    finally:
        db.close()


@router.get("/results", summary="[Short-Term] Get skill gap results by title", tags=["Short-Term Analysis"])
def get_results(
    title: str = Query(None, description="Fetch the results of the analysis with this title"),
    run_id: str = Query(None, description="Alternatively, filter by a specific run_id"),
    occupation: str = Query(None, description="Filter: skill required by this occupation"),
    in_curriculum: bool = Query(None, description="Filter: present (true) / absent (false) in curricula"),
    min_gap: float = Query(None, description="Minimum gap score (positive = hot skills)"),
    max_gap: float = Query(None, description="Maximum gap score (negative = oversupplied)"),
    limit: int = Query(None, description="Max results", ge=1, le=10000),
    order: str = Query("desc", description="Sort: desc (hot first) or asc (oversupplied first)")
):
    """
    Returns short-term gap results. Prefer `title` (the analysis key); `run_id`
    is still supported. gap_score > 0 = hot, < 0 = oversupplied.
    """
    db = SessionLocal()
    try:
        q = db.query(SkillGapResult)
        if title:
            q = q.filter(SkillGapResult.title == title)
        if run_id:
            q = q.filter(SkillGapResult.run_id == run_id)
        if in_curriculum is not None:
            q = q.filter(SkillGapResult.in_curriculum == in_curriculum)
        if min_gap is not None:
            q = q.filter(SkillGapResult.gap_score >= min_gap)
        if max_gap is not None:
            q = q.filter(SkillGapResult.gap_score <= max_gap)

        q = q.order_by(SkillGapResult.gap_score.asc() if order == "asc" else SkillGapResult.gap_score.desc())
        if limit:
            q = q.limit(limit)
        results = q.all()

        if occupation:
            results = [
                r for r in results
                if r.occupations and any(occupation.lower() in o.lower() for o in r.occupations)
            ]

        if not results:
            return {"message": "No results found.", "data": []}
        return results
    except Exception as e:
        return {"message": "Error fetching results.", "error": str(e)}
    finally:
        db.close()


@router.get("/results/summary", summary="[Short-Term] Summary: top hot & oversupplied skills", tags=["Short-Term Analysis"])
def get_summary(
    title: str = Query(None, description="Summary for the analysis with this title"),
    run_id: str = Query(None, description="Alternatively, a specific run_id"),
    top_n: int = Query(10, description="Top N skills per category", ge=1, le=100)
):
    """
    Summary: hot_skills (gap_score > 0) and oversupplied_skills (gap_score < 0),
    split by sign so a skill never appears in both.
    """
    db = SessionLocal()
    try:
        q = db.query(SkillGapResult)
        if title:
            q = q.filter(SkillGapResult.title == title)
        if run_id:
            q = q.filter(SkillGapResult.run_id == run_id)
        all_results = q.all()
        if not all_results:
            return {"message": "No results found.", "data": []}

        def fmt(s):
            return {
                "skill": s.skill_name,
                "gap_score": s.gap_score,
                "demand_score": s.demand_score,
                "supply_score": s.supply_score,
                "demand_count": s.demand_count,
                "supply_count": s.supply_count,
                "occupations": s.occupations or [],
                "in_curriculum": s.in_curriculum,
                "curriculum_courses": s.curriculum_courses or []
            }

        hot = [s for s in all_results if (s.gap_score or 0) > 0]
        oversupplied = [s for s in all_results if (s.gap_score or 0) < 0]

        hot_sorted = sorted(hot, key=lambda x: x.gap_score or 0, reverse=True)
        oversupplied_sorted = sorted(oversupplied, key=lambda x: x.gap_score or 0)

        return {
            "run_id": run_id,
            "title": title,
            "total_skills": len(all_results),
            "hot_skills": [fmt(s) for s in hot_sorted[:top_n]],
            "oversupplied_skills": [fmt(s) for s in oversupplied_sorted[:top_n]]
        }
    except Exception as e:
        return {"message": "Error.", "error": str(e)}
    finally:
        db.close()