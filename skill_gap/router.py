"""
router.py
=========
FastAPI endpoints (Short-Term Analysis):
- GET  /health
- GET  /sectors
- GET  /occupations
- POST /analyze          (short term, multi-occupation, distinct union)
- POST /analyze_sync     (blocking variant that returns the result)
- GET  /status/{run_id}
- GET  /runs
- GET  /results
- GET  /results/summary
"""
import uuid
import logging
from typing import List
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, Query, Body
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
    occupations: List[str] = Field(
        ...,
        min_items=1,
        description="List of occupations selected by the user.",
        example=["Web and multimedia developers", "Web technicians"]
    )
    threshold: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Minimum Importance threshold per skill (0.0 = rely on top_n)."
    )
    top_n: int = Field(
        100, ge=1, le=500,
        description="Maximum number of skills across the distinct pool."
    )


# ==========================================
# CORE ANALYSIS
# ==========================================
def _run_shortterm_gap(run_id: str, occupations: List[str], threshold: float, top_n: int) -> dict:
    """
    Multi-occupation short-term gap analysis with a single distinct ranking.

    1. Fetch skills for each occupation from Trig.
    2. Build a DISTINCT skill pool (by skill_id), keeping which occupations
       required each skill; cap at top_n by importance.
    3. Fetch demand/supply counts ONCE per unique skill.
    4. Compute rank scores + gap over the unified pool.
    5. Cross-check each skill against the curriculum DB (YES/NO + courses).
    6. Persist per skill with the run_id.
    """
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)

        occupations = [o.strip() for o in occupations if o and o.strip()]
        if not occupations:
            return {"error": "No occupations provided", "run_id": run_id}

        logger.info(f"[{run_id}] Fetching skills for {len(occupations)} occupations...")
        occ_skills_map = fetch_all_skills_parallel(occupations, min_val=threshold)

        occupations_with_skills = list(occ_skills_map.keys())
        occupations_without_skills = [o for o in occupations if o not in occ_skills_map]

        if not occ_skills_map:
            return {
                "error": "No skills returned for the selected occupations",
                "occupations_without_skills": occupations_without_skills,
                "run_id": run_id
            }

        # Distinct union pool (single ranking basis), capped at top_n
        pool = build_distinct_skill_pool(occ_skills_map, top_n=top_n)

        token = get_tracker_token()
        if not token:
            return {"error": "No tracker token", "run_id": run_id}

        # Demand + supply once per unique skill
        enriched = fetch_counts_parallel(pool, token, max_workers=3)

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

        # Curriculum cross-check (YES/NO + courses) for all skills at once
        curric = check_skills_in_curriculum([g["skill_id"] for g in gap_list])

        # Persist
        saved = 0
        for item in gap_list:
            c = curric.get(item["skill_id"], {"in_curriculum": False, "courses": []})
            db.add(SkillGapResult(
                run_id=run_id,
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
                threshold=threshold,
                top_n=top_n
            ))
            saved += 1
        db.commit()
        logger.info(f"🎉 [{run_id}] Analysis complete. {saved} skills saved.")

        return {
            "run_id": run_id,
            "occupations_with_skills": occupations_with_skills,
            "occupations_without_skills": occupations_without_skills,
            "total_unique_skills": len(gap_list),
            "top_n": top_n,
            "threshold": threshold,
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
        logger.error(f"❌ [{run_id}] Gap analysis error: {e}")
        db.rollback()
        return {"error": str(e), "run_id": run_id}
    finally:
        db.close()


def _background_wrapper(run_id: str, occupations: List[str], threshold: float, top_n: int):
    _run_shortterm_gap(run_id, occupations, threshold, top_n)


# ==========================================
# ENDPOINTS
# ==========================================
@router.get("/health", tags=["Meta"])
def health():
    return {"status": "running"}


@router.get("/sectors", summary="View available sectors", tags=["Occupations"])
def get_sectors(starts_with: str = Query(None, description="Filter sectors starting with these characters")):
    """Returns all available sectors from the CSV. Optional starts_with filter."""
    sectors = load_sectors()
    if starts_with:
        sectors = [s for s in sectors if s.lower().startswith(starts_with.lower())]
    return {"sectors": sectors, "count": len(sectors)}


@router.get("/occupations", summary="View available occupations", tags=["Occupations"])
def get_occupations(sector: str = Query(None, description="Filter by sector name")):
    """Returns all available occupations. Optional sector filter."""
    occupations = load_occupations(sector_filter=sector)
    return {"occupations": sorted(occupations), "count": len(occupations)}


@router.post("/analyze", summary="[Short-Term] Multi-occupation gap analysis (Background)", tags=["Short-Term Analysis"])
def trigger_analysis(payload: GapAnalyzeRequest = Body(...), background_tasks: BackgroundTasks = None):
    """
    Triggers the short-term demand vs supply gap analysis in the background
    for a user-selected list of occupations. Skills are merged into a single
    distinct pool and ranked together. Returns a run_id immediately.
    """
    run_id = str(uuid.uuid4())
    background_tasks.add_task(_background_wrapper, run_id, payload.occupations, payload.threshold, payload.top_n)
    return {
        "message": "Gap analysis started in background.",
        "run_id": run_id,
        "parameters": {
            "occupations": payload.occupations,
            "threshold": payload.threshold,
            "top_n": payload.top_n
        }
    }


@router.post("/analyze_sync", summary="[Short-Term] Multi-occupation gap analysis (Blocking)", tags=["Short-Term Analysis"])
def trigger_analysis_sync(payload: GapAnalyzeRequest = Body(...)):
    """
    Same as /analyze but blocking: runs and directly returns the full result
    (ranked distinct skills + curriculum cross-check + run_id).
    """
    run_id = str(uuid.uuid4())
    return _run_shortterm_gap(run_id, payload.occupations, payload.threshold, payload.top_n)


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


@router.get("/runs", summary="[Short-Term] List all analysis runs", tags=["Short-Term Analysis"])
def list_runs():
    db = SessionLocal()
    try:
        rows = db.query(SkillGapResult).order_by(SkillGapResult.created_at.desc()).all()
        seen = {}
        for r in rows:
            if r.run_id not in seen:
                seen[r.run_id] = {
                    "run_id": r.run_id,
                    "threshold": r.threshold,
                    "top_n": r.top_n,
                    "created_at": r.created_at,
                    "skills_count": 0,
                    "_occ": set()
                }
            seen[r.run_id]["skills_count"] += 1
            if r.occupations:
                seen[r.run_id]["_occ"].update(r.occupations)
        runs = []
        for v in seen.values():
            occ = sorted(v.pop("_occ"))
            v["occupations"] = occ
            runs.append(v)
        return {"runs": runs}
    finally:
        db.close()


@router.get("/results", summary="[Short-Term] Get skill gap results", tags=["Short-Term Analysis"])
def get_results(
    run_id: str = Query(None, description="Filter by a specific analysis run"),
    occupation: str = Query(None, description="Filter: skill required by this occupation"),
    in_curriculum: bool = Query(None, description="Filter: only skills present (true) / absent (false) in curricula"),
    min_gap: float = Query(None, description="Minimum gap score (positive = hot skills)"),
    max_gap: float = Query(None, description="Maximum gap score (negative = oversupplied)"),
    limit: int = Query(None, description="Max results", ge=1, le=10000),
    order: str = Query("desc", description="Sort: desc (hot first) or asc (oversupplied first)")
):
    """
    Returns short-term gap analysis results.
    - **gap_score > 0** -> Hot skill (demand > supply)
    - **gap_score < 0** -> Oversupplied skill (supply > demand)
    - **in_curriculum** -> whether the skill is taught anywhere in the DB
    - **curriculum_courses** -> in which courses it is taught
    """
    db = SessionLocal()
    try:
        q = db.query(SkillGapResult)
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

        # occupation filter is applied in Python (JSON column)
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
    run_id: str = Query(None, description="Filter by a specific analysis run"),
    top_n: int = Query(10, description="Top N skills per category", ge=1, le=100)
):
    """
    Summary for a run (Short-Term):
    - **hot_skills**: top N with the highest gap_score
    - **oversupplied_skills**: top N with the lowest gap_score
    Each skill includes its curriculum presence.
    """
    db = SessionLocal()
    try:
        q = db.query(SkillGapResult)
        if run_id:
            q = q.filter(SkillGapResult.run_id == run_id)
        all_results = q.all()
        if not all_results:
            return {"message": "No results found.", "data": []}

        sorted_skills = sorted(all_results, key=lambda x: x.gap_score or 0, reverse=True)

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

        return {
            "run_id": run_id,
            "total_skills": len(all_results),
            "hot_skills": [fmt(s) for s in sorted_skills[:top_n]],
            "oversupplied_skills": [fmt(s) for s in sorted_skills[-top_n:]]
        }
    except Exception as e:
        return {"message": "Error.", "error": str(e)}
    finally:
        db.close()