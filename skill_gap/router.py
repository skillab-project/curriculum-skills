"""
router.py
=========
All FastAPI endpoints (Short-Term Analysis):
- GET  /health
- GET  /sectors
- GET  /occupations
- POST /analyze  (short term)
- GET  /results
- GET  /results/summary
"""

import logging
from collections import defaultdict
from fastapi import APIRouter, BackgroundTasks, Query
from sqlalchemy.orm import Session

from skill_gap.database import SkillGapResult, SessionLocal, Base, engine
from skill_gap.services import (
    load_sectors,
    load_occupations,
    fetch_all_skills_parallel,
    fetch_counts_parallel,
    compute_rank_score,
    compute_gap,
    get_tracker_token
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ==========================================
# BACKGROUND TASK
# ==========================================
def run_gap_analysis(threshold: float, sector, min_val: float, occupation_filter: str = None):
    """Background task: gap analysis demand vs supply per occupation (Short-Term)."""
    db = SessionLocal()
    try:
        Base.metadata.create_all(bind=engine)

        occupations = load_occupations(sector_filter=sector)
        if not occupations:
            logger.error("No occupations found.")
            return

        # Filter by occupation if provided
        if occupation_filter:
            occupations = [o for o in occupations if occupation_filter.lower() in o.lower()]
            logger.info(f"Filtered to {len(occupations)} occupations matching '{occupation_filter}'")
            if not occupations:
                logger.error(f"No occupations matched filter: '{occupation_filter}'")
                return

        logger.info(f"Starting gap analysis: {len(occupations)} occupations, threshold={threshold}, sector={sector or 'ALL'}, occupation_filter={occupation_filter or 'ALL'}")

        occ_skills_map = fetch_all_skills_parallel(occupations, min_val=min_val)

        token = get_tracker_token()
        if not token:
            logger.error("No tracker token — aborting.")
            return

        total_saved = 0

        for occ, skills in occ_skills_map.items():
            if not skills:
                continue

            logger.info(f"Processing: {occ} ({len(skills)} skills)")

            # Parallel fetch of demand + supply counts
            enriched = fetch_counts_parallel(skills, token, max_workers=3)

            # Rank scores
            demand_ranked = compute_rank_score(
                [{"skill_id": s["skill_id"], "skill_name": s["skill_name"], "demand_count": s["demand_count"]} for s in enriched],
                "demand_count"
            )
            supply_ranked = compute_rank_score(
                [{"skill_id": s["skill_id"], "skill_name": s["skill_name"], "supply_count": s["supply_count"]} for s in enriched],
                "supply_count"
            )

            # Gap
            gap_list = compute_gap(demand_ranked, supply_ranked)

            # Save
            for item in gap_list:
                existing = db.query(SkillGapResult).filter_by(
                    occupation=occ,
                    skill_id=item["skill_id"],
                    threshold=threshold,
                    sector=sector
                ).first()

                if existing:
                    existing.demand_count = item["demand_count"]
                    existing.supply_count = item["supply_count"]
                    existing.demand_score = item["demand_score"]
                    existing.supply_score = item["supply_score"]
                    existing.gap_score = item["gap_score"]
                else:
                    db.add(SkillGapResult(
                        occupation=occ,
                        skill_name=item["skill_name"],
                        skill_id=item["skill_id"],
                        demand_count=item["demand_count"],
                        supply_count=item["supply_count"],
                        demand_score=item["demand_score"],
                        supply_score=item["supply_score"],
                        gap_score=item["gap_score"],
                        threshold=threshold,
                        sector=sector
                    ))
                total_saved += 1

            db.commit()
            logger.info(f"  ✅ {occ}: {len(gap_list)} skills saved.")

        logger.info(f"🎉 Analysis complete. Total records: {total_saved}")

    except Exception as e:
        logger.error(f"❌ Gap analysis error: {e}")
        db.rollback()
    finally:
        db.close()


# ==========================================
# ENDPOINTS
# ==========================================

@router.get("/health", tags=["Meta"])
def health():
    return {"status": "running"}


@router.get("/sectors", summary="View available sectors", tags=["Occupations"])
def get_sectors(
    starts_with: str = Query(None, description="Filter sectors starting with these characters")
):
    """Returns all available sectors from the CSV. Optional starts_with filter."""
    sectors = load_sectors()
    if starts_with:
        sectors = [s for s in sectors if s.lower().startswith(starts_with.lower())]
    return {"sectors": sectors, "count": len(sectors)}


@router.get("/occupations", summary="View available occupations", tags=["Occupations"])
def get_occupations(
    sector: str = Query(None, description="Filter by sector name")
):
    """Returns all available occupations. Optional sector filter."""
    occupations = load_occupations(sector_filter=sector)
    return {"occupations": sorted(occupations), "count": len(occupations)}


@router.post("/analyze", summary="[Short-Term] Trigger skill gap analysis (Background)", tags=["Short-Term Analysis"])
def trigger_analysis(
    background_tasks: BackgroundTasks,
    threshold: float = Query(0.7, description="Minimum skill value threshold", ge=0.0, le=1.0),
    min_val: float = Query(0.0, description="Minimum skill value for filtering", ge=0.0, le=1.0),
    sector: str = Query(None, description="Optional sector filter"),
    occupation: str = Query(None, description="Optional occupation filter (partial match)")
):
    """
    Triggers the short-term demand vs supply gap analysis in the background.
    Provide an occupation to analyze a single occupation.
    """
    background_tasks.add_task(run_gap_analysis, threshold, sector, min_val, occupation)
    return {
        "message": "Gap analysis started in background.",
        "parameters": {"threshold": threshold, "min_val": min_val, "sector": sector or "ALL", "occupation": occupation or "ALL"}
    }


@router.get("/results", summary="[Short-Term] Get skill gap results", tags=["Short-Term Analysis"])
def get_results(
    occupation: str = Query(None, description="Filter by occupation name (partial match)"),
    sector: str = Query(None, description="Filter by sector used in analysis"),
    threshold: float = Query(None, description="Filter by threshold used in analysis", ge=0.0, le=1.0),
    min_gap: float = Query(None, description="Minimum gap score (positive = hot skills)"),
    max_gap: float = Query(None, description="Maximum gap score (negative = oversupplied)"),
    limit: int = Query(None, description="Max results", ge=1, le=10000),
    order: str = Query("desc", description="Sort: desc (hot first) or asc (oversupplied first)")
):
    """
    Returns short-term gap analysis results.

    - **gap_score > 0** → Hot skill (in demand > available)
    - **gap_score < 0** → Oversupplied skill (available > in demand)
    - **gap_score = 0** → Balanced
    """
    db = SessionLocal()
    try:
        q = db.query(SkillGapResult)

        if occupation:
            q = q.filter(SkillGapResult.occupation.ilike(f"%{occupation}%"))
        if sector:
            q = q.filter(SkillGapResult.sector == sector)
        if threshold is not None:
            q = q.filter(SkillGapResult.threshold.between(threshold - 0.001, threshold + 0.001))
        if min_gap is not None:
            q = q.filter(SkillGapResult.gap_score >= min_gap)
        if max_gap is not None:
            q = q.filter(SkillGapResult.gap_score <= max_gap)

        q = q.order_by(SkillGapResult.gap_score.asc() if order == "asc" else SkillGapResult.gap_score.desc())

        if limit:
            q = q.limit(limit)

        results = q.all()
        if not results:
            return {"message": "No results found.", "data": []}
        return results

    except Exception as e:
        return {"message": "Error fetching results.", "error": str(e)}
    finally:
        db.close()


@router.get("/results/summary", summary="[Short-Term] Summary: top hot & oversupplied skills per occupation", tags=["Short-Term Analysis"])
def get_summary(
    sector: str = Query(None, description="Filter by sector"),
    threshold: float = Query(None, description="Filter by threshold", ge=0.0, le=1.0),
    top_n: int = Query(10, description="Top N skills per category", ge=1, le=100)
):
    """
    Summary per occupation (Short-Term):
    - **hot_skills**: top N with the highest gap_score
    - **oversupplied_skills**: top N with the lowest gap_score
    """
    db = SessionLocal()
    try:
        q = db.query(SkillGapResult)
        if sector:
            q = q.filter(SkillGapResult.sector == sector)
        if threshold is not None:
            q = q.filter(SkillGapResult.threshold.between(threshold - 0.001, threshold + 0.001))

        all_results = q.all()
        if not all_results:
            return {"message": "No results found.", "data": []}

        grouped = defaultdict(list)
        for r in all_results:
            grouped[r.occupation].append(r)

        summary = []
        for occ, skills in grouped.items():
            sorted_skills = sorted(skills, key=lambda x: x.gap_score or 0, reverse=True)
            summary.append({
                "occupation": occ,
                "total_skills": len(skills),
                "hot_skills": [
                    {
                        "skill": s.skill_name,
                        "gap_score": s.gap_score,
                        "demand_score": s.demand_score,
                        "supply_score": s.supply_score,
                        "demand_count": s.demand_count,
                        "supply_count": s.supply_count
                    }
                    for s in sorted_skills[:top_n]
                ],
                "oversupplied_skills": [
                    {
                        "skill": s.skill_name,
                        "gap_score": s.gap_score,
                        "demand_score": s.demand_score,
                        "supply_score": s.supply_score,
                        "demand_count": s.demand_count,
                        "supply_count": s.supply_count
                    }
                    for s in sorted_skills[-top_n:]
                ]
            })

        return sorted(summary, key=lambda x: x["occupation"])

    except Exception as e:
        return {"message": "Error.", "error": str(e)}
    finally:
        db.close()