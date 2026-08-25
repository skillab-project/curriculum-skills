import os
import uuid
import logging
from typing import List

from fastapi import APIRouter, Depends, BackgroundTasks, Query, Body
from sqlalchemy.orm import Session, declarative_base, sessionmaker, scoped_session
from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP, Float, text, create_engine
from pydantic import BaseModel, Field

from policy_engine import EducationRecommendationSystem

logger = logging.getLogger(__name__)

router = APIRouter()

# ==========================================
# DATABASE CONNECTION
# ==========================================
DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','root')}"
    f"@{os.getenv('DB_HOST','mysql-curriculum-skill')}:{os.getenv('DB_PORT','3306')}"
    f"/{os.getenv('DB_NAME','skillcrawl')}"
)
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


def get_db():
    _ensure_policy_schema_once()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ==========================================
SERVICE2_URL = os.getenv(
    "REQUIRED_SKILLS_SERVICE_URL",
    "https://portal.skillab-project.eu/diversity-analysis"
)

# ==========================================
# MYSQL MODEL — per university, with a run_id per analysis
# ==========================================
BasePolicy = declarative_base()


class PolicyRecommendation(BasePolicy):
    __tablename__ = "policy_recommendations"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), nullable=False, index=True)          # unique per analysis
    university_name = Column(String(255), nullable=False, index=True)
    country = Column(String(100), nullable=True, index=True)
    coverage_score = Column(Float, nullable=True)
    present_skills_count = Column(Integer, nullable=True)
    missing_skills_count = Column(Integer, nullable=True)
    missing_departments = Column(JSON, nullable=True)
    missing_courses = Column(JSON, nullable=True)
    # Analysis parameters
    threshold = Column(Float, nullable=True)
    top_n = Column(Integer, nullable=True)
    occupations = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


# ==========================================
# SCHEMA SELF-MIGRATION
# ==========================================
_POLICY_COLUMNS = {
    "run_id": "VARCHAR(36) NULL",
    "university_name": "VARCHAR(255) NULL",
    "country": "VARCHAR(100) NULL",
    "coverage_score": "FLOAT NULL",
    "present_skills_count": "INT NULL",
    "missing_skills_count": "INT NULL",
    "missing_departments": "JSON NULL",
    "missing_courses": "JSON NULL",
    "threshold": "FLOAT NULL",
    "top_n": "INT NULL",
    "occupations": "JSON NULL",
    "created_at": "TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP",
}
_INDEXED_POLICY_COLUMNS = ("run_id", "university_name", "country")


def _ensure_policy_schema() -> bool:
    """Create the table if missing, then add any columns the model has that the
    existing table lacks. Returns True on success. Never raises."""
    try:
        BasePolicy.metadata.create_all(bind=engine)
    except Exception as e:
        logger.error(f"❌ policy create_all failed: {e}")
        return False

    try:
        with engine.begin() as conn:
            existing = {
                row[0]
                for row in conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() "
                    "AND table_name = 'policy_recommendations'"
                ))
            }
            if not existing:
                # Table not found in this schema (nothing to migrate here).
                return True
            for col, ddl in _POLICY_COLUMNS.items():
                if col not in existing:
                    logger.warning("Adding missing column policy_recommendations.%s", col)
                    conn.execute(text(f"ALTER TABLE policy_recommendations ADD COLUMN {col} {ddl}"))
                    if col in _INDEXED_POLICY_COLUMNS:
                        try:
                            conn.execute(text(
                                f"CREATE INDEX idx_policy_{col} ON policy_recommendations ({col})"
                            ))
                        except Exception:
                            pass  # index may already exist; not critical
        return True
    except Exception as e:
        logger.error(f"❌ policy schema migration failed: {e}")
        return False


_schema_ensured = False


def _ensure_policy_schema_once() -> None:
    """Run the schema check at most once per process (retries until it succeeds)."""
    global _schema_ensured
    if _schema_ensured:
        return
    if _ensure_policy_schema():
        _schema_ensured = True


# ==========================================
# REQUEST SCHEMA
# ==========================================
class PolicyAnalyzeRequest(BaseModel):
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
        description="Maximum number of skills per occupation (e.g. top 100)."
    )


# ==========================================
# SAVE HELPER
# ==========================================
def _save_results_to_db(db: Session, results: dict, run_id: str,
                        occupations: List[str], threshold: float, top_n: int):
    """
    Each run is new (unique run_id) -> always insert, never overwrite.
    """
    universities = results.get("universities", {})
    count = 0
    for uni, data in universities.items():
        db.add(PolicyRecommendation(
            run_id=run_id,
            university_name=uni,
            country=data.get("country"),
            coverage_score=data.get("coverage_score", 0.0),
            present_skills_count=data.get("present_skills_count"),
            missing_skills_count=data.get("missing_skills_count"),
            missing_departments=data.get("missing_departments"),
            missing_courses=data.get("missing_courses"),
            threshold=threshold,
            top_n=top_n,
            occupations=occupations
        ))
        count += 1
    db.commit()
    logger.info(f"✅ Saved {count} university records for run_id={run_id}.")
    return count


# ==========================================
# WRAPPERS
# ==========================================
def run_policy_analysis_logic(db: Session, run_id: str, occupations: List[str],
                              threshold: float, top_n: int):
    logger.info(
        f"Starting Policy Gap Analysis run_id={run_id} "
        f"(occupations={occupations}, threshold={threshold}, top_n={top_n})"
    )

    system = EducationRecommendationSystem(SERVICE2_URL)
    results = system.run_analysis(
        occupations=occupations,
        skill_threshold=threshold,
        top_n=top_n
    )

    if "error" in results:
        logger.error(f"❌ Analysis failed: {results.get('error')}")
        return

    try:
        _save_results_to_db(db, results, run_id, occupations, threshold, top_n)
    except Exception as e:
        logger.error(f"❌ Database save error: {e}")
        db.rollback()


def background_task_wrapper(run_id: str, occupations: List[str],
                            threshold: float, top_n: int):
    db = SessionLocal()
    try:
        run_policy_analysis_logic(db, run_id, occupations, threshold, top_n)
    finally:
        db.close()


# ==========================================
# ENDPOINTS
# ==========================================

@router.post("/policy/analyze", summary="Trigger multi-occupation skill-gap analysis (Background)")
def trigger_analysis(
    payload: PolicyAnalyzeRequest = Body(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Accepts a list of occupations, runs the gap analysis in the background,
    and IMMEDIATELY returns a run_id. The user polls /policy/status/{run_id}
    and then reads /policy/results?run_id=...
    """
    _ensure_policy_schema_once()

    run_id = str(uuid.uuid4())

    background_tasks.add_task(
        background_task_wrapper,
        run_id,
        payload.occupations,
        payload.threshold,
        payload.top_n
    )

    return {
        "message": "Analysis started in background.",
        "run_id": run_id,
        "parameters": {
            "occupations": payload.occupations,
            "threshold": payload.threshold,
            "top_n": payload.top_n
        }
    }


@router.post("/policy/analyze_sync", summary="Run multi-occupation skill-gap analysis (Blocking)")
def trigger_analysis_sync(
    payload: PolicyAnalyzeRequest = Body(...),
    db: Session = Depends(get_db)
):
    """
    Same logic but blocking: runs, saves, and directly returns the full
    result (universities + countries + run_id) without polling. Suitable
    for testing or small occupation lists.
    """
    _ensure_policy_schema_once()

    run_id = str(uuid.uuid4())

    system = EducationRecommendationSystem(SERVICE2_URL)
    results = system.run_analysis(
        occupations=payload.occupations,
        skill_threshold=payload.threshold,
        top_n=payload.top_n
    )

    if "error" in results:
        results["run_id"] = run_id
        return results

    try:
        _save_results_to_db(db, results, run_id, payload.occupations,
                            payload.threshold, payload.top_n)
    except Exception as e:
        logger.error(f"❌ Database save error: {e}")
        db.rollback()

    results["run_id"] = run_id
    return results


@router.get("/policy/status/{run_id}", summary="Check if an analysis run has finished")
def get_run_status(run_id: str, db: Session = Depends(get_db)):
    """
    Returns whether the background run has completed (records exist) or is
    still in progress.
    """
    count = db.query(PolicyRecommendation).filter_by(run_id=run_id).count()
    return {
        "run_id": run_id,
        "status": "completed" if count > 0 else "pending",
        "universities_analyzed": count
    }


@router.get("/policy/runs", summary="List all analysis runs with their occupations")
def list_runs(db: Session = Depends(get_db)):
    """
    Returns every analysis that has been run (per run_id), so the user knows
    which run corresponds to which occupations.
    """
    rows = (
        db.query(PolicyRecommendation)
        .order_by(PolicyRecommendation.created_at.desc())
        .all()
    )

    seen = {}
    for r in rows:
        if r.run_id not in seen:
            seen[r.run_id] = {
                "run_id": r.run_id,
                "occupations": r.occupations,
                "threshold": r.threshold,
                "top_n": r.top_n,
                "created_at": r.created_at,
                "universities_count": 0
            }
        seen[r.run_id]["universities_count"] += 1

    return {"runs": list(seen.values())}


@router.get("/policy/results", summary="Get per-university recommendations for a run")
def get_results(
    db: Session = Depends(get_db),
    run_id: str = Query(None, description="Filter by a specific analysis run"),
    country: str = Query(None, description="Optional: filter by country name"),
    university: str = Query(None, description="Optional: filter by university name"),
    limit: int = Query(None, ge=1, le=1000)
):
    """
    Per-university coverage/gap. Provide run_id to get the results of a
    specific analysis (search coverage by Universities).
    """
    try:
        q = db.query(PolicyRecommendation)

        if run_id is not None:
            q = q.filter(PolicyRecommendation.run_id == run_id)
        if country is not None:
            q = q.filter(PolicyRecommendation.country.ilike(f"%{country}%"))
        if university is not None:
            q = q.filter(PolicyRecommendation.university_name.ilike(f"%{university}%"))

        q = q.order_by(PolicyRecommendation.coverage_score.desc())
        if limit is not None:
            q = q.limit(limit)

        results = q.all()
        if not results:
            return {"message": "No results found for the given filters.", "data": []}
        return results

    except Exception as e:
        return {"message": "No results yet or table missing.", "error": str(e)}


@router.get("/policy/results/by_country", summary="Aggregate coverage per country for a run")
def get_results_by_country(
    db: Session = Depends(get_db),
    run_id: str = Query(None, description="Filter by a specific analysis run")
):
    """
    Aggregated coverage per country, computed from the per-university records
    of a run: average coverage and number of universities.
    """
    try:
        q = db.query(PolicyRecommendation)
        if run_id is not None:
            q = q.filter(PolicyRecommendation.run_id == run_id)

        rows = q.all()
        if not rows:
            return {"message": "No results found for the given filters.", "data": []}

        agg = {}
        for r in rows:
            c = r.country or "Unknown"
            agg.setdefault(c, [])
            agg[c].append(r.coverage_score or 0.0)

        data = [
            {
                "country": c,
                "avg_university_coverage": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "universities_count": len(scores)
            }
            for c, scores in agg.items()
        ]
        data.sort(key=lambda x: x["avg_university_coverage"], reverse=True)
        return {"data": data}

    except Exception as e:
        return {"message": "No results yet or table missing.", "error": str(e)}