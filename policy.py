import os
import uuid
import logging
from typing import List, Optional
from datetime import date

from fastapi import APIRouter, Depends, BackgroundTasks, Query, Body, HTTPException
from sqlalchemy.orm import Session, declarative_base, sessionmaker, scoped_session
from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP, Float, Date, text, create_engine
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
    # Analysis metadata / key
    title = Column(String(512), nullable=True, index=True)           # unique title (analysis key)
    description = Column(String(2048), nullable=True)
    analysis_date = Column(Date, nullable=True)
    filter_country = Column(String(255), nullable=True)              # analysis filter (stored)
    university_name = Column(String(255), nullable=False, index=True)
    country = Column(String(100), nullable=True, index=True)         # the university's own country
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
    "title": "VARCHAR(512) NULL",
    "description": "VARCHAR(2048) NULL",
    "analysis_date": "DATE NULL",
    "filter_country": "VARCHAR(255) NULL",
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
_INDEXED_POLICY_COLUMNS = ("run_id", "university_name", "country", "title")


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
                            pass
        return True
    except Exception as e:
        logger.error(f"❌ policy schema migration failed: {e}")
        return False


_schema_ensured = False


def _ensure_policy_schema_once() -> None:
    global _schema_ensured
    if _schema_ensured:
        return
    if _ensure_policy_schema():
        _schema_ensured = True


def _title_exists(title: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(PolicyRecommendation).filter(PolicyRecommendation.title == title).first() is not None
    finally:
        db.close()


# ==========================================
# REQUEST SCHEMA
# ==========================================
class PolicyAnalyzeRequest(BaseModel):
    title: str = Field(..., description="Unique title for this analysis (used as key).")
    description: Optional[str] = Field(None, description="Optional description.")
    day: Optional[int] = Field(None, description="Day of analysis date.")
    month: Optional[int] = Field(None, description="Month of analysis date.")
    year: Optional[int] = Field(None, description="Year of analysis date.")
    country: Optional[str] = Field(None, description="Country filter (stored as analysis filter).")
    occupations: List[str] = Field(
        ..., min_items=1,
        description="List of occupations selected by the user.",
        example=["Web and multimedia developers", "Web technicians"]
    )
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum Importance threshold per skill.")
    top_n: int = Field(100, ge=1, le=500, description="Maximum number of skills per occupation.")


# ==========================================
# SAVE HELPER
# ==========================================
def _save_results_to_db(db: Session, results: dict, run_id: str,
                        occupations: List[str], threshold: float, top_n: int,
                        title: str, description: Optional[str],
                        analysis_date, filter_country: Optional[str]):
    """Each run is new (unique run_id) -> always insert, never overwrite."""
    universities = results.get("universities", {})
    count = 0
    for uni, data in universities.items():
        db.add(PolicyRecommendation(
            run_id=run_id,
            title=title,
            description=description,
            analysis_date=analysis_date,
            filter_country=filter_country,
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
def run_policy_analysis_logic(db: Session, run_id: str, payload: PolicyAnalyzeRequest, analysis_date):
    logger.info(
        f"Starting Policy Gap Analysis run_id={run_id} "
        f"(title={payload.title}, occupations={payload.occupations}, "
        f"threshold={payload.threshold}, top_n={payload.top_n})"
    )

    system = EducationRecommendationSystem(SERVICE2_URL)
    results = system.run_analysis(
        occupations=payload.occupations,
        skill_threshold=payload.threshold,
        top_n=payload.top_n
    )

    if "error" in results:
        logger.error(f"❌ Analysis failed: {results.get('error')}")
        return

    try:
        _save_results_to_db(
            db, results, run_id, payload.occupations, payload.threshold, payload.top_n,
            title=payload.title, description=payload.description,
            analysis_date=analysis_date, filter_country=payload.country
        )
    except Exception as e:
        logger.error(f"❌ Database save error: {e}")
        db.rollback()


def background_task_wrapper(run_id: str, payload: PolicyAnalyzeRequest, analysis_date):
    db = SessionLocal()
    try:
        run_policy_analysis_logic(db, run_id, payload, analysis_date)
    finally:
        db.close()


def _parse_date(payload: PolicyAnalyzeRequest):
    if payload.year and payload.month and payload.day:
        try:
            return date(payload.year, payload.month, payload.day)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date (day/month/year).")
    return None


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
    Accepts occupations + title/description/date/country, runs the gap analysis
    in the background, and returns a run_id. Title is the unique key.
    """
    _ensure_policy_schema_once()

    if _title_exists(payload.title):
        raise HTTPException(
            status_code=409,
            detail=f"An analysis with title '{payload.title}' already exists. Use a different title."
        )

    analysis_date = _parse_date(payload)
    run_id = str(uuid.uuid4())

    background_tasks.add_task(background_task_wrapper, run_id, payload, analysis_date)

    return {
        "message": "Analysis started in background.",
        "run_id": run_id,
        "title": payload.title,
        "parameters": {
            "occupations": payload.occupations,
            "threshold": payload.threshold,
            "top_n": payload.top_n,
            "country": payload.country
        }
    }


@router.post("/policy/analyze_sync", summary="Run multi-occupation skill-gap analysis (Blocking)")
def trigger_analysis_sync(
    payload: PolicyAnalyzeRequest = Body(...),
    db: Session = Depends(get_db)
):
    """Same logic but blocking; returns the full result. Rejects duplicate titles."""
    _ensure_policy_schema_once()

    if _title_exists(payload.title):
        raise HTTPException(
            status_code=409,
            detail=f"An analysis with title '{payload.title}' already exists. Use a different title."
        )

    analysis_date = _parse_date(payload)
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
        _save_results_to_db(
            db, results, run_id, payload.occupations, payload.threshold, payload.top_n,
            title=payload.title, description=payload.description,
            analysis_date=analysis_date, filter_country=payload.country
        )
    except Exception as e:
        logger.error(f"❌ Database save error: {e}")
        db.rollback()

    results["run_id"] = run_id
    results["title"] = payload.title
    results["description"] = payload.description
    results["date"] = analysis_date.isoformat() if analysis_date else None
    results["filter_country"] = payload.country
    return results


@router.get("/policy/status/{run_id}", summary="Check if an analysis run has finished")
def get_run_status(run_id: str, db: Session = Depends(get_db)):
    count = db.query(PolicyRecommendation).filter_by(run_id=run_id).count()
    return {
        "run_id": run_id,
        "status": "completed" if count > 0 else "pending",
        "universities_analyzed": count
    }


@router.get("/policy/runs", summary="List all analysis runs with their filters")
def list_runs(db: Session = Depends(get_db)):
    """Every analysis with its title, description, date, and filters."""
    # Select only the columns this listing needs.
    rows = (
        db.query(
            PolicyRecommendation.run_id,
            PolicyRecommendation.title,
            PolicyRecommendation.description,
            PolicyRecommendation.analysis_date,
            PolicyRecommendation.created_at,
            PolicyRecommendation.occupations,
            PolicyRecommendation.threshold,
            PolicyRecommendation.top_n,
            PolicyRecommendation.filter_country,
        )
        .order_by(PolicyRecommendation.created_at.desc())
        .all()
    )

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
                    "occupations": r.occupations,
                    "threshold": r.threshold,
                    "top_n": r.top_n,
                    "country": r.filter_country,
                },
                "universities_count": 0
            }
        seen[r.run_id]["universities_count"] += 1

    return {"runs": list(seen.values())}


@router.get("/policy/results", summary="Get per-university recommendations by title")
def get_results(
    db: Session = Depends(get_db),
    title: str = Query(None, description="Fetch the results of the analysis with this title"),
    run_id: str = Query(None, description="Alternatively, filter by a specific run_id"),
    country: str = Query(None, description="Optional: filter by the university's country"),
    university: str = Query(None, description="Optional: filter by university name"),
    limit: int = Query(None, ge=1, le=1000)
):
    """
    Per-university coverage/gap. Prefer `title` (the analysis key); `run_id`
    is still supported. Optional country/university filters on the read side.
    """
    try:
        q = db.query(PolicyRecommendation)

        if title is not None:
            q = q.filter(PolicyRecommendation.title == title)
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


@router.get("/policy/results/by_country", summary="Aggregate coverage per country for an analysis")
def get_results_by_country(
    db: Session = Depends(get_db),
    title: str = Query(None, description="Aggregate the analysis with this title"),
    run_id: str = Query(None, description="Alternatively, a specific run_id")
):
    """Aggregated coverage per country, from the per-university records."""
    try:
        q = db.query(PolicyRecommendation)
        if title is not None:
            q = q.filter(PolicyRecommendation.title == title)
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