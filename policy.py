import os
import logging
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Query
from sqlalchemy.orm import Session, declarative_base, sessionmaker, scoped_session
from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP, Float, text, create_engine

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
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
SERVICE2_URL = os.getenv(
    "REQUIRED_SKILLS_SERVICE_URL",
    "https://portal.skillab-project.eu/required-skills"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "New_occupation_table.csv")

if not os.path.exists(CSV_PATH):
    logger.warning(f"⚠️ CSV file NOT found at: {CSV_PATH}")
else:
    logger.info(f"✅ CSV file found at: {CSV_PATH}")

# ==========================================
# MYSQL MODEL
# ==========================================
BasePolicy = declarative_base()


class PolicyRecommendation(BasePolicy):
    __tablename__ = "policy_recommendations"
    id = Column(Integer, primary_key=True, index=True)
    country = Column(String(100), nullable=False, index=True)
    coverage_score = Column(Float, nullable=True)
    missing_departments = Column(JSON, nullable=True)
    missing_courses = Column(JSON, nullable=True)
    # Αποθηκεύουμε και τις παραμέτρους της ανάλυσης για φιλτράρισμα αργότερα
    threshold = Column(Float, nullable=True)
    sector = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


# ==========================================
# WRAPPERS
# ==========================================
def run_policy_analysis_logic(db: Session, threshold: float, sector: str = None):
    logger.info(f"Starting Education Policy Analysis (Threshold: {threshold}, Sector: {sector or 'ALL'})")

    system = EducationRecommendationSystem(SERVICE2_URL, "unused", CSV_PATH)
    results = system.run_analysis(skill_threshold=threshold, sector=sector)

    if "error" in results:
        logger.error(f"❌ Analysis failed: {results.get('error')}")
        return

    try:
        count = 0
        for country, data in results.items():
            # Ψάχνουμε existing record με ίδιο country + threshold + sector
            existing = db.query(PolicyRecommendation).filter_by(
                country=country,
                threshold=threshold,
                sector=sector
            ).first()

            cov_score = data.get("coverage_score", 0.0)

            if existing:
                existing.coverage_score = cov_score
                existing.missing_departments = data["missing_departments"]
                existing.missing_courses = data["missing_courses"]
            else:
                new_rec = PolicyRecommendation(
                    country=country,
                    coverage_score=cov_score,
                    missing_departments=data["missing_departments"],
                    missing_courses=data["missing_courses"],
                    threshold=threshold,
                    sector=sector
                )
                db.add(new_rec)
            count += 1

        db.commit()
        logger.info(f"✅ Analysis completed. Saved/Updated {count} country records.")
    except Exception as e:
        logger.error(f"❌ Database save error: {e}")
        db.rollback()


def background_task_wrapper(threshold: float, sector: str = None):
    db = SessionLocal()
    try:
        run_policy_analysis_logic(db, threshold, sector)
    finally:
        db.close()


# ==========================================
# ENDPOINTS
# ==========================================

@router.get("/policy/sectors", summary="View available education sectors")
def get_sectors(
    starts_with: str = Query(None, description="Optional: Filter sectors that start with these characters")
):
    """
    Επιστρέφει όλα τα διαθέσιμα sectors.
    Αν δοθεί η παράμετρος starts_with, επιστρέφει μόνο τα sectors που αρχίζουν με τα συγκεκριμένα γράμματα.
    """
    system = EducationRecommendationSystem(SERVICE2_URL, "unused", CSV_PATH)
    sectors = system.get_available_sectors()

    if starts_with:
        sectors = [s for s in sectors if s.lower().startswith(starts_with.lower())]

    return {"sectors": sectors, "count": len(sectors)}


@router.post("/policy/analyze", summary="Trigger multi-country policy analysis (Background)")
def trigger_analysis(
        background_tasks: BackgroundTasks,
        threshold: float = Query(0.7, description="Minimum skill value threshold", ge=0.0, le=1.0),
        sector: str = Query(None, description="Optional: Filter by sector name"),
        db: Session = Depends(get_db)
):
    try:
        BasePolicy.metadata.create_all(bind=engine)
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")

    background_tasks.add_task(background_task_wrapper, threshold, sector)

    return {
        "message": "Analysis started in background.",
        "parameters": {"threshold": threshold, "sector": sector or "ALL"}
    }

@router.get("/policy/results", summary="Get policy recommendations from DB")
def get_results(
    db: Session = Depends(get_db),
    threshold: float = Query(None, description="Optional: Filter by threshold used in analysis", ge=0.0, le=1.0),
    sector: str = Query(None, description="Optional: Filter by sector used in analysis"),
    country: str = Query(None, description="Optional: Filter by country name"),
    limit: int = Query(None, description="Optional: Max number of results to return", ge=1, le=1000)
):
    try:
        id_query = db.query(PolicyRecommendation.id, PolicyRecommendation.coverage_score)

        if threshold is not None:
            id_query = id_query.filter(
                PolicyRecommendation.threshold.between(threshold - 0.001, threshold + 0.001)
            )
        if sector is not None:
            id_query = id_query.filter(PolicyRecommendation.sector == sector)
        if country is not None:
            id_query = id_query.filter(PolicyRecommendation.country.ilike(f"%{country}%"))

        q = id_query.order_by(PolicyRecommendation.coverage_score.desc())
        if limit is not None:
            q = q.limit(limit)
        top_ids = [row.id for row in q.all()]

        if not top_ids:
            return {"message": "No results found for the given filters.", "data": []}

        results = db.query(PolicyRecommendation).filter(
            PolicyRecommendation.id.in_(top_ids)
        ).all()

        results.sort(key=lambda r: r.coverage_score or 0, reverse=True)
        return results

    except Exception as e:
        return {"message": "No results yet or table missing.", "error": str(e)}