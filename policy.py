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

# ΝΕΟ PATH: Βρίσκει τον φάκελο data στο root directory
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

    # ΣΤΗΛΗ ΓΙΑ ΤΟ COVERAGE (π.χ. 45.2%)
    coverage_score = Column(Float, nullable=True)

    missing_departments = Column(JSON, nullable=True)
    missing_courses = Column(JSON, nullable=True)
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
            existing = db.query(PolicyRecommendation).filter_by(country=country).first()

            # Παίρνουμε το σκορ από τα αποτελέσματα
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
                    missing_courses=data["missing_courses"]
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

@router.get("/policy/sectors", summary="View available education sectors (US 37.1)")
def get_sectors():
    system = EducationRecommendationSystem(SERVICE2_URL, "unused", CSV_PATH)
    sectors = system.get_available_sectors()
    return {"sectors": sectors}


@router.post("/policy/analyze", summary="Trigger multi-country policy analysis (Background) (US 37.2)")
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


@router.get("/policy/results", summary="Get all policy recommendations from DB")
def get_all_results(db: Session = Depends(get_db)):
    try:
        return db.query(PolicyRecommendation).all()
    except Exception as e:
        return {"message": "No results yet or table missing.", "error": str(e)}


@router.get("/policy/results/{country}", summary="Get policy recommendations for specific country")
def get_country_result(country: str, db: Session = Depends(get_db)):
    res = db.query(PolicyRecommendation).filter(PolicyRecommendation.country.ilike(country)).first()
    if not res:
        raise HTTPException(status_code=404, detail=f"No recommendations found for country: {country}")
    return res