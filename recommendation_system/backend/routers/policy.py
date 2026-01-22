import os
import logging
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Query
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP, text

# Imports από τη δομή του project
from recommendation_system.backend.database import get_db, engine, SessionLocal
from recommendation_system.backend.policy_engine import EducationRecommendationSystem

# Ρύθμιση Logger
logger = logging.getLogger(__name__)

router = APIRouter()

# ==========================================
# 1. RΥΘΜΙΣΕΙΣ & ENVIRONMENT VARIABLES
# ==========================================

# Service 2: Required Skills (Εξωτερικό API)
SERVICE2_URL = os.getenv(
    "REQUIRED_SKILLS_SERVICE_URL",
    "https://portal.skillab-project.eu/required-skills"
)

# Υπολογισμός διαδρομής CSV
# .../backend/routers/policy.py -> .../backend/data/New_occupation_table.csv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "New_occupation_table.csv")

# Έλεγχος αρχείου CSV (για debugging)
if not os.path.exists(CSV_PATH):
    logger.warning(f"⚠️ CSV file NOT found at: {CSV_PATH}. Please upload it.")
else:
    logger.info(f"✅ CSV file found at: {CSV_PATH}")

# ==========================================
# 2. ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΟΥ MYSQL (ΓΙΑ ΤΑ ΑΠΟΤΕΛΕΣΜΑΤΑ)
# ==========================================
# Χρησιμοποιούμε δικό του Base για να μην επηρεάσουμε τους άλλους πίνακες
BasePolicy = declarative_base()


class PolicyRecommendation(BasePolicy):
    __tablename__ = "policy_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    country = Column(String(100), nullable=False, index=True)

    # Αποθήκευση των αποτελεσμάτων ως JSON
    missing_departments = Column(JSON, nullable=True)
    missing_courses = Column(JSON, nullable=True)

    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


# ==========================================
# 3. BACKGROUND TASK WRAPPERS
# ==========================================

def run_policy_analysis_logic(db: Session, threshold: float):
    """
    Η κύρια λογική που τρέχει στο background.
    """
    logger.info(f"Starting Education Policy Analysis (Threshold: {threshold})")

    # Αρχικοποίηση Engine (Service 3 δεν χρειάζεται URL πια, βάζουμε dummy string)
    system = EducationRecommendationSystem(SERVICE2_URL, "unused", CSV_PATH)

    # Εκτέλεση ανάλυσης (τραβάει δεδομένα με mysql.connector μέσω του policy_engine)
    results = system.run_analysis(skill_threshold=threshold)

    if "error" in results:
        logger.error(f"❌ Analysis failed: {results.get('error')}")
        return

    try:
        count = 0
        for country, data in results.items():
            # Έλεγχος αν υπάρχει ήδη εγγραφή (SQLAlchemy)
            existing = db.query(PolicyRecommendation).filter_by(country=country).first()

            if existing:
                existing.missing_departments = data["missing_departments"]
                existing.missing_courses = data["missing_courses"]
            else:
                new_rec = PolicyRecommendation(
                    country=country,
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


def background_task_wrapper(threshold: float):
    """
    Δημιουργεί νέο Session για το background task, γιατί το session του request κλείνει.
    """
    db = SessionLocal()
    try:
        run_policy_analysis_logic(db, threshold)
    finally:
        db.close()


# ==========================================
# 4. ENDPOINTS
# ==========================================

@router.post("/policy/analyze", summary="Trigger multi-country policy analysis (Background)")
def trigger_analysis(
        background_tasks: BackgroundTasks,
        threshold: float = Query(0.7, description="Minimum skill value threshold (0.0 - 1.0)", ge=0.0, le=1.0),
        db: Session = Depends(get_db)
):
    """
    Ξεκινάει την ανάλυση στο παρασκήνιο.
    Δημιουργεί τον πίνακα 'policy_recommendations' αν δεν υπάρχει.
    """

    # --- LAZY TABLE CREATION ---
    try:
        # Δημιουργία πίνακα στη βάση (αν δεν υπάρχει ήδη)
        BasePolicy.metadata.create_all(bind=engine)
        logger.info("Checked/Created 'policy_recommendations' table.")
    except Exception as e:
        logger.error(f"❌ Error creating table: {e}")
        # Δεν κάνουμε raise εδώ για να προσπαθήσει να τρέξει το task,
        # αλλά καλό είναι να το ξέρουμε στα logs.

    # Εκκίνηση Background Task με τον wrapper
    background_tasks.add_task(background_task_wrapper, threshold)

    return {
        "message": "Analysis started in background.",
        "info": "Results will be saved to MySQL. Use GET /policy/results later to view them."
    }


@router.get("/policy/results", summary="Get all policy recommendations from DB")
def get_all_results(db: Session = Depends(get_db)):
    try:
        # Επιστρέφει όλα τα αποτελέσματα από τον πίνακα
        return db.query(PolicyRecommendation).all()
    except Exception as e:
        return {"message": "No results yet or table missing.", "error": str(e)}


@router.get("/policy/results/{country}", summary="Get policy recommendations for specific country")
def get_country_result(country: str, db: Session = Depends(get_db)):
    # Αναζήτηση βάσει χώρας (Case Insensitive με ilike)
    res = db.query(PolicyRecommendation).filter(PolicyRecommendation.country.ilike(country)).first()
    if not res:
        raise HTTPException(status_code=404, detail=f"No recommendations found for country: {country}")
    return res