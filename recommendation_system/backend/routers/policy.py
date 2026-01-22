import os
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Query
from sqlalchemy.orm import Session, declarative_base
from sqlalchemy import Column, Integer, String, JSON, TIMESTAMP, text

# Χρησιμοποιούμε την υπάρχουσα MySQL σύνδεση
from recommendation_system.backend.database import get_db, engine
from recommendation_system.backend.policy_engine import EducationRecommendationSystem

router = APIRouter()

SERVICE2_URL = "https://portal.skillab-project.eu/required-skills"
SERVICE3_URL = "https://portal.skillab-project.eu/curriculum-skills"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "New_occupation_table.csv")

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΜΟΝΤΕΛΟΥ MYSQL (In-file)
# ==========================================
BasePolicy = declarative_base()


class PolicyRecommendation(BasePolicy):
    __tablename__ = "policy_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    country = Column(String(100), nullable=False, index=True)

    # Στη MySQL το JSON υποστηρίζεται κανονικά (MySQL 5.7+)
    missing_departments = Column(JSON, nullable=True)
    missing_courses = Column(JSON, nullable=True)

    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


# ==========================================
# 2. LOGIC & ENDPOINTS
# ==========================================

def run_policy_analysis_task(db: Session, threshold: float):
    print(f"Starting Education Policy Analysis with threshold: {threshold} (MySQL)...")

    system = EducationRecommendationSystem(SERVICE2_URL, SERVICE3_URL, CSV_PATH)
    results = system.run_analysis(skill_threshold=threshold)

    if "error" in results:
        print(f"❌ Analysis failed: {results['error']}")
        return

    try:
        count = 0
        for country, data in results.items():
            # Έλεγχος αν υπάρχει εγγραφή για τη χώρα
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
        print(f"✅ Analysis completed. Updated/Created {count} records in MySQL.")
    except Exception as e:
        print(f"❌ Database error: {e}")
        db.rollback()
    finally:
        db.close()


@router.post("/policy/analyze", summary="Trigger multi-country policy analysis (Background)")
def trigger_analysis(
        background_tasks: BackgroundTasks,
        threshold: float = Query(0.7, description="Minimum skill value threshold (0.0 - 1.0)", ge=0.0, le=1.0),
        db: Session = Depends(get_db)
):
    """
    Ξεκινάει την ανάλυση στο παρασκήνιο.
    Αυτόματα δημιουργεί τον πίνακα 'policy_recommendations' στη MySQL αν δεν υπάρχει.
    """

    # --- TABLE CREATION (MySQL) ---
    try:
        # Αυτή η εντολή δημιουργεί τον πίνακα ΜΟΝΟ αν δεν υπάρχει
        BasePolicy.metadata.create_all(bind=engine)
        print("✅ Checked/Created 'policy_recommendations' table in MySQL.")
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        raise HTTPException(status_code=500, detail=f"Could not initialize MySQL table: {e}")
    # -----------------------------------

    background_tasks.add_task(run_policy_analysis_task, db, threshold)

    return {
        "message": "Analysis started in background. Table checked/created in MySQL.",
        "parameters": {"threshold": threshold}
    }


@router.get("/policy/results", summary="Get all policy recommendations from DB")
def get_all_results(db: Session = Depends(get_db)):
    try:
        results = db.query(PolicyRecommendation).all()
        return results
    except Exception as e:
        # Πιθανόν ο πίνακας να μην υπάρχει ακόμα
        return {"message": "No results found (Table might not exist yet). Run POST first.", "error": str(e)}


@router.get("/policy/results/{country}", summary="Get policy recommendations for specific country")
def get_country_result(country: str, db: Session = Depends(get_db)):
    try:
        result = db.query(PolicyRecommendation).filter(PolicyRecommendation.country.ilike(country)).first()
        if not result:
            raise HTTPException(status_code=404, detail="Country not found in analysis results")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error (Run analysis first): {str(e)}")