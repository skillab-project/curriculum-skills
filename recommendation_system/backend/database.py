# =============================================
# backend/database.py
# =============================================

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from recommendation_system.backend.models import Base  # Υποθέτουμε ότι το Base εισάγεται από το models.py

# ============================
# Περιβαλλοντικές μεταβλητές για σύνδεση στη βάση
# ============================
DB_HOST = os.getenv("DB_HOST", os.getenv("DATABASE_HOST", "mysql-curriculum-skill"))
DB_PORT = os.getenv("DB_PORT", os.getenv("DATABASE_PORT", "3306"))
DB_USER = os.getenv("DB_USER", os.getenv("DATABASE_USER", "root"))
DB_PASSWORD = os.getenv("DB_PASSWORD", os.getenv("DATABASE_PASSWORD", "root"))
DB_NAME = os.getenv("DB_NAME", os.getenv("DATABASE_NAME", "skillcrawl"))

# ============================
# Connection string
# ============================
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ============================
# Engine
# ============================
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

# ============================
# Session factory
# ============================
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# ============================
# Δημιουργία όλων των πινάκων
# ============================
def init_db():
    """Δημιουργεί όλους τους πίνακες αν δεν υπάρχουν."""
    Base.metadata.create_all(bind=engine)

# ============================
# Dependency για FastAPI
# ============================
def get_db():
    """
    Dependency για FastAPI: παρέχει session για κάθε request και το κλείνει μετά.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
