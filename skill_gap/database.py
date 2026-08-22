"""
database.py
===========
SQLAlchemy setup, DB model and connection helpers.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, TIMESTAMP, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','root')}"
    f"@{os.getenv('DB_HOST','mysql-curriculum-skill')}:{os.getenv('DB_PORT','3306')}"
    f"/{os.getenv('DB_NAME','skillcrawl')}"
)
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


class SkillGapResult(Base):
    __tablename__ = "skill_gap_results"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(36), nullable=False, index=True)   # unique per analysis run
    skill_name = Column(String(512), nullable=True)
    skill_id = Column(String(512), nullable=True, index=True)  # ESCO url
    # Which of the selected occupations required this skill
    occupations = Column(JSON, nullable=True)
    # Counts from tracker API
    demand_count = Column(Integer, nullable=True)   # how many job ads contain this skill
    supply_count = Column(Integer, nullable=True)   # how many CVs contain this skill
    # Rank-based scores (0-100%)
    demand_score = Column(Float, nullable=True)     # position in demand list
    supply_score = Column(Float, nullable=True)     # position in supply list
    # Gap = demand_score - supply_score
    # > 0 -> hot skill, < 0 -> oversupplied
    gap_score = Column(Float, nullable=True)
    # Curriculum cross-check (against the universities DB)
    in_curriculum = Column(Boolean, nullable=True)          # YES/NO: taught anywhere in the DB
    curriculum_courses = Column(JSON, nullable=True)        # in which courses it is taught
    # Analysis parameters
    threshold = Column(Float, nullable=True)
    top_n = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()