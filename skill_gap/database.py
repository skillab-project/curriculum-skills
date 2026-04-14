"""
database.py
===========
SQLAlchemy setup, DB model and connection helpers.
"""

import os
from sqlalchemy import create_engine, Column, Integer, String, Float, TIMESTAMP, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','root')}"
    f"@{os.getenv('DB_HOST','mysql-skill-gap')}:{os.getenv('DB_PORT','3306')}"
    f"/{os.getenv('DB_NAME','skillcrawl')}"
)

engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()


class SkillGapResult(Base):
    __tablename__ = "skill_gap_results"

    id = Column(Integer, primary_key=True, index=True)
    occupation = Column(String(255), nullable=False, index=True)
    skill_name = Column(String(512), nullable=True)
    skill_id = Column(String(512), nullable=True)

    # Counts from tracker API
    demand_count = Column(Integer, nullable=True)   # how many job ads contain this skill
    supply_count = Column(Integer, nullable=True)   # how many CVs contain this skill

    # Rank-based scores (0-100%)
    demand_score = Column(Float, nullable=True)     # position in demand list
    supply_score = Column(Float, nullable=True)     # position in supply list

    # Gap = demand_score - supply_score
    # > 0 → hot skill, < 0 → oversupplied
    gap_score = Column(Float, nullable=True)

    # Analysis parameters
    threshold = Column(Float, nullable=True)
    sector = Column(String(255), nullable=True)

    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()