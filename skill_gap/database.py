"""
database.py
===========
SQLAlchemy setup, DB model and connection helpers.
"""
import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, TIMESTAMP, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

logger = logging.getLogger(__name__)

DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','root')}"
    f"@{os.getenv('DB_HOST','mysql-curriculum-skill')}:{os.getenv('DB_PORT','3306')}"
    f"/{os.getenv('DB_NAME','skillcrawl')}"
)
engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
_SessionFactory = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
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


# ==========================================================
# SCHEMA SELF-MIGRATION
# ==========================================================
_SKILLGAP_COLUMNS = {
    "run_id": "VARCHAR(36) NULL",
    "skill_name": "VARCHAR(512) NULL",
    "skill_id": "VARCHAR(512) NULL",
    "occupations": "JSON NULL",
    "demand_count": "INT NULL",
    "supply_count": "INT NULL",
    "demand_score": "FLOAT NULL",
    "supply_score": "FLOAT NULL",
    "gap_score": "FLOAT NULL",
    "in_curriculum": "TINYINT(1) NULL",
    "curriculum_courses": "JSON NULL",
    "threshold": "FLOAT NULL",
    "top_n": "INT NULL",
    "created_at": "TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP",
}
_INDEXED_SKILLGAP_COLUMNS = ("run_id", "skill_id")


def _ensure_skillgap_schema() -> bool:
    """Create the table if missing, then add any columns the model has that the
    existing table lacks. Returns True on success. Never raises."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.error(f"skill_gap create_all failed: {e}")
        return False

    try:
        with engine.begin() as conn:
            existing = {
                row[0]
                for row in conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() "
                    "AND table_name = 'skill_gap_results'"
                ))
            }
            if not existing:
                return True
            for col, ddl in _SKILLGAP_COLUMNS.items():
                if col not in existing:
                    logger.warning("Adding missing column skill_gap_results.%s", col)
                    conn.execute(text(f"ALTER TABLE skill_gap_results ADD COLUMN {col} {ddl}"))
                    if col in _INDEXED_SKILLGAP_COLUMNS:
                        try:
                            conn.execute(text(
                                f"CREATE INDEX idx_skillgap_{col} ON skill_gap_results ({col})"
                            ))
                        except Exception:
                            pass  # index may already exist; not critical
        return True
    except Exception as e:
        logger.error(f"skill_gap schema migration failed: {e}")
        return False


_schema_ensured = False


def _ensure_skillgap_schema_once() -> None:
    """Run the schema check at most once per process (retries until it succeeds)."""
    global _schema_ensured
    if _schema_ensured:
        return
    if _ensure_skillgap_schema():
        _schema_ensured = True


def SessionLocal():
    """Session factory that also self-heals the skill_gap_results schema once
    before handing back a session. Kept callable exactly like the previous
    scoped_session so all `db = SessionLocal()` call sites are unchanged."""
    _ensure_skillgap_schema_once()
    return _SessionFactory()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()