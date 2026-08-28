"""
database.py
===========
SQLAlchemy setup, DB model and connection helpers.
"""
import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Boolean, TIMESTAMP, Date, text
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
    # Analysis metadata / key
    title = Column(String(512), nullable=True, index=True)    # unique title (analysis key)
    description = Column(String(2048), nullable=True)
    analysis_date = Column(Date, nullable=True)
    # Analysis filters
    country = Column(String(255), nullable=True)              # scopes curriculum coverage
    university = Column(String(512), nullable=True)           # stored filter
    skill_name = Column(String(512), nullable=True)
    skill_id = Column(String(512), nullable=True, index=True)  # ESCO url
    occupations = Column(JSON, nullable=True)
    demand_count = Column(Integer, nullable=True)
    supply_count = Column(Integer, nullable=True)
    demand_score = Column(Float, nullable=True)
    supply_score = Column(Float, nullable=True)
    gap_score = Column(Float, nullable=True)
    in_curriculum = Column(Boolean, nullable=True)
    curriculum_courses = Column(JSON, nullable=True)
    threshold = Column(Float, nullable=True)
    top_n = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


# ==========================================================
# SCHEMA SELF-MIGRATION
# ==========================================================
_SKILLGAP_COLUMNS = {
    "run_id": "VARCHAR(36) NULL",
    "title": "VARCHAR(512) NULL",
    "description": "VARCHAR(2048) NULL",
    "analysis_date": "DATE NULL",
    "country": "VARCHAR(255) NULL",
    "university": "VARCHAR(512) NULL",
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
_INDEXED_SKILLGAP_COLUMNS = ("run_id", "skill_id", "title")


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
            meta = conn.execute(text(
                "SELECT column_name, column_type, is_nullable, column_key, extra "
                "FROM information_schema.columns "
                "WHERE table_schema = DATABASE() "
                "AND table_name = 'skill_gap_results'"
            )).fetchall()
            if not meta:
                return True

            existing = {row[0] for row in meta}

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
                            pass

            model_cols = set(_SKILLGAP_COLUMNS.keys()) | {"id"}
            for col_name, col_type, is_nullable, col_key, extra in meta:
                if col_name in model_cols:
                    continue
                if str(col_key).upper() == "PRI" or "auto_increment" in str(extra or "").lower():
                    continue
                if str(is_nullable).upper() == "NO":
                    logger.warning(
                        "Relaxing leftover NOT NULL column skill_gap_results.%s to allow NULL", col_name
                    )
                    try:
                        conn.execute(text(
                            f"ALTER TABLE skill_gap_results MODIFY `{col_name}` {col_type} NULL"
                        ))
                    except Exception as e:
                        logger.error("Could not relax leftover column %s: %s", col_name, e)
        return True
    except Exception as e:
        logger.error(f"skill_gap schema migration failed: {e}")
        return False


_schema_ensured = False


def _ensure_skillgap_schema_once() -> None:
    global _schema_ensured
    if _schema_ensured:
        return
    if _ensure_skillgap_schema():
        _schema_ensured = True


def SessionLocal():
    _ensure_skillgap_schema_once()
    return _SessionFactory()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()