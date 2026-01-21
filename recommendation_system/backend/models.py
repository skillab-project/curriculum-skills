# backend/models.py
# ------------------------------------------------------------
# SQLAlchemy ORM Models για το Academic Recommender System
# Περιλαμβάνει:
#   - University
#   - DegreeProgram
#   - Course
#   - Skill
#   - CourseSkill
#   - Occupation
#   - SkillOccupation
# ------------------------------------------------------------

from sqlalchemy import (
    Column, Integer, String, Text, Enum, JSON, ForeignKey, TIMESTAMP,
    UniqueConstraint, CheckConstraint, text
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# ============================================================
#  UNIVERSITY
# ============================================================
class University(Base):
    __tablename__ = "University"

    university_id = Column(Integer, primary_key=True)
    university_name = Column(String(255), nullable=False)
    country = Column(String(100), nullable=False)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("university_name", "country", name="uq_university"),
    )

    # Relationships
    programs = relationship(
        "DegreeProgram", back_populates="university", cascade="all, delete-orphan"
    )
    courses = relationship(
        "Course", back_populates="university", cascade="all, delete-orphan"
    )

# ============================================================
#  DEGREE PROGRAM
# ============================================================
class DegreeProgram(Base):
    __tablename__ = "DegreeProgram"

    program_id = Column(Integer, primary_key=True)
    university_id = Column(
        Integer, ForeignKey("University.university_id", ondelete="CASCADE"), nullable=False
    )
    degree_type = Column(
        Enum("BSc", "MSc", "PhD", "Other", name="degree_type_enum"), nullable=False
    )
    degree_titles = Column(JSON, nullable=True)
    language = Column(String(100))
    duration_semesters = Column(Text)
    total_ects = Column(Text)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    # Relationships
    university = relationship("University", back_populates="programs")
    courses = relationship(
        "Course", back_populates="program", cascade="all, delete-orphan"
    )

# ============================================================
#  COURSE
# ============================================================
class Course(Base):
    __tablename__ = "Course"

    course_id = Column(Integer, primary_key=True)
    university_id = Column(
        Integer, ForeignKey("University.university_id", ondelete="CASCADE"), nullable=False
    )
    program_id = Column(
        Integer, ForeignKey("DegreeProgram.program_id", ondelete="SET NULL"), nullable=True
    )
    lesson_name = Column(String(255), nullable=False)
    language = Column(Text)
    website = Column(Text)
    semester_number = Column(Text)
    semester_label = Column(Text)
    ects_list = Column(JSON, nullable=True)
    mand_opt_list = Column(JSON, nullable=True)
    msc_bsc_list = Column(JSON, nullable=True)
    fee_list = Column(JSON, nullable=True)
    hours = Column(Text)
    description = Column(Text)
    objectives = Column(Text)
    learning_outcomes = Column(Text)
    course_content = Column(Text)
    assessment = Column(Text)
    exam = Column(Text)
    prerequisites = Column(Text)
    general_competences = Column(Text)
    educational_material = Column(Text)
    attendance_type = Column(Text)
    professors = Column(JSON, nullable=True)
    extras = Column(JSON, nullable=True)
    degree_titles = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, nullable=True, server_onupdate=text("CURRENT_TIMESTAMP"))

    # Relationships
    university = relationship("University", back_populates="courses")
    program = relationship("DegreeProgram", back_populates="courses")
    skills = relationship(
        "CourseSkill", back_populates="course", cascade="all, delete-orphan"
    )

# ============================================================
#  SKILL
# ============================================================
class Skill(Base):
    __tablename__ = "Skill"

    skill_id = Column(Integer, primary_key=True)
    skill_name = Column(String(255), nullable=False)
    skill_url = Column(String(512)) 
    esco_id = Column(String(64))
    esco_level = Column(String(32))
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("skill_name", "skill_url", name="uq_skill"),
    )

    # Relationships
    courses = relationship("CourseSkill", back_populates="skill", cascade="all, delete-orphan")
    occupations = relationship("SkillOccupation", back_populates="skill", cascade="all, delete-orphan")

# ============================================================
#  COURSE ↔ SKILL (πολλα-προς-πολλά)
# ============================================================
class CourseSkill(Base):
    __tablename__ = "CourseSkill"

    course_id = Column(
        Integer, ForeignKey("Course.course_id", ondelete="CASCADE"), primary_key=True
    )
    skill_id = Column(
        Integer, ForeignKey("Skill.skill_id", ondelete="CASCADE"), primary_key=True
    )
    categories = Column(JSON, nullable=False)

    __table_args__ = (
        CheckConstraint("JSON_VALID(categories)", name="chk_json_valid_categories"),
        CheckConstraint("JSON_TYPE(categories) = 'ARRAY'", name="chk_json_type_categories"),
    )

    # Relationships
    course = relationship("Course", back_populates="skills")
    skill = relationship("Skill", back_populates="courses")

# ============================================================
#  OCCUPATION
# ============================================================
class Occupation(Base):
    __tablename__ = "Occupation"

    occupation_id = Column(String(64), primary_key=True)
    occupation_name = Column(String(255), nullable=False)
    occupation_url = Column(Text)
    esco_code = Column(String(64))
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))

    __table_args__ = (
        UniqueConstraint("occupation_name", "occupation_id", name="uq_occupation"),
    )

    skills = relationship("SkillOccupation", back_populates="occupation", cascade="all, delete-orphan")

# ============================================================
#  SKILL ↔ OCCUPATION (πολλα-προς-πολλά)
# ============================================================
class SkillOccupation(Base):
    __tablename__ = "SkillOccupation"

    skill_id = Column(Integer, ForeignKey("Skill.skill_id", ondelete="CASCADE"), primary_key=True)
    occupation_id = Column(String(64), ForeignKey("Occupation.occupation_id", ondelete="CASCADE"), primary_key=True)

    # Relationships
    skill = relationship("Skill", back_populates="occupations")
    occupation = relationship("Occupation", back_populates="skills")

from sqlalchemy import DateTime
from sqlalchemy.sql import func
from sqlalchemy import Float

class UserInteraction(Base):
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True, index=True)

    # Χωρίς authentication – απλό user_id (π.χ. 1)
    user_id = Column(Integer, index=True, nullable=False)

    # Course seen / clicked
    course_name = Column(String(255), index=True, nullable=False)

    # Skills snapshot (ΠΟΛΥ σημαντικό: default=list)
    skills = Column(JSON, default=list)

    interest_score = Column(Float, default=1.0)  
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
