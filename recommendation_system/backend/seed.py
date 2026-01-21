# backend/seed.py
# ============================================================
# Seed για Academic Recommender System με επιπλέον δεδομένα
# ============================================================

from sqlalchemy.orm import Session
from sqlalchemy import select
from recommendation_system.backend.database import SessionLocal, init_db
from recommendation_system.backend.models import (
    University, DegreeProgram, Course, Skill, CourseSkill, Occupation, SkillOccupation
)

# ============================================================
# HELPER
# ============================================================
def get_or_create(db: Session, model, unique_fields: dict, defaults: dict = None):
    """Βρίσκει ή δημιουργεί εγγραφή χωρίς duplicate errors"""
    stmt = select(model).filter_by(**unique_fields)
    instance = db.execute(stmt).scalar_one_or_none()
    if instance:
        if defaults:
            for k, v in defaults.items():
                setattr(instance, k, v)
        return instance
    else:
        data = {**unique_fields, **(defaults or {})}
        instance = model(**data)
        db.add(instance)
        return instance

# ============================================================
# SEED FUNCTION
# ============================================================
def run_seed():
    init_db()
    db: Session = SessionLocal()
    try:
        # --------------------------------------------------------
        # UNIVERSITIES
        # --------------------------------------------------------
        universities_data = [
            {"university_name": "Aristotle University of Thessaloniki", "country": "Greece"},
            {"university_name": "National and Kapodistrian University of Athens", "country": "Greece"},
            {"university_name": "University of Patras", "country": "Greece"},
            {"university_name": "Technical University of Munich", "country": "Germany"},
            {"university_name": "Politecnico di Milano", "country": "Italy"},
        ]
        universities = []
        for u in universities_data:
            uni = get_or_create(db, University, {"university_name": u["university_name"], "country": u["country"]})
            universities.append(uni)
        db.flush()

        # --------------------------------------------------------
        # DEGREE PROGRAMS
        # --------------------------------------------------------
        programs_data = [
            {"university": universities[0], "degree_type": "BSc", "degree_titles": ["Computer Science"], "language": "English", "duration_semesters": "8", "total_ects": "240"},
            {"university": universities[0], "degree_type": "MSc", "degree_titles": ["Data Science"], "language": "English", "duration_semesters": "4", "total_ects": "120"},
            {"university": universities[1], "degree_type": "BSc", "degree_titles": ["Informatics"], "language": "Greek", "duration_semesters": "8", "total_ects": "240"},
            {"university": universities[1], "degree_type": "MSc", "degree_titles": ["Artificial Intelligence"], "language": "English", "duration_semesters": "4", "total_ects": "120"},
            {"university": universities[2], "degree_type": "BSc", "degree_titles": ["Electrical & Computer Engineering"], "language": "Greek", "duration_semesters": "10", "total_ects": "300"},
            {"university": universities[2], "degree_type": "MSc", "degree_titles": ["Software Engineering"], "language": "English", "duration_semesters": "4", "total_ects": "120"},
            {"university": universities[3], "degree_type": "BSc", "degree_titles": ["Informatics"], "language": "English", "duration_semesters": "6", "total_ects": "180"},
            {"university": universities[3], "degree_type": "MSc", "degree_titles": ["Robotics, Cognition, Intelligence"], "language": "English", "duration_semesters": "4", "total_ects": "120"},
            {"university": universities[4], "degree_type": "BSc", "degree_titles": ["Computer Engineering"], "language": "English", "duration_semesters": "6", "total_ects": "180"},
            {"university": universities[4], "degree_type": "MSc", "degree_titles": ["Machine Learning & AI"], "language": "English", "duration_semesters": "4", "total_ects": "120"},
        ]
        programs = []
        for p in programs_data:
            prog = get_or_create(
                db, DegreeProgram,
                {"university_id": p["university"].university_id, "degree_type": p["degree_type"], "language": p["language"]},
                {"degree_titles": p["degree_titles"], "duration_semesters": p["duration_semesters"], "total_ects": p["total_ects"]}
            )
            programs.append(prog)
        db.flush()

        # --------------------------------------------------------
        # COURSES (Υποχρεωτικά)
        # --------------------------------------------------------
        courses_data = [
            {"lesson_name": "Programming I", "description": "Python intro", "program": programs[0], "language": "English", "semester_number": "1", "ects_list": [5]},
            {"lesson_name": "Data Structures", "description": "DSA", "program": programs[0], "language": "English", "semester_number": "2", "ects_list": [5]},
            {"lesson_name": "Machine Learning", "description": "ML core", "program": programs[1], "language": "English", "semester_number": "1", "ects_list": [5]},
            {"lesson_name": "Algorithms", "description": "Design & analysis", "program": programs[2], "language": "Greek", "semester_number": "2", "ects_list": [5]},
            {"lesson_name": "Computer Vision", "description": "CNNs & vision", "program": programs[3], "language": "English", "semester_number": "1", "ects_list": [5]},
            {"lesson_name": "Distributed Systems", "description": "Consistency & consensus", "program": programs[5], "language": "English", "semester_number": "2", "ects_list": [5]},
            {"lesson_name": "Parallel Computing", "description": "Threads, MPI, GPU", "program": programs[6], "language": "English", "semester_number": "3", "ects_list": [5]},
            {"lesson_name": "Robotics I", "description": "Kinematics & control", "program": programs[7], "language": "English", "semester_number": "1", "ects_list": [5]},
            {"lesson_name": "Recommender Systems", "description": "CF & content-based", "program": programs[9], "language": "English", "semester_number": "2", "ects_list": [5]},
        ]
        courses = []
        for c in courses_data:
            degree_type = c["program"].degree_type
            course = get_or_create(
                db, Course,
                {"lesson_name": c["lesson_name"], "university_id": c["program"].university_id},
                {
                    "program_id": c["program"].program_id,
                    "description": c["description"],
                    "language": c["language"],
                    "semester_number": c["semester_number"],
                    "ects_list": c["ects_list"],
                    "mand_opt_list": ["Mandatory"],
                    "msc_bsc_list": [degree_type],
                    
                }
            )
            courses.append(course)
        db.flush()

        # --------------------------------------------------------
        # EXTRA ELECTIVE COURSES
        # --------------------------------------------------------
        extra_courses_data = [
            {"lesson_name": "AI Ethics", "description": "Ethical considerations in AI", "program": programs[3], "language": "English", "semester_number": "2", "ects_list": [3], "website": "https://example.com/ai-ethics"},
            {"lesson_name": "Deep Learning", "description": "Neural networks and applications", "program": programs[1], "language": "English", "semester_number": "2", "ects_list": [5], "website": "https://example.com/deep-learning"},
            {"lesson_name": "Natural Language Processing", "description": "Text mining and NLP techniques", "program": programs[9], "language": "English", "semester_number": "2", "ects_list": [5], "website": "https://example.com/nlp"},
        ]
        for c in extra_courses_data:
            degree_type = c["program"].degree_type
            course = get_or_create(
            db, Course,
            {"lesson_name": c["lesson_name"], "university_id": c["program"].university_id},
            {
            "program_id": c["program"].program_id,
            "description": c["description"],
            "language": c["language"],
            "semester_number": c["semester_number"],
            "ects_list": c["ects_list"],
            "mand_opt_list": ["Optional"],
            "msc_bsc_list": [degree_type],
            "website": c.get("website", "")
            }
            )
            courses.append(course)
        db.flush()

        # --------------------------------------------------------
        # SKILLS
        # --------------------------------------------------------
        skills_data = [
            {"skill_name": "Python", "skill_url": "https://esco.example/python", "esco_id": "ESCO:PY", "esco_level": "4"},
            {"skill_name": "Data Structures", "skill_url": "https://esco.example/data-structures", "esco_id": "ESCO:DS", "esco_level": "5"},
            {"skill_name": "Algorithms", "skill_url": "https://esco.example/algorithms", "esco_id": "ESCO:ALG", "esco_level": "5"},
            {"skill_name": "Machine Learning", "skill_url": "https://esco.example/ml", "esco_id": "ESCO:ML", "esco_level": "6"},
            {"skill_name": "Computer Vision", "skill_url": "https://esco.example/cv", "esco_id": "ESCO:CV", "esco_level": "6"},
            {"skill_name": "Distributed Systems", "skill_url": "https://esco.example/distributed", "esco_id": "ESCO:DSYS", "esco_level": "5"},
            {"skill_name": "Robotics", "skill_url": "https://esco.example/robotics", "esco_id": "ESCO:ROB", "esco_level": "6"},
            {"skill_name": "Recommender Systems", "skill_url": "https://esco.example/recommenders", "esco_id": "ESCO:RS", "esco_level": "6"},
            {"skill_name": "AI Ethics", "skill_url": "https://esco.example/ai-ethics", "esco_id": "ESCO:AIE", "esco_level": "4"},
            {"skill_name": "Deep Learning", "skill_url": "https://esco.example/deep-learning", "esco_id": "ESCO:DL", "esco_level": "6"},
            {"skill_name": "NLP", "skill_url": "https://esco.example/nlp", "esco_id": "ESCO:NLP", "esco_level": "6"},
        ]
        skills = []
        for s in skills_data:
            skill = get_or_create(db, Skill, {"skill_name": s["skill_name"], "skill_url": s["skill_url"]}, s)
            skills.append(skill)
        db.flush()

        # --------------------------------------------------------
        # COURSE ↔ SKILL
        # --------------------------------------------------------
        course_skill_links = [
            ("Programming I", "Python", ["programming", "basics"]),
            ("Data Structures", "Data Structures", ["theory", "coding"]),
            ("Algorithms", "Algorithms", ["algorithm design"]),
            ("Machine Learning", "Machine Learning", ["supervised", "unsupervised"]),
            ("Computer Vision", "Computer Vision", ["cnn", "image processing"]),
            ("Distributed Systems", "Distributed Systems", ["consensus", "scalability"]),
            ("Robotics I", "Robotics", ["kinematics", "control"]),
            ("Recommender Systems", "Recommender Systems", ["collaborative filtering"]),
            ("AI Ethics", "AI Ethics", ["ethics", "policy"]),
            ("Deep Learning", "Deep Learning", ["neural networks", "cnn"]),
            ("Natural Language Processing", "NLP", ["text", "language", "NLP"]),
        ]
        for cname, sname, cats in course_skill_links:
            course = next(c for c in courses if c.lesson_name == cname)
            skill = next(s for s in skills if s.skill_name == sname)
            exists = db.execute(
                select(CourseSkill).where(CourseSkill.course_id == course.course_id, CourseSkill.skill_id == skill.skill_id)
            ).scalar_one_or_none()
            if not exists:
                db.add(CourseSkill(course_id=course.course_id, skill_id=skill.skill_id, categories=cats))
        db.flush()

        # --------------------------------------------------------
        # OCCUPATIONS
        # --------------------------------------------------------
        occupations_data = [
            {"occupation_id": "O-001", "occupation_name": "Data Scientist", "occupation_url": "https://esco.example/occupations/data-scientist", "esco_code": "1234.1"},
            {"occupation_id": "O-002", "occupation_name": "Software Engineer", "occupation_url": "https://esco.example/occupations/software-engineer", "esco_code": "2512.0"},
            {"occupation_id": "O-003", "occupation_name": "ML Engineer", "occupation_url": "https://esco.example/occupations/ml-engineer", "esco_code": "2511.5"},
        ]
        occupations = []
        for o in occupations_data:
            occ = get_or_create(db, Occupation, {"occupation_id": o["occupation_id"]}, o)
            occupations.append(occ)
        db.flush()

        # --------------------------------------------------------
        # SKILL ↔ OCCUPATION
        # --------------------------------------------------------
        skill_occupation_links = [
            ("Machine Learning", "O-001"),
            ("Algorithms", "O-002"),
            ("Distributed Systems", "O-002"),
            ("Machine Learning", "O-003"),
        ]
        for sname, oid in skill_occupation_links:
            skill = next(s for s in skills if s.skill_name == sname)
            exists = db.execute(
                select(SkillOccupation).where(SkillOccupation.skill_id == skill.skill_id, SkillOccupation.occupation_id == oid)
            ).scalar_one_or_none()
            if not exists:
                db.add(SkillOccupation(skill_id=skill.skill_id, occupation_id=oid))

        # --------------------------------------------------------
        db.commit()
        print("✅ Seed completed successfully.")
    except Exception as e:
        db.rollback()
        print(f"❌ Seed failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
