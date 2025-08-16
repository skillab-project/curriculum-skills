import os
import json
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict
from fuzzywuzzy import process, fuzz

from output import (
    _norm,
    _score,
    _find_uni_by_name,
    _load_universities
)

try:
    import orjson as _jsonlib
    def _json_load(f): return _jsonlib.loads(f.read())
except Exception:
    import json as _jsonlib
    def _json_load(f): return _jsonlib.load(f)

import mysql.connector
import re

aliases: DefaultDict[str, List[str]] = defaultdict(list)


def is_database_connected(db_config: Dict[str, Any]) -> bool:
    try:
        conn = mysql.connector.connect(**db_config)
        ok = conn.is_connected()
        conn.close()
        return ok
    except mysql.connector.Error:
        return False


def _json_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    return json.dumps(v, ensure_ascii=False)


def _ensure_list(v: Any) -> Optional[List[Any]]:
    if v is None:
        return None
    if isinstance(v, list):
        return v
    return [v]

_num = re.compile(r"(\d+(?:[.,]\d+)?)")

def _to_decimal(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = _num.search(val)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                return None
    return None
    
    
def upsert_skill_and_link_with_categories(_conn, course_id: int,
                                          skill_url: Optional[str],
                                          skill_name: str,
                                          esco_id: Optional[str],
                                          esco_level: Optional[str],
                                          categories: List[str]):
    """
    Upsert Skill including esco_id + esco_level, then upsert CourseSkill with merged categories.
    Prefer identifying a skill by URL; if not present, fall back to name.
    """
    cur = _conn.cursor()
    try:
        skill_id = None
        if skill_url:
            cur.execute("SELECT skill_id FROM Skill WHERE skill_url = %s LIMIT 1", (skill_url,))
            row = cur.fetchone()
            if row:
                skill_id = row[0]
                cur.execute("""
                    UPDATE Skill
                       SET skill_name = COALESCE(skill_name, %s),
                           esco_id    = COALESCE(esco_id, %s),
                           esco_level = COALESCE(esco_level, %s)
                     WHERE skill_id = %s
                """, (skill_name, esco_id, esco_level, skill_id))

        if skill_id is None:
            cur.execute("SELECT skill_id FROM Skill WHERE skill_url IS NULL AND skill_name = %s LIMIT 1", (skill_name,))
            row = cur.fetchone()
            if row:
                skill_id = row[0]
                cur.execute("""
                    UPDATE Skill
                       SET skill_url  = COALESCE(skill_url, %s),
                           esco_id    = COALESCE(esco_id, %s),
                           esco_level = COALESCE(esco_level, %s)
                     WHERE skill_id = %s
                """, (skill_url, esco_id, esco_level, skill_id))

        if skill_id is None:
            cur.execute("""
                INSERT INTO Skill (skill_name, skill_url, esco_id, esco_level)
                VALUES (%s, %s, %s, %s)
            """, (skill_name, skill_url, esco_id, esco_level))
            skill_id = cur.lastrowid

        cur.execute("SELECT categories FROM CourseSkill WHERE course_id = %s AND skill_id = %s", (course_id, skill_id))
        ex = cur.fetchone()
        if ex:
            try:
                merged = sorted(set((json.loads(ex[0] or "[]")) + (categories or [])))
            except Exception:
                merged = sorted(set(categories or []))
            cur.execute("UPDATE CourseSkill SET categories = %s WHERE course_id = %s AND skill_id = %s",
                        (json.dumps(merged), course_id, skill_id))
        else:
            cur.execute("INSERT INTO CourseSkill (course_id, skill_id, categories) VALUES (%s, %s, %s)",
                        (course_id, skill_id, json.dumps(sorted(set(categories or [])))))
        _conn.commit()
    finally:
        cur.close()



def _opt_str(val):
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None

def _as_list(val):
    """Return a list of strings, allowing input to be a single item, list, set, or None."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        out = [str(x).strip() for x in val if x is not None and str(x).strip()]
    else:
        s = str(val).strip()
        out = [s] if s else []
    return out
    
def _insert_degree_program(cur, *, university_id: int, dp: dict) -> int:
    vals = _sanitize_program_for_db(dp)
    cur.execute("""
        INSERT INTO DegreeProgram
          (university_id, degree_type, degree_titles, language, duration_semesters, total_ects)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        university_id,
        vals["degree_type"],
        json.dumps(vals["degree_titles"]) if vals["degree_titles"] else None,
        vals["language"],
        vals["duration_semesters"],
        vals["total_ects"],
    ))
    return cur.lastrowid

def _insert_course(cur, university_id: int, lesson_name: str, program_id: Optional[int], c: dict) -> int:
    vals = _sanitize_course_for_db(c)
    cur.execute("""
        INSERT INTO Course
          (university_id, program_id, lesson_name, language, website,
           semester_number, semester_label, ects_list, mand_opt_list, msc_bsc_list, fee_list, hours,
           description, objectives, learning_outcomes, course_content, assessment, exam, prerequisites,
           general_competences, educational_material, attendance_type, professors, extras, degree_titles)
        VALUES
          (%s,%s,%s,%s,%s,
           %s,%s,%s,%s,%s,%s,%s,
           %s,%s,%s,%s,%s,%s,%s,
           %s,%s,%s,%s,%s,%s)
    """, (
        university_id, program_id, lesson_name, vals["language"], vals["website"],
        vals["semester_number"], vals["semester_label"],
        json.dumps(vals["ects_list"]) if vals["ects_list"] else None,
        json.dumps(vals["mand_opt_list"]) if vals["mand_opt_list"] else None,
        json.dumps(vals["msc_bsc_list"]) if vals["msc_bsc_list"] else None,
        json.dumps(vals["fee_list"]) if vals["fee_list"] else None,
        vals["hours"],
        vals["description"], vals["objectives"], vals["learning_outcomes"], vals["course_content"],
        vals["assessment"], vals["exam"], vals["prerequisites"],
        vals["general_competences"], vals["educational_material"],
        vals["attendance_type"],
        json.dumps(vals["professors"]) if vals["professors"] else None,
        json.dumps(vals["extras"]) if vals["extras"] else None,
        json.dumps(vals["degree_titles"]) if vals["degree_titles"] else None,
    ))
    return cur.lastrowid


def _sanitize_program_for_db(dp: dict) -> dict:
    return {
        "degree_type":        _opt_str(dp.get("degree_type")) or "Other",
        "degree_titles":      _as_list(dp.get("degree_titles") or dp.get("degree_title")),
        "language":           _opt_str(dp.get("language")),
        "duration_semesters": _opt_str(dp.get("duration_semesters") or dp.get("semesters") or dp.get("duration")),
        "total_ects":         _opt_str(dp.get("total_ects") or dp.get("ects") or dp.get("credits")),
    }

def _sanitize_course_for_db(c: dict) -> dict:
    return {
        "language":             _opt_str(c.get("language")),
        "website":              _opt_str(c.get("website") or c.get("url")),
        "semester_number":      _opt_str(c.get("semester_number") or c.get("semester")),
        "semester_label":       _opt_str(c.get("semester_label")),
        "ects_list":            _as_list(c.get("ects_list") or c.get("ects")),
        "mand_opt_list":        _as_list(c.get("mand_opt_list") or c.get("mand_opt")),
        "msc_bsc_list":         _as_list(c.get("msc_bsc_list") or c.get("msc_bsc")),
        "fee_list":             _as_list(c.get("fee_list") or c.get("fee")),
        "hours":                _opt_str(c.get("hours")),
        "description":          _opt_str(c.get("description")),
        "objectives":           _opt_str(c.get("objectives")),
        "learning_outcomes":    _opt_str(c.get("learning_outcomes")),
        "course_content":       _opt_str(c.get("course_content")),
        "assessment":           _opt_str(c.get("assessment")),
        "exam":                 _opt_str(c.get("exam")),
        "prerequisites":        _opt_str(c.get("prerequisites")),
        "general_competences":  _opt_str(c.get("general_competences")),
        "educational_material": _opt_str(c.get("educational_material")),
        "attendance_type":      _opt_str(c.get("attendance_type") or c.get("attendence_type")),
        "professors":           _as_list(c.get("professors") or c.get("professor")),
        "extras":               (c.get("extras") if isinstance(c.get("extras"), dict) else None),
        "degree_titles":        _as_list(c.get("degree_titles") or c.get("degree_title")),
    }

def _course_from_curriculnlp_labels_payload(payload: Any, filename_hint: str = "") -> Optional[Dict[str, Any]]:
    """
    Build a Course-row dict from a CurricuNLP/Label Studio payload that contains `labels`.
    Expected label names:
      lesson_name, professor, ects, semester, hours, assessment, exam, description, objectives,
      learning_outcomes, mand_opt, year, department, msc_bsc, university, website, language,
      prerequisites, general_competences, course_content, educational_material, attendence_type, fee
    """
    labels = payload["labels"] if isinstance(payload, dict) else payload
    if not isinstance(labels, list) or not labels:
        return None

    from collections import defaultdict as _dd
    by = _dd(list)
    for it in labels:
        try:
            lbl = (it.get("class_or_confidence") or "").strip().lower()
            tok = (it.get("token") or "").strip()
            if lbl and tok:
                by[lbl].append(tok)
        except Exception:
            continue

    def _first(lbl, default=None):
        vals = by.get(lbl, [])
        return vals[0] if vals else default

    def _join_unique(lbl):
        vals = by.get(lbl, [])
        if not vals: return ""
        return "\n\n".join(dict.fromkeys(v for v in vals if v.strip()))

    import re as _re
    def _num_from_list(lbl):
        for s in by.get(lbl, []):
            m = _re.search(r"(\d+(?:\.\d+)?)", s or "")
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass
        return None

    course: Dict[str, Any] = {
        "lesson_name": _first("lesson_name") or payload.get("lesson_name") or payload.get("title") or filename_hint or "Untitled Course",
        "website": _first("website") or payload.get("website") or payload.get("url"),
        "language": _first("language"),
        "msc_bsc": _first("msc_bsc"),
        "degree_title": None,  
        "ects": _num_from_list("ects"),
        "hours": _num_from_list("hours"),
        "semester_label": _first("semester"),
        "semester_number": None,
        "description": _join_unique("description"),
        "objectives": _join_unique("objectives"),
        "learning_outcomes": _join_unique("learning_outcomes"),
        "course_content": _join_unique("course_content"),
        "assessment": _join_unique("assessment"),
        "exam": _join_unique("exam"),
        "prerequisites": _join_unique("prerequisites"),
        "general_competences": _join_unique("general_competences"),
        "educational_material": _join_unique("educational_material"),
        "attendance_type": _first("attendence_type"),
    }

    if course["semester_label"]:
        m = _re.search(r"\b(\d+)\b", course["semester_label"])
        if m:
            try:
                course["semester_number"] = int(m.group(1))
            except Exception:
                pass

    def _jsonify_list(lbl):
        vals = [v for v in by.get(lbl, []) if isinstance(v, str) and v.strip()]
        return json.dumps(sorted(set(vals))) if vals else None

    course["professors"] = _jsonify_list("professor")
    course["msc_bsc_list"] = _jsonify_list("msc_bsc")
    course["degree_titles"] = _jsonify_list("degree_title")
    course["ects_list"] = _jsonify_list("ects")
    course["mand_opt_list"] = _jsonify_list("mand_opt")
    course["fee_list"] = _jsonify_list("fee")


    known = {
        "lesson_name","professor","ects","semester","hours","assessment","exam","description","objectives",
        "learning_outcomes","mand_opt","year","department","msc_bsc","university","website","language",
        "prerequisites","general_competences","course_content","educational_material","attendence_type","fee",
        "degree_title"
    }
    extras = {}
    if by.get("year"): extras["year"] = by["year"]
    if by.get("department"): extras["department"] = by["department"]
    if by.get("university"): extras["university"] = by["university"]
    for k, v in by.items():
        if k not in known:
            extras[k] = v
    if extras:
        course["extras"] = json.dumps(extras)

    course["lesson_name"] = str(course["lesson_name"])[:255]
    if not course.get("msc_bsc"):
        inferred = _infer_msc_bsc_from_text(course["lesson_name"])
        if inferred:
            course["msc_bsc"] = inferred
            try:
                cur = set(json.loads(course.get("msc_bsc_list") or "[]"))
            except Exception:
                cur = set()
            cur.add(inferred)
            course["msc_bsc_list"] = json.dumps(sorted(cur))

    return course


LABEL_TO_FIELD = {
    "lesson_name": "lesson_name",
    "description": "description",
    "objectives": "objectives",
    "learning_outcomes": "learning_outcomes",
    "course_content": "course_content",
    "assessment": "assessment",
    "exam": "exam",
    "prerequisites": "prerequisites",
    "general_competences": "general_competences",
    "educational_material": "educational_material",
    "language": "language",
    "website": "website",
    "hours": "hours",
    "semester": "semester_label",
    "ects": "ects_list",
    "mand_opt": "mand_opt_list",
    "msc_bsc": "msc_bsc_list",
    "attendence_type": "attendance_type",
    "professor": "professors",
    "department": "extras",
    "fee": "fee_list",
    "year": "extras",
    "university": "extras",
}

def _labels_to_course(lesson_name: Optional[str], website: Optional[str], labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    course: Dict[str, Any] = {}
    if lesson_name: course["lesson_name"] = lesson_name[:255]
    if website: course["website"] = website
    extras_obj: Dict[str, Any] = {}

    def add_to(field: str, value: Any):
        if value in (None, "", [], {}): return
        if field.endswith("_list"):
            course[field] = (course.get(field) or []) + [value]
        elif field in ("professors", "degree_titles"):
            course[field] = (course.get(field) or []) + [value]
        else:
            prev = course.get(field)
            if isinstance(prev, str):
                course[field] = (prev + "\n\n" + str(value)).strip()
            elif prev is None:
                course[field] = value
            else:
                course[field] = value

    for it in labels or []:
        label = (it.get("label") or it.get("value") or "").strip().lower()
        text  = (it.get("text") or it.get("span") or "").strip()
        if not label or not text:
            continue
        field = LABEL_TO_FIELD.get(label)
        if not field:
            continue
        if field == "extras":
            extras_obj[label] = text
        else:
            add_to(field, text)

    if extras_obj:
        existing = course.get("extras")
        if isinstance(existing, dict):
            existing.update(extras_obj)
        else:
            course["extras"] = extras_obj
    return course


def _normalize_text_for_ner(text: str, max_chars: int = 40000) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    return t[:max_chars]



def _build_course_extras(course: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    known = {
        "lesson_name", "title", "language", "website",
        "semester_number", "semester_label",
        "ects_list", "mand_opt_list", "msc_bsc_list", "fee_list",
        "hours",
        "description", "objectives", "learning_outcomes", "course_content",
        "assessment", "exam", "prerequisites", "general_competences",
        "educational_material", "attendance_type",
        "professors", "extras", "degree_titles",
        "skills"
    }
    base = course.get("extras")
    if base is not None and not isinstance(base, dict):
        base = {"extras_raw": base}
    if base is None:
        base = {}
    for k, v in course.items():
        if k not in known:
            base[k] = v
    return base or None


def _get_or_create_university(cur, university_name: str, country: str) -> int:
    cur.execute(
        """
        SELECT university_id
        FROM University
        WHERE university_name = %s AND country = %s
        LIMIT 1
        """,
        (university_name, country),
    )
    row = cur.fetchone()
    if row:
        return int(row[0])
    cur.execute(
        """
        INSERT INTO University (university_name, country)
        VALUES (%s, %s)
        """,
        (university_name, country or "Unknown"),
    )
    return int(cur.lastrowid)


def _insert_degree_program(
    cur,
    university_id: int,
    degree_type: str,
    degree_titles: Optional[List[str]] = None,
    language: Optional[str] = None,
    duration_semesters: Optional[int] = None,
    total_ects: Optional[int] = None,
) -> int:
    cur.execute(
        """
        INSERT INTO DegreeProgram
            (university_id, degree_type, degree_titles, language, duration_semesters, total_ects)
        VALUES
            (%s, %s, %s, %s, %s, %s)
        """,
        (
            university_id,
            degree_type or "Other",
            _json_or_none(degree_titles or []),
            language,
            duration_semesters,
            total_ects,
        ),
    )
    return int(cur.lastrowid)


def _insert_course(
    cur,
    university_id: int,
    lesson_name: str,
    program_id: Optional[int],
    course: Dict[str, Any],
) -> int:
    ects = _ensure_list(course.get("ects_list"))
    mand_opt = _ensure_list(course.get("mand_opt_list"))
    msc_bsc = _ensure_list(course.get("msc_bsc_list"))
    fees = _ensure_list(course.get("fee_list"))
    professors = course.get("professors")
    if professors is not None and not isinstance(professors, list):
        professors = [professors]
    extras = _build_course_extras(course)
    degree_titles = _ensure_list(course.get("degree_titles"))

    cur.execute(
        """
        INSERT INTO Course (
            university_id, program_id, lesson_name, language, website,
            semester_number, semester_label,
            ects_list, mand_opt_list, msc_bsc_list, fee_list,
            hours,
            description, objectives, learning_outcomes, course_content, assessment, exam,
            prerequisites, general_competences, educational_material, attendance_type,
            professors, extras, degree_titles
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s,
            %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s
        )
        """,
        (
            university_id,
            program_id,
            lesson_name,
            course.get("language"),
            course.get("website"),
            course.get("semester_number"),
            course.get("semester_label"),
            _json_or_none(ects),
            _json_or_none(mand_opt),
            _json_or_none(msc_bsc),
            _json_or_none(fees),
            course.get("hours"),
            course.get("description"),
            course.get("objectives"),
            course.get("learning_outcomes"),
            course.get("course_content"),
            course.get("assessment"),
            course.get("exam"),
            course.get("prerequisites"),
            course.get("general_competences"),
            course.get("educational_material"),
            course.get("attendance_type"),
            _json_or_none(professors),
            _json_or_none(extras),
            _json_or_none(degree_titles),
        ),
    )
    return int(cur.lastrowid)


def _upsert_skill(
    cur,
    name: Optional[str],
    url: Optional[str],
    esco_id: Optional[str],
    esco_level: Optional[str]
) -> int:
    url = (url or "").strip() or None
    name = (name or "").strip() or None
    if url:
        cur.execute("SELECT skill_id FROM Skill WHERE skill_url = %s LIMIT 1", (url,))
        row = cur.fetchone()
        if row:
            skill_id = int(row[0])
            cur.execute(
                """
                UPDATE Skill
                   SET skill_name = COALESCE(skill_name, %s),
                       esco_id = COALESCE(esco_id, %s),
                       esco_level = COALESCE(esco_level, %s)
                 WHERE skill_id = %s
                """,
                (name, esco_id, esco_level, skill_id),
            )
            return skill_id
        cur.execute(
            """
            INSERT INTO Skill (skill_name, skill_url, esco_id, esco_level)
            VALUES (%s, %s, %s, %s)
            """,
            (name, url, esco_id, esco_level),
        )
        return int(cur.lastrowid)
    if name:
        cur.execute("SELECT skill_id FROM Skill WHERE skill_name = %s LIMIT 1", (name,))
        row = cur.fetchone()
        if row:
            return int(row[0])
        cur.execute(
            "INSERT INTO Skill (skill_name, esco_id, esco_level) VALUES (%s, %s, %s)",
            (name, esco_id, esco_level),
        )
        return int(cur.lastrowid)
    cur.execute("INSERT INTO Skill (skill_name) VALUES (%s)", ("Unknown Skill",))
    return int(cur.lastrowid)


def _merge_categories(existing: Optional[List[str]], new: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for src in (existing or []):
        s = str(src).strip()
        if s and s not in seen:
            out.append(s); seen.add(s)
    for src in (new or []):
        s = str(src).strip()
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out


def _link_course_skill(cur, course_id: int, skill_id: int, categories: Optional[List[str]] = None) -> None:
    cur.execute(
        "SELECT categories FROM CourseSkill WHERE course_id = %s AND skill_id = %s",
        (course_id, skill_id),
    )
    row = cur.fetchone()
    if row:
        try:
            existing = json.loads(row[0]) if row[0] else []
        except Exception:
            existing = []
        merged = _merge_categories(existing, categories)
        cur.execute(
            "UPDATE CourseSkill SET categories = %s WHERE course_id = %s AND skill_id = %s",
            (_json_or_none(merged), course_id, skill_id),
        )
    else:
        cur.execute(
            "INSERT INTO CourseSkill (course_id, skill_id, categories) VALUES (%s, %s, %s)",
            (course_id, skill_id, _json_or_none(categories or [])),
        )


def _iter_course_skills(raw: Any) -> List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[List[str]]]]:
    out: List[Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[List[str]]]] = []
    if not raw:
        return out
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict):
                cats = it.get("categories")
                if isinstance(cats, str):
                    cats = [cats]
                out.append(
                    (
                        it.get("skill_name"),
                        it.get("skill_url"),
                        it.get("esco_id"),
                        it.get("esco_level"),
                        cats if isinstance(cats, list) else None,
                    )
                )
    elif isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                cats = v.get("categories")
                if isinstance(cats, str):
                    cats = [cats]
                out.append((v.get("skill_name"), v.get("skill_url") or k, v.get("esco_id"), v.get("esco_level"), cats if isinstance(cats, list) else None))
            else:
                out.append((None, k, None, None, None))
    return out


def write_json_to_database(data: Dict[str, Any], db_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lenient writer:
      - Uses SAVEPOINTs to isolate failures per program/course/skill.
      - Leaves numeric-ish fields as text (no coercion).
      - Skips only the failing row and continues.
    """
    conn = mysql.connector.connect(**db_config)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        uni_name = (data.get("university_name") or data.get("university") or "").strip()
        country  = (data.get("country") or "Unknown").strip()
        if not uni_name:
            raise ValueError("university_name is required in JSON payload")

        university_id = _get_or_create_university(cur, uni_name, country)

        created_program_ids: List[int] = []
        created_course_ids:  List[int] = []
        created_skill_ids:   List[int] = []

        saved_programs = skipped_programs = 0
        saved_courses  = skipped_courses  = 0
        errors: List[Dict[str, Any]] = []

        degree_programs = data.get("degree_programs") or data.get("programs") or []
        if isinstance(degree_programs, list):
            for p_idx, dp in enumerate(degree_programs):
                if not isinstance(dp, dict):
                    continue
                cur.execute("SAVEPOINT sp_program")
                try:
                    dp = _sanitize_program_for_db(dp)
                    program_id = _insert_degree_program(
                        cur,
                        university_id=university_id,
                        degree_type=str(dp.get("degree_type") or "Other"),
                        degree_titles=_ensure_list(dp.get("degree_titles")),
                        language=dp.get("language"),
                        duration_semesters=dp.get("duration_semesters"),
                        total_ects=dp.get("total_ects"),
                    )
                    created_program_ids.append(program_id)
                    saved_programs += 1
                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_program")
                    skipped_programs += 1
                    errors.append({
                        "scope": "program",
                        "index": p_idx,
                        "degree_titles": _ensure_list(dp.get("degree_titles")),
                        "error": str(e)
                    })
                    continue

                dp_courses = dp.get("courses") or []
                if isinstance(dp_courses, list):
                    for c_idx, course in enumerate(dp_courses):
                        if not isinstance(course, dict):
                            continue
                        lesson_name = str(course.get("lesson_name") or course.get("title") or "").strip()
                        if not lesson_name:
                            skipped_courses += 1
                            errors.append({
                                "scope": "course",
                                "program_id": program_id,
                                "index": c_idx,
                                "error": "missing lesson_name"
                            })
                            continue

                        cur.execute("SAVEPOINT sp_course")
                        try:
                            c2 = _sanitize_course_for_db(course)
                            course_id = _insert_course(cur, university_id, lesson_name, program_id, c2)
                            created_course_ids.append(course_id)
                            saved_courses += 1

                            for s_name, s_url, s_esco_id, s_level, s_cats in _iter_course_skills(c2.get("skills")):
                                cur.execute("SAVEPOINT sp_skill")
                                try:
                                    sid = _upsert_skill(cur, s_name, s_url, s_esco_id, s_level)
                                    created_skill_ids.append(sid)
                                    _link_course_skill(cur, course_id, sid, s_cats)
                                except Exception as se:
                                    cur.execute("ROLLBACK TO SAVEPOINT sp_skill")
                                    errors.append({
                                        "scope": "skill",
                                        "lesson_name": lesson_name,
                                        "skill_name": s_name,
                                        "skill_url": s_url,
                                        "error": str(se)
                                    })
                        except Exception as ce:
                            cur.execute("ROLLBACK TO SAVEPOINT sp_course")
                            skipped_courses += 1
                            errors.append({
                                "scope": "course",
                                "program_id": program_id,
                                "lesson_name": lesson_name,
                                "error": str(ce)
                            })

        root_courses = data.get("courses") or []
        if isinstance(root_courses, list):
            for c_idx, course in enumerate(root_courses):
                if not isinstance(course, dict):
                    continue
                lesson_name = str(course.get("lesson_name") or course.get("title") or "").strip()
                if not lesson_name:
                    skipped_courses += 1
                    errors.append({"scope": "course", "index": c_idx, "error": "missing lesson_name"})
                    continue

                cur.execute("SAVEPOINT sp_course")
                try:
                    c2 = _sanitize_course_for_db(course)
                    course_id = _insert_course(cur, university_id, lesson_name, None, c2)
                    created_course_ids.append(course_id)
                    saved_courses += 1

                    for s_name, s_url, s_esco_id, s_level, s_cats in _iter_course_skills(c2.get("skills")):
                        cur.execute("SAVEPOINT sp_skill")
                        try:
                            sid = _upsert_skill(cur, s_name, s_url, s_esco_id, s_level)
                            created_skill_ids.append(sid)
                            _link_course_skill(cur, course_id, sid, s_cats)
                        except Exception as se:
                            cur.execute("ROLLBACK TO SAVEPOINT sp_skill")
                            errors.append({
                                "scope": "skill",
                                "lesson_name": lesson_name,
                                "skill_name": s_name,
                                "skill_url": s_url,
                                "error": str(se)
                            })
                except Exception as ce:
                    cur.execute("ROLLBACK TO SAVEPOINT sp_course")
                    skipped_courses += 1
                    errors.append({"scope": "course", "lesson_name": lesson_name, "error": str(ce)})

        conn.commit()
        return {
            "university_id": university_id,
            "created_program_ids": created_program_ids,
            "created_course_ids": created_course_ids,
            "created_skill_ids": sorted(set(created_skill_ids)),
            "saved_programs": saved_programs,
            "skipped_programs": skipped_programs,
            "saved_courses": saved_courses,
            "skipped_courses": skipped_courses,
            "errors": errors,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass



def write_json_file_to_database(json_path: str, db_config: Dict[str, Any]) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON must be an object")
    return write_json_to_database(payload, db_config)


def write_json_dir_to_database(dir_path: str, db_config: Dict[str, Any]) -> Dict[str, Any]:
    results = {"ok": [], "errors": {}}
    for name in os.listdir(dir_path):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(dir_path, name)
        try:
            out = write_json_file_to_database(path, db_config)
            results["ok"].append({"file": name, **out})
        except Exception as e:
            results["errors"][name] = str(e)
    return results
    
    
UNI_WORDS = {
    "university","universitat","universite","universite","universita","universidad",
    "universidade","universiteit","universitet","universitatea","uni"
}

LONG_TEXT_FIELDS = [
    "description","objectives","learning_outcomes","course_content",
    "assessment","exam","prerequisites","general_competences","educational_material"
]

LISTY_FIELD_HINTS = {
    "ects","msc_bsc","msc_bsc_list","degree_titles","languages","language_list",
    "professor","professors","labels","websites"
}

def _norm_text(s: Optional[str]) -> str:
    if not s: return ""
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _strip_uni_from_title(title: str, university_name: Optional[str]) -> str:
    t = _norm_text(title)
    if not t: return t
    u = _norm_text(university_name or "")
    if u:
        t = re.sub(re.escape(u), " ", t, flags=re.I)
        parts = [p for p in re.split(r"[^A-Za-zA-y]+", u) if len(p) > 2]
        for w in parts + list(UNI_WORDS):
            t = re.sub(rf"\b{re.escape(w)}\b", " ", t, flags=re.I)
    t = re.sub(r"\b(of|the|at|de|di|der|des|von|fur|dla|dla|da)\b", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" -:|,.;")
    return t or title

def _canonical_key(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _best_match(key: str, existing_keys: List[str], threshold: int = 88) -> Optional[str]:
    if not existing_keys: return None
    best, score = process.extractOne(key, existing_keys, scorer=fuzz.token_set_ratio)
    return best if score >= threshold else None

def _merge_text(a: Optional[str], b: Optional[str]) -> str:
    aa, bb = _norm_text(a), _norm_text(b)
    if not aa: return bb
    if not bb: return aa
    if aa in bb: return bb
    if bb in aa: return aa
    return (aa + "\n\n" + bb)

def _merge_list(a: Any, b: Any) -> List[Any]:
    la = a if isinstance(a, list) else ([a] if a not in (None, "", []) else [])
    lb = b if isinstance(b, list) else ([b] if b not in (None, "", []) else [])
    out = []
    seen = set()
    for x in la + lb:
        key = json.dumps(x, sort_keys=True, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out

def _deep_merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge_dict(out[k], v)
        elif k in out and out[k] != v:
            if isinstance(out[k], str) and isinstance(v, str):
                chosen = out[k] if len(out[k]) >= len(v) else v
                out[k] = chosen
                out.setdefault("__conflicts__", {})[k] = [a, b]
            else:
                out.setdefault("__conflicts__", {})[k] = [out[k], v]
        else:
            out[k] = v
    return out

def _merge_labels(a: Any, b: Any) -> List[Dict[str, Any]]:
    la = a if isinstance(a, list) else []
    lb = b if isinstance(b, list) else []
    seen = set()
    out = []
    for it in la + lb:
        if not isinstance(it, dict): continue
        key = (it.get("class_or_confidence"), it.get("token"))
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def _merge_course_records(base: Dict[str, Any], nxt: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in nxt.items():
        if k == "lesson_name":
            a, b = _norm_text(out.get("lesson_name")), _norm_text(v)
            if not a or (b and len(b) > len(a)):
                out["lesson_name"] = b or a
            continue

        if k in LONG_TEXT_FIELDS:
            out[k] = _merge_text(out.get(k), v)
        elif k in ("labels",):
            out[k] = _merge_labels(out.get(k), v)
        elif isinstance(v, list) or k in LISTY_FIELD_HINTS:
            out[k] = _merge_list(out.get(k), v)
        elif isinstance(v, dict):
            out[k] = _deep_merge_dict(out.get(k) or {}, v)
        else:
            vv = _norm_text(v) if isinstance(v, str) else v
            curr = out.get(k)
            if not curr and vv not in (None, ""):
                out[k] = vv
            elif isinstance(curr, str) and isinstance(vv, str) and len(vv) > len(curr):
                out[k] = vv
    return out

def _prepare_and_merge_courses(raw_courses: List[Dict[str, Any]],
                               university_name: Optional[str],
                               file_hint: Optional[str] = None,
                               fuzzy_threshold: int = 88) -> List[Dict[str, Any]]:
    """
    - strip uni name from lesson_name
    - use lesson_name and file base name as fuzzy keys
    - merge near-duplicates
    """
    merged: Dict[str, Dict[str, Any]] = {}
    aliases: Dict[str, List[str]] = defaultdict(list) 

    for c in (raw_courses or []):
        c = dict(c or {})
        ln = _strip_uni_from_title(c.get("lesson_name") or "", university_name)
        if not ln and file_hint:
            ln = _strip_uni_from_title(file_hint, university_name)
        c["lesson_name"] = ln or c.get("lesson_name") or "Untitled"

        if c.get("website"):
            c.setdefault("websites", [])
            if c["website"] not in c["websites"]:
                c["websites"].append(c["website"])

        name_key = _canonical_key(c["lesson_name"])
        file_key = _canonical_key(_strip_uni_from_title((file_hint or ""), university_name)) if file_hint else ""
        candidates = [k for k in [name_key, file_key] if k]


        match = None
        for cand in candidates:
            match = _best_match(cand, list(merged.keys()), threshold=fuzzy_threshold)
            if match:
                break

        if not match:
            key = name_key or (file_key if file_key else f"course_{len(merged)+1}")
            merged[key] = c
            aliases[key].extend([x for x in candidates if x])
        else:
            merged[match] = _merge_course_records(merged[match], c)
            aliases[match].extend([x for x in candidates if x])

    out = []
    for key, course in merged.items():
        course["lesson_name"] = _strip_uni_from_title(course.get("lesson_name") or "Untitled", university_name)
        if "websites" in course:
            course["websites"] = sorted({w for w in course["websites"] if w})
        out.append(course)
    return out

def _infer_msc_bsc_from_text(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()

    phd = [r'\bph\.?\s*d\b', r'\bdphil\b', r'\bdoctor(?:al|ate)\b']
    if any(re.search(p, t) for p in phd):
        return "PhD"

    masters = [
        r'\bmsc\b', r"\bmaster(?:'s)?\b", r'\bm\.?\s?eng\b', r'\bmba\b',
        r'\bmres\b', r'\bmphil\b', r'\bllm\b', r'\bmsci\b', r'\bmtech\b',
        r'\bpostgraduate\b'
    ]
    if any(re.search(p, t) for p in masters):
        return "MSc"

    bachelors = [
        r'\bbachelor(?:\'s)?\b', r'\bbsc\b', r'\bb\.?\s?eng\b', r'\bllb\b',
        r'\bbtech\b', r'\bundergra(?:d|duate)\b'
    ]
    if any(re.search(p, t) for p in bachelors):
        return "BSc"

    return None