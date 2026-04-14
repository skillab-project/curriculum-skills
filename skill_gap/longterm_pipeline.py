"""
longterm_pipeline.py
====================
Unified module combining:
  - curriculum_services.py
  - curriculum_router.py
  - longterm_full_pipeline_router.py
 
Exports two routers — include in main.py:
  - curriculum_router       →  prefix /curriculum
  - full_pipeline_router    →  prefix /longterm

External dependency (keep unchanged):
  - longterm_services.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import json
import logging
import time
import uuid
import requests
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel

from skill_gap.longterm_services import (
    analyze_pdf,
    wait_for_job,
    map_to_esco,
    generate_policy_recommendations,
    save_analysis,
    save_policy,
    get_analysis_by_job,
)

logger = logging.getLogger(__name__)

TRENDS_BASE_URL = os.getenv(
    "TRENDS_API_BASE_URL",
    "https://portal.skillab-project.eu/future-technology-trends-identifier"
)
STORAGE_DIR = Path(os.getenv("LONGTERM_STORAGE_DIR", "/app/longterm_storage"))


# ==============================================================================
# CURRICULUM SERVICES  (was curriculum_services.py)
# ==============================================================================

def _user_dir(user_id: str) -> Path:
    d = STORAGE_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_curriculum(user_id: str, curriculum_id: str, data: Dict[str, Any]) -> Path:
    path = _user_dir(user_id) / f"curriculum_{curriculum_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved curriculum user={user_id} curriculum_id={curriculum_id}")
    return path


def get_user_curricula(user_id: str) -> List[Dict[str, Any]]:
    results = []
    for f in sorted(_user_dir(user_id).glob("curriculum_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")
    return results


def get_curriculum_by_id(user_id: str, curriculum_id: str) -> Optional[Dict[str, Any]]:
    path = _user_dir(user_id) / f"curriculum_{curriculum_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def delete_curriculum(user_id: str, curriculum_id: str) -> bool:
    path = _user_dir(user_id) / f"curriculum_{curriculum_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def _extract_flat_skills(esco_result: Any, threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Extracts a flat list of { skill_label, score, source_technology } from map-to-esco output.
    Handles Format A (flat list) and Format B ({"occupations":[...], "skills":[...]}).
    """
    skills: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def _harvest(items: list) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            tech = item.get("technology", "")
            for m in item.get("matches", []):
                label = m.get("label", "")
                score = m.get("score", 0.0)
                if label and label not in seen and score >= threshold:
                    seen.add(label)
                    skills.append({"skill_label": label, "score": score, "source_technology": tech})

    if isinstance(esco_result, list):
        _harvest(esco_result)
    elif isinstance(esco_result, dict):
        for value in esco_result.values():
            if isinstance(value, list):
                _harvest(value)

    return sorted(skills, key=lambda x: x["score"], reverse=True)


def extract_skills_from_json(skills_json: Any) -> List[str]:
    """Extracts a flat list of skill labels from an uploaded JSON (multiple formats)."""
    if isinstance(skills_json, dict):
        if "skills" in skills_json:
            return extract_skills_from_json(skills_json["skills"])
        flat: List[str] = []
        for v in skills_json.values():
            flat.extend(extract_skills_from_json(v))
        return flat
    if isinstance(skills_json, list):
        flat = []
        for item in skills_json:
            if isinstance(item, str):
                flat.append(item)
            elif isinstance(item, dict):
                label = (item.get("label") or item.get("skill_label")
                         or item.get("skill") or item.get("name"))
                if label:
                    flat.append(label)
                for m in item.get("matches", []):
                    lbl = m.get("label") or m.get("skill_label")
                    if lbl:
                        flat.append(lbl)
        return list(set(flat))
    return []


def extract_skills_from_pdf_via_trends(
    pdf_bytes: bytes, filename: str, user_id: str,
    top_n: int = 10, threshold: float = 0.4, target: str = "skills"
) -> Dict[str, Any]:
    """Upload curriculum PDF → poll → map-to-esco → flat skill list."""
    resp = requests.post(
        f"{TRENDS_BASE_URL}/analyze/pdf",
        files={"file": (filename, pdf_bytes, "application/pdf")},
        data={"user_id": user_id},
        timeout=60
    )
    resp.raise_for_status()
    job_id = resp.json().get("job_id")
    if not job_id:
        raise ValueError("No job_id returned from analyze/pdf")

    elapsed, poll_interval, max_wait = 0, 5, 300
    while elapsed < max_wait:
        s = requests.get(f"{TRENDS_BASE_URL}/jobs/{job_id}", timeout=30).json()
        if s.get("status") == "done":
            break
        if s.get("status") == "failed":
            raise RuntimeError(f"Job {job_id} failed: {s.get('message')}")
        time.sleep(poll_interval)
        elapsed += poll_interval
    else:
        raise TimeoutError(f"Job {job_id} timed out")

    map_resp = requests.post(
        f"{TRENDS_BASE_URL}/map-to-esco",
        json={"job_id": job_id, "top_n": top_n, "threshold": threshold, "target": target},
        timeout=60
    )
    map_resp.raise_for_status()
    esco_result = map_resp.json()
    flat_skills = _extract_flat_skills(esco_result, threshold)
    return {"job_id": job_id, "filename": filename, "esco_mapping": esco_result,
            "skills": flat_skills, "skill_count": len(flat_skills)}


def compare_curriculum_with_trends(
    curriculum_skills: List[str], trend_skills: List[str], fuzzy_threshold: int = 80
) -> Dict[str, Any]:
    """Compares curriculum skills with trend skills. Returns covered/missing/coverage_pct."""
    def normalize(s: str) -> str:
        return s.lower().strip()

    curriculum_norm = {normalize(s): s for s in curriculum_skills}
    covered, missing = [], []

    for original in trend_skills:
        norm = normalize(original)
        if norm in curriculum_norm:
            covered.append({"skill": original, "match_type": "exact",
                            "matched_curriculum_skill": curriculum_norm[norm]})
            continue
        best_score, best_match = 0.0, None
        for nc, oc in curriculum_norm.items():
            ratio = SequenceMatcher(None, norm, nc).ratio() * 100
            if ratio > best_score:
                best_score, best_match = ratio, oc
        if best_score >= fuzzy_threshold:
            covered.append({"skill": original, "match_type": "fuzzy",
                            "matched_curriculum_skill": best_match,
                            "similarity_score": round(best_score, 1)})
        else:
            missing.append({"skill": original,
                            "recommendation": f"Consider adding '{original}' to the curriculum"})

    total = len(trend_skills)
    return {
        "summary": {
            "total_trend_skills": total,
            "covered_count": len(covered),
            "missing_count": len(missing),
            "coverage_pct": round(len(covered) / total * 100, 2) if total else 0.0
        },
        "covered_skills": covered,
        "missing_skills": missing,
        "recommendations": [m["skill"] for m in missing]
    }


def compare_all_curricula_with_trends(
    user_id: str, trend_skills: List[str], fuzzy_threshold: int = 80
) -> Dict[str, Any]:
    """Compares ALL curricula of the user with the trend skills."""
    curricula = get_user_curricula(user_id)
    if not curricula:
        return {"error": f"No curricula found for user={user_id}"}

    all_skills: Set[str] = set()
    per_curriculum = []
    for c in curricula:
        cur_skills = [(s.get("skill_label") if isinstance(s, dict) else str(s))
                      for s in c.get("skills", [])]
        cur_skills = [s for s in cur_skills if s]
        all_skills.update(cur_skills)
        per_curriculum.append({
            "curriculum_id": c.get("curriculum_id"),
            "filename": c.get("filename"),
            "gap_analysis": compare_curriculum_with_trends(cur_skills, trend_skills, fuzzy_threshold)
        })

    return {
        "user_id": user_id,
        "curricula_count": len(curricula),
        "aggregate_gap": compare_curriculum_with_trends(list(all_skills), trend_skills, fuzzy_threshold),
        "per_curriculum": per_curriculum
    }


# ==============================================================================
# PIPELINE HELPERS  (used by full_pipeline_router)
# ==============================================================================

def _extract_skill_labels_from_esco(esco_result: Any, threshold: float = 0.0) -> List[str]:
    """
    Extracts unique skill labels from map-to-esco output.
    Handles Format A (flat list) and Format B ({"occupations":[...], "skills":[...]}).
    Returns sorted by score (desc).
    """
    labels: Dict[str, float] = {}

    def _harvest(items: list) -> None:
        for item in items:
            if not isinstance(item, dict):
                continue
            for match in item.get("matches", []):
                label = match.get("label", "")
                score = match.get("score", 0.0)
                if label and score >= threshold:
                    if label not in labels or score > labels[label]:
                        labels[label] = score

    if isinstance(esco_result, list):
        _harvest(esco_result)
    elif isinstance(esco_result, dict):
        for value in esco_result.values():
            if isinstance(value, list):
                _harvest(value)

    return [label for label, _ in sorted(labels.items(), key=lambda x: x[1], reverse=True)]


def _extract_policy_skill_labels(policy_result: Any) -> List[str]:
    """Extracts unique skill/occupation labels from policy/recommendations output (recursive)."""
    labels: Set[str] = set()

    def _harvest(obj: Any) -> None:
        if isinstance(obj, str):
            labels.add(obj)
        elif isinstance(obj, dict):
            for key in ("skill", "label", "skill_label", "name", "occupation", "title"):
                if key in obj and isinstance(obj[key], str):
                    labels.add(obj[key])
            for key in ("skills", "occupations", "actions", "recommendations", "matches"):
                if key in obj:
                    _harvest(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _harvest(item)

    _harvest(policy_result)
    return list(labels)


def _run_gap_analysis(
    user_id: str, curriculum_id: Optional[str],
    trend_skill_labels: List[str], fuzzy_threshold: int
) -> Dict[str, Any]:
    if not trend_skill_labels:
        return {"info": "No trend skills available for comparison."}

    if curriculum_id and curriculum_id != "string":
        curriculum = get_curriculum_by_id(user_id, curriculum_id)
        if not curriculum:
            return {"info": f"Curriculum '{curriculum_id}' not found for user '{user_id}'."}
        edu_skills = [(s.get("skill_label") if isinstance(s, dict) else str(s))
                      for s in curriculum.get("skills", [])]
        edu_skills = [s for s in edu_skills if s]
        if not edu_skills:
            return {"info": f"Curriculum '{curriculum_id}' has no skills stored."}
        return compare_curriculum_with_trends(edu_skills, trend_skill_labels, fuzzy_threshold)

    result = compare_all_curricula_with_trends(user_id, trend_skill_labels, fuzzy_threshold)
    return {"info": result["error"]} if "error" in result else result


# ==============================================================================
# CURRICULUM ROUTER   →  app.include_router(curriculum_router)
# ==============================================================================

curriculum_router = APIRouter(prefix="/curriculum", tags=["Curriculum Analysis"])


class CompareWithJobRequest(BaseModel):
    user_id: str
    job_id: str
    curriculum_id: Optional[str] = None
    fuzzy_threshold: int = 80
    esco_threshold: float = 0.4


@curriculum_router.post("/upload/pdf", summary="Upload curriculum PDF → extract skills via TRENDS API")
async def upload_curriculum_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form(...),
    curriculum_name: str = Form(None),
    top_n: int = Form(10),
    threshold: float = Form(0.4),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The file is empty.")
    curriculum_id = str(uuid.uuid4())[:8]
    try:
        result = extract_skills_from_pdf_via_trends(
            pdf_bytes, file.filename, user_id, top_n=top_n, threshold=threshold, target="skills"
        )
        curriculum_data = {
            "curriculum_id": curriculum_id, "user_id": user_id,
            "filename": file.filename, "curriculum_name": curriculum_name or file.filename,
            "job_id": result.get("job_id"), "skills": result.get("skills", []),
            "skill_count": result.get("skill_count", 0), "esco_mapping": result.get("esco_mapping")
        }
        save_curriculum(user_id, curriculum_id, curriculum_data)
        return {"message": "Curriculum uploaded and skills extracted successfully.",
                "curriculum_id": curriculum_id, "filename": file.filename,
                "skill_count": curriculum_data["skill_count"],
                "skills_preview": curriculum_data["skills"][:5],
                "full_data_endpoint": f"/curriculum/{user_id}/{curriculum_id}"}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error extracting skills: {e}")


@curriculum_router.post("/upload/json", summary="Upload curriculum skills as JSON")
async def upload_curriculum_json(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    curriculum_name: str = Form(None),
):
    if not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")
    try:
        skills_json = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    skill_labels = extract_skills_from_json(skills_json)
    if not skill_labels:
        raise HTTPException(status_code=400, detail="No skills found in the JSON file.")
    curriculum_id = str(uuid.uuid4())[:8]
    curriculum_data = {
        "curriculum_id": curriculum_id, "user_id": user_id,
        "filename": file.filename, "curriculum_name": curriculum_name or file.filename,
        "job_id": None,
        "skills": [{"skill_label": s, "score": 1.0, "source_technology": "uploaded"} for s in skill_labels],
        "skill_count": len(skill_labels), "esco_mapping": None
    }
    save_curriculum(user_id, curriculum_id, curriculum_data)
    return {"message": "Curriculum skills uploaded successfully.",
            "curriculum_id": curriculum_id, "filename": file.filename,
            "skill_count": len(skill_labels), "skills_preview": skill_labels[:5],
            "full_data_endpoint": f"/curriculum/{user_id}/{curriculum_id}"}


@curriculum_router.get("/{user_id}", summary="Get all curricula for a user")
def get_curricula(user_id: str):
    curricula = get_user_curricula(user_id)
    if not curricula:
        return {"user_id": user_id, "count": 0, "curricula": []}
    return {"user_id": user_id, "count": len(curricula), "curricula": [
        {"curriculum_id": c.get("curriculum_id"), "curriculum_name": c.get("curriculum_name"),
         "filename": c.get("filename"), "skill_count": c.get("skill_count", 0),
         "job_id": c.get("job_id")} for c in curricula
    ]}


@curriculum_router.get("/{user_id}/{curriculum_id}", summary="Get a specific curriculum")
def get_curriculum(user_id: str, curriculum_id: str):
    c = get_curriculum_by_id(user_id, curriculum_id)
    if not c:
        raise HTTPException(status_code=404, detail=f"Curriculum not found: {curriculum_id}")
    return c


@curriculum_router.delete("/{user_id}/{curriculum_id}", summary="Delete a curriculum")
def remove_curriculum(user_id: str, curriculum_id: str):
    if not delete_curriculum(user_id, curriculum_id):
        raise HTTPException(status_code=404, detail=f"Curriculum not found: {curriculum_id}")
    return {"message": f"Curriculum {curriculum_id} deleted successfully."}


@curriculum_router.post("/compare/job", summary="Compare curriculum vs trend skills from a job")
def compare_with_job(req: CompareWithJobRequest):
    analysis = get_analysis_by_job(req.user_id, req.job_id)
    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis found for job={req.job_id}.")
    esco_mapping = analysis.get("esco_mapping")
    if not esco_mapping:
        raise HTTPException(status_code=400, detail="ESCO mapping not found. Run map-to-esco first.")
    trend_skill_labels = _extract_skill_labels_from_esco(esco_mapping, threshold=req.esco_threshold)
    if not trend_skill_labels:
        raise HTTPException(status_code=400, detail="No skills found in the ESCO mapping.")
    if req.curriculum_id:
        c = get_curriculum_by_id(req.user_id, req.curriculum_id)
        if not c:
            raise HTTPException(status_code=404, detail=f"Curriculum not found: {req.curriculum_id}")
        cur_skills = [(s.get("skill_label") if isinstance(s, dict) else str(s))
                      for s in c.get("skills", [])]
        return {"user_id": req.user_id, "job_id": req.job_id,
                "curriculum_id": req.curriculum_id, "curriculum_name": c.get("curriculum_name"),
                "trend_skills_count": len(trend_skill_labels),
                "gap_analysis": compare_curriculum_with_trends(
                    [s for s in cur_skills if s], trend_skill_labels, req.fuzzy_threshold)}
    result = compare_all_curricula_with_trends(req.user_id, trend_skill_labels, req.fuzzy_threshold)
    return {"job_id": req.job_id, **result}


@curriculum_router.post("/compare/file", summary="Compare curriculum vs skills from uploaded JSON")
async def compare_with_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    curriculum_id: str = Form(None),
    fuzzy_threshold: int = Form(80),
):
    if not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only JSON files are accepted.")
    content = await file.read()
    try:
        skills_json = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    trend_skill_labels = extract_skills_from_json(skills_json)
    if not trend_skill_labels:
        raise HTTPException(status_code=400, detail="No skills found in the JSON file.")
    if curriculum_id:
        c = get_curriculum_by_id(user_id, curriculum_id)
        if not c:
            raise HTTPException(status_code=404, detail=f"Curriculum not found: {curriculum_id}")
        cur_skills = [(s.get("skill_label") if isinstance(s, dict) else str(s))
                      for s in c.get("skills", [])]
        return {"user_id": user_id, "source_file": file.filename,
                "curriculum_id": curriculum_id, "curriculum_name": c.get("curriculum_name"),
                "trend_skills_count": len(trend_skill_labels),
                "gap_analysis": compare_curriculum_with_trends(
                    [s for s in cur_skills if s], trend_skill_labels, fuzzy_threshold)}
    result = compare_all_curricula_with_trends(user_id, trend_skill_labels, fuzzy_threshold)
    return {"source_file": file.filename, **result}


# ==============================================================================
# FULL PIPELINE ROUTER   →  app.include_router(full_pipeline_router)
# ==============================================================================

full_pipeline_router = APIRouter(prefix="/longterm", tags=["Long-Term Full Pipeline"])


@full_pipeline_router.post(
    "/run-full-pipeline",
    summary="Run complete long-term pipeline: PDF → ESCO → policy → curriculum gap analysis"
)
async def run_full_pipeline(
    file: UploadFile = File(..., description="PDF file for technology trends analysis"),
    user_id: str = Form(..., description="Logged-in user ID"),
    curriculum_id: str = Form(None, description="Optional curriculum ID (defaults to all curricula)"),
    top_n: int = Form(5),
    esco_threshold: float = Form(0.4),
    fuzzy_threshold: int = Form(80),
    target: str = Form("both", description="'occupations', 'skills', or 'both'"),
    similarity_threshold: float = Form(0.5),
    max_actions_per_tech: int = Form(5),
):
    """
    **Unified Long-Term Pipeline** — one endpoint for all steps.

    1. PDF Analysis → extract emerging technologies
    2. Polling → wait for completion
    3. ESCO Mapping → FORECAST TRENDS (skills & occupations)
    4. Policy Recommendations → POLICY TRENDS
    5. Gap Analysis → EDU skills (curricula) vs FORECAST + POLICY TRENDS

    ⚠️ Runs synchronously (~2-5 minutes). Use timeout ≥ 360s.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The file is empty.")

    # STEP 1 — Analyze PDF
    logger.info(f"[pipeline] Step 1 user={user_id} file={file.filename}")
    try:
        analyze_result = analyze_pdf(pdf_bytes, file.filename, user_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Step 1 (analyze_pdf) failed: {e}")
    job_id = analyze_result.get("job_id")
    if not job_id:
        raise HTTPException(status_code=502, detail="Step 1: No job_id returned.")
    save_analysis(user_id, job_id, {
        "job_id": job_id, "user_id": user_id, "filename": file.filename,
        "status": analyze_result.get("status"), "message": analyze_result.get("message"),
        "esco_mapping": None, "download": None,
    })

    # STEP 2 — Poll
    logger.info(f"[pipeline] Step 2 polling job={job_id}")
    try:
        final_status = wait_for_job(job_id, poll_interval=5, max_wait=300)
    except TimeoutError:
        raise HTTPException(status_code=504, detail=f"Step 2: Job {job_id} timed out.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Step 2 (wait_for_job) failed: {e}")
    if final_status.get("status") == "failed":
        raise HTTPException(status_code=502, detail=f"Step 2: Job failed — {final_status.get('message')}")

    # STEP 3 — ESCO Mapping (FORECAST TRENDS)
    logger.info(f"[pipeline] Step 3 map_to_esco job={job_id}")
    try:
        esco_result = map_to_esco(job_id, top_n=top_n, threshold=esco_threshold, target=target)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Step 3 (map_to_esco) failed: {e}")
    forecast_skill_labels = _extract_skill_labels_from_esco(esco_result, threshold=esco_threshold)
    logger.info(f"[pipeline] {len(forecast_skill_labels)} forecast skill labels")
    stored = get_analysis_by_job(user_id, job_id) or {}
    stored["esco_mapping"] = esco_result
    save_analysis(user_id, job_id, stored)

    # STEP 4 — Policy Recommendations (POLICY TRENDS)
    logger.info(f"[pipeline] Step 4 policy_recommendations job={job_id}")
    try:
        policy_result = generate_policy_recommendations(
            job_id=job_id, user_id=user_id, target=target,
            similarity_threshold=similarity_threshold,
            max_actions_per_tech=max_actions_per_tech,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Step 4 (policy_recommendations) failed: {e}")
    save_policy(user_id, job_id, {
        "job_id": job_id, "user_id": user_id, "target": target,
        "similarity_threshold": similarity_threshold,
        "max_actions_per_tech": max_actions_per_tech,
        "recommendations": policy_result,
    })
    policy_skill_labels = _extract_policy_skill_labels(policy_result)
    logger.info(f"[pipeline] {len(policy_skill_labels)} policy skill labels")

    # STEP 5 — Gap Analysis
    logger.info(f"[pipeline] Step 5 gap_analysis curriculum_id={curriculum_id}")
    gap_vs_forecast = _run_gap_analysis(user_id, curriculum_id, forecast_skill_labels, fuzzy_threshold)
    gap_vs_policy = _run_gap_analysis(user_id, curriculum_id, policy_skill_labels, fuzzy_threshold)

    return {
        "job_id": job_id,
        "user_id": user_id,
        "filename": file.filename,
        "curriculum_id": curriculum_id or "all",
        "forecast_trends": {
            "esco_mapping": esco_result,
            "skill_labels": forecast_skill_labels,
            "skill_count": len(forecast_skill_labels),
        },
        "policy_trends": {
            "recommendations": policy_result,
            "skill_labels": policy_skill_labels,
            "skill_count": len(policy_skill_labels),
        },
        "gap_analysis": {
            "vs_forecast_trends": gap_vs_forecast,
            "vs_policy_trends": gap_vs_policy,
        },
    }