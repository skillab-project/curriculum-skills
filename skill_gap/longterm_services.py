"""
longterm_services.py
====================
Helper functions for Long-Term analysis:
- Calls the SKILLAB Future Technology Trends Identifier API
  (analyze PDF → track status → map to ESCO → generate policy recommendations)
- Stores results per user in local files (JSON)
- Retrieves results per user
"""

import os
import json
import time
import logging
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

TRENDS_BASE_URL = os.getenv(
    "TRENDS_API_BASE_URL",
    "https://portal.skillab-project.eu/future-technology-trends-identifier"
)

# Local storage directory for results per user
STORAGE_DIR = Path(os.getenv("LONGTERM_STORAGE_DIR", "/app/longterm_storage"))


# ==========================================
# STORAGE HELPERS
# ==========================================
def _user_dir(user_id: str) -> Path:
    """Returns (and creates if missing) the user's storage directory."""
    d = STORAGE_DIR / user_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_analysis(user_id: str, job_id: str, data: Dict[str, Any]) -> Path:
    """Saves PDF analysis results for the user."""
    path = _user_dir(user_id) / f"analysis_{job_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved analysis for user={user_id}, job={job_id} → {path}")
    return path


def save_policy(user_id: str, job_id: str, data: Dict[str, Any]) -> Path:
    """Saves policy recommendations for the user."""
    path = _user_dir(user_id) / f"policy_{job_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved policy for user={user_id}, job={job_id} → {path}")
    return path


def get_user_analyses(user_id: str) -> List[Dict[str, Any]]:
    """Returns all analyses for the user."""
    d = _user_dir(user_id)
    results = []
    for f in sorted(d.glob("analysis_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")
    return results


def get_user_policies(user_id: str) -> List[Dict[str, Any]]:
    """Returns all policies for the user."""
    d = _user_dir(user_id)
    results = []
    for f in sorted(d.glob("policy_*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                results.append(json.load(fh))
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")
    return results


def get_analysis_by_job(user_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Returns a specific analysis for the user."""
    path = _user_dir(user_id) / f"analysis_{job_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_policy_by_job(user_id: str, job_id: str) -> Optional[Dict[str, Any]]:
    """Returns a specific policy for the user."""
    path = _user_dir(user_id) / f"policy_{job_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# TRENDS API CALLS
# ==========================================
def analyze_pdf(pdf_bytes: bytes, filename: str, user_id: str) -> Dict[str, Any]:
    """
    Sends a PDF to the TRENDS API for analysis.
    Returns: { job_id, status, message }
    """
    try:
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        data = {"user_id": user_id}
        resp = requests.post(
            f"{TRENDS_BASE_URL}/analyze/pdf",
            files=files,
            data=data,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"analyze_pdf failed: {e}")
        raise


def track_job_status(job_id: str) -> Dict[str, Any]:
    """
    Checks the status of a job.
    Returns: { job_id, status, message, result_path }
    status: "running" | "done" | "failed"
    """
    try:
        resp = requests.get(
            f"{TRENDS_BASE_URL}/jobs/{job_id}",
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"track_job_status failed for job={job_id}: {e}")
        raise


def wait_for_job(job_id: str, poll_interval: int = 5, max_wait: int = 300) -> Dict[str, Any]:
    """
    Waits until the job completes (polling).
    max_wait: max seconds to wait (default 5 min)
    Returns the final status dict.
    """
    elapsed = 0
    while elapsed < max_wait:
        status = track_job_status(job_id)
        if status.get("status") in ("done", "failed"):
            return status
        logger.info(f"Job {job_id} status: {status.get('status')} — waiting {poll_interval}s...")
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")


def map_to_esco(job_id: str, top_n: int = 5, threshold: float = 0.4, target: str = "both") -> Dict[str, Any]:
    """
    Maps identified technologies to ESCO occupations & skills.
    Returns a list of { technology, matches: [{label, score}] }
    """
    try:
        payload = {
            "job_id": job_id,
            "top_n": top_n,
            "threshold": threshold,
            "target": target
        }
        resp = requests.post(
            f"{TRENDS_BASE_URL}/map-to-esco",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"map_to_esco failed for job={job_id}: {e}")
        raise


def generate_policy_recommendations(
    job_id: str,
    user_id: str,
    target: str = "both",
    similarity_threshold: float = 0.5,
    max_actions_per_tech: int = 5
) -> Dict[str, Any]:
    """
    Generates policy recommendations for emerging technologies.
    Returns recommendations per technology.
    """
    try:
        payload = {
            "job_id": job_id,
            "user_id": user_id,
            "target": target,
            "similarity_threshold": similarity_threshold,
            "max_actions_per_tech": max_actions_per_tech
        }
        resp = requests.post(
            f"{TRENDS_BASE_URL}/policy/recommendations",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"generate_policy_recommendations failed for job={job_id}: {e}")
        raise


def download_results(job_id: str) -> Dict[str, Any]:
    """
    Downloads the results of a job from the TRENDS API.
    """
    try:
        resp = requests.get(
            f"{TRENDS_BASE_URL}/results/{job_id}/download",
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"download_results failed for job={job_id}: {e}")
        raise