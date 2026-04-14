"""
services.py
===========
Όλες οι βοηθητικές συναρτήσεις:
- CSV helpers (sectors, occupations)
- Tracker authentication
- Skill fetching από SERVICE2_URL
- Demand (jobs) & Supply (CVs) count fetching
- Gap analysis logic (rank score, gap score)
"""

import os
import logging
import requests
import pandas as pd
import urllib3
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SERVICE2_URL = os.getenv("REQUIRED_SKILLS_SERVICE_URL", "https://portal.skillab-project.eu/required-skills")
TRACKER_BASE_URL = os.getenv("API_TRACKER_BASE_URL", "https://skillab-tracker.csd.auth.gr/api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "New_occupation_table.csv")


# ==========================================
# TRACKER AUTH
# ==========================================
# services.py — αντικατέστησε την get_tracker_token με αυτή:

import time

_token_cache = {"token": None, "expires_at": 0}


def get_tracker_token() -> str:
    """
    Επιστρέφει cached token αν δεν έχει λήξει.
    Κάνει login μόνο αν δεν υπάρχει token ή έχει λήξει (cache 50 λεπτά).
    """
    now = time.time()

    # Αν έχουμε valid cached token, επέστρεψέ το αμέσως
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        logger.info("Using cached tracker token.")
        return _token_cache["token"]

    login_url = f"{TRACKER_BASE_URL}/login"
    payload = {
        "username": os.getenv("TRACKER_USERNAME"),
        "password": os.getenv("TRACKER_PASSWORD")
    }
    headers = {"Content-Type": "application/json", "accept": "application/json"}

    logger.info(f"Attempting tracker login for user: {os.getenv('TRACKER_USERNAME')}")
    try:
        response = requests.post(
            login_url, json=payload, headers=headers,
            verify=False, timeout=120  # ← 2 λεπτά max
        )
        response.raise_for_status()
        token = response.text.strip().strip('"')

        # Cache για 50 λεπτά
        _token_cache["token"] = token
        _token_cache["expires_at"] = now + (50 * 60)
        logger.info("✅ Tracker login successful. Token cached for 50 minutes.")
        return token
    except Exception as e:
        logger.error(f"❌ Tracker login failed: {e}")
        return ""

# ==========================================
# CSV HELPERS
# ==========================================
def _read_csv() -> Optional[pd.DataFrame]:
    if not os.path.exists(CSV_PATH):
        logger.error(f"CSV not found at: {CSV_PATH}")
        return None
    try:
        try:
            df = pd.read_csv(CSV_PATH, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8')
        except Exception:
            df = pd.read_csv(CSV_PATH, sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')
        df.columns = df.columns.str.strip().str.replace('"', '')
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return None


def load_sectors() -> List[str]:
    df = _read_csv()
    if df is None or 'Label3' not in df.columns:
        return []
    return sorted(df['Label3'].dropna().astype(str).str.strip().str.strip('"').unique().tolist())


def load_occupations(sector_filter: Optional[str] = None) -> List[str]:
    df = _read_csv()
    if df is None:
        return []

    if sector_filter and 'Label3' in df.columns:
        df = df[df['Label3'].astype(str).str.contains(sector_filter, case=False, na=False)]

    def clean(text):
        if not isinstance(text, str): return ""
        return text.replace("β€™", "'").replace("â€™", "'").strip().strip('"').strip("'")

    occupations = set()
    if 'Label4' in df.columns:
        occupations.update(df['Label4'].dropna().apply(clean).tolist())
    if 'Label3' in df.columns:
        occupations.update(df['Label3'].dropna().apply(clean).tolist())

    result = [o for o in occupations if o and len(o) > 2]
    logger.info(f"✅ Loaded {len(result)} occupations")
    return result


# ==========================================
# SKILL FETCHING FROM SERVICE2
# ==========================================
def _fetch_skills_for_occupation(occupation: str, min_val: float) -> tuple:
    try:
        resp = requests.post(
            f"{SERVICE2_URL}/required_skills_service",
            json={"occupation_name": occupation},
            timeout=30
        )
        if resp.status_code == 200 and resp.text:
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], str):
                if "cannot open" in data[0].lower() or "error" in data[0].lower():
                    return occupation, []
            skills = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        val = item.get('Value', 0) or 0
                        if val >= min_val:
                            skills.append({
                                "skill_name": item.get("Skill"),
                                "skill_id": item.get("SkillId") or item.get("skill_id") or item.get("Id"),
                                "value": val
                            })
            return occupation, skills
        return occupation, []
    except Exception as e:
        logger.warning(f"Failed skills for '{occupation}': {e}")
        return occupation, []


def fetch_all_skills_parallel(occupations: List[str], min_val: float = 0.0, max_workers: int = 3) -> Dict[str, List]:
    results = {}
    total = len(occupations)
    logger.info(f"Fetching skills for {total} occupations (max_workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_skills_for_occupation, occ, min_val): occ for occ in occupations}
        done = 0
        for future in as_completed(futures):
            occ, skills = future.result()
            if skills:
                results[occ] = skills
            done += 1
            if done % 10 == 0:
                logger.info(f"  Skills progress: {done}/{total}")

    logger.info(f"✅ Skills fetched for {len(results)}/{total} occupations with results.")
    return results


# ==========================================
# TRACKER API: DEMAND (jobs) & SUPPLY (CVs)
# ==========================================
def _get_skill_job_count(skill_id: str, token: str) -> int:
    """
    Παίρνει count job ads για ένα skill_id.
    Χρησιμοποιεί form-urlencoded όπως το Postman collection.
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = f"skill_ids={requests.utils.quote(skill_id)}&skill_ids_logic=or"
        resp = requests.post(
            f"{TRACKER_BASE_URL}/jobs?page=1&page_size=1",
            headers=headers, data=data, verify=False, timeout=20
        )
        if resp.ok:
            count = resp.json().get("count", 0)
            logger.debug(f"  Job count for {skill_id[-20:]}: {count}")
            return count
        logger.warning(f"  Job count failed {resp.status_code}: {resp.text[:80]}")
        return 0
    except Exception as e:
        logger.warning(f"  _get_skill_job_count error: {e}")
        return 0


def _get_skill_profile_count(skill_id: str, token: str) -> int:
    """
    Παίρνει count profiles/CVs για ένα skill_id.
    Χρησιμοποιεί form-urlencoded όπως το Postman collection.
    """
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = f"skill_ids={requests.utils.quote(skill_id)}&skill_ids_logic=or"
        resp = requests.post(
            f"{TRACKER_BASE_URL}/profiles?page=1&page_size=1",
            headers=headers, data=data, verify=False, timeout=20
        )
        if resp.ok:
            count = resp.json().get("count", 0)
            logger.debug(f"  Profile count for {skill_id[-20:]}: {count}")
            return count
        logger.warning(f"  Profile count failed {resp.status_code}: {resp.text[:80]}")
        return 0
    except Exception as e:
        logger.warning(f"  _get_skill_profile_count error: {e}")
        return 0


def fetch_counts_parallel(skills: List[Dict], token: str, max_workers: int = 10) -> List[Dict]:
    """Παράλληλη ανάκτηση demand + supply counts για όλα τα skills."""
    logger.info(f"  Fetching counts for {len(skills)} skills (max_workers={max_workers})...")

    def fetch_one(skill: Dict) -> Dict:
        sid = skill.get("skill_id")
        if not sid:
            logger.warning(f"  Skill '{skill.get('skill_name')}' has no skill_id — skipping")
            return {**skill, "demand_count": 0, "supply_count": 0}
        demand = _get_skill_job_count(sid, token)
        supply = _get_skill_profile_count(sid, token)
        return {**skill, "demand_count": demand, "supply_count": supply}

    enriched = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_one, s) for s in skills]
        for f in as_completed(futures):
            enriched.append(f.result())

    # Log sample για debugging
    sample = enriched[:3]
    for s in sample:
        logger.info(f"  Sample: {s.get('skill_name')} | demand={s.get('demand_count')} | supply={s.get('supply_count')}")

    return enriched


# ==========================================
# GAP ANALYSIS LOGIC
# ==========================================

def compute_rank_score(skills: List[Dict], count_key: str) -> List[Dict]:
    """
    Μόνο τα skills με count > 0 παίρνουν rank score.
    Skills με count = 0 παίρνουν score = 0.0
    Formula: ((N - position) + 1) / N * 100
    """
    score_key = f"{count_key}_score"

    # Χώρισε σε αυτά που έχουν count > 0 και αυτά που έχουν 0
    with_count = [s for s in skills if (s.get(count_key) or 0) > 0]
    without_count = [s for s in skills if (s.get(count_key) or 0) == 0]

    # Ταξινόμηση φθίνουσα βάσει count
    ranked = sorted(with_count, key=lambda x: x.get(count_key, 0), reverse=True)
    N = len(ranked)

    # Score μόνο για αυτά με count > 0
    for i, skill in enumerate(ranked):
        skill[score_key] = round(((N - i) / N) * 100, 2)

    # Score = 0 για αυτά με count = 0
    for skill in without_count:
        skill[score_key] = 0.0

    return ranked + without_count

def compute_gap(demand_skills: List[Dict], supply_skills: List[Dict]) -> List[Dict]:
    """
    gap_score = demand_score - supply_score
    +100% = hot skill (ζητείται αλλά δεν προσφέρεται)
    -100% = oversupplied (προσφέρεται αλλά δεν ζητείται)
    """
    demand_map = {s.get("skill_id"): s for s in demand_skills if s.get("skill_id")}
    supply_map = {s.get("skill_id"): s for s in supply_skills if s.get("skill_id")}

    all_ids = set(demand_map.keys()) | set(supply_map.keys())
    gap_results = []

    for sid in all_ids:
        d = demand_map.get(sid, {})
        s = supply_map.get(sid, {})

        d_score = d.get("demand_count_score", 0.0)
        s_score = s.get("supply_count_score", 0.0)
        gap = round(d_score - s_score, 2)

        gap_results.append({
            "skill_id": sid,
            "skill_name": d.get("skill_name") or s.get("skill_name"),
            "demand_count": d.get("demand_count", 0),
            "supply_count": s.get("supply_count", 0),
            "demand_score": d_score,
            "supply_score": s_score,
            "gap_score": gap
        })

    return sorted(gap_results, key=lambda x: x["gap_score"], reverse=True)