import os
import re
import json

UNIS_PATH = os.getenv("UNIS_JSON_PATH", "/mnt/data/world_universities_and_domains.json")
_UNI_DATA = None
_NAME_INDEX = None

def _norm(s: str) -> str:
    return re.sub(r"[_\W]+", " ", s).lower().strip()

def _score(a: str, b: str) -> float:
    ta = set(_norm(a).split())
    tb = set(_norm(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _load_universities():
    global _UNI_DATA, _NAME_INDEX
    if _UNI_DATA is not None and _NAME_INDEX is not None:
        return
    try:
        with open(UNIS_PATH, "r", encoding="utf-8") as f:
            _UNI_DATA = json.load(f)
    except Exception:
        _UNI_DATA = []
    _NAME_INDEX = {}
    for e in _UNI_DATA:
        n = _norm(e.get("name", ""))
        if n:
            _NAME_INDEX[n] = e

def _find_uni_by_name(query: str) -> dict:
    _load_universities()
    qn = _norm(query or "")
    if not qn:
        return {}
    if qn in _NAME_INDEX:
        e = _NAME_INDEX[qn]
        return {
            "name": e.get("name"),
            "country": e.get("country"),
            "domain": (e.get("domains") or [None])[0]
        }
    best = None
    best_score = 0.0
    for e in _UNI_DATA:
        name = e.get("name", "")
        sc = _score(qn, name)
        if sc > best_score:
            best_score = sc
            best = e
    if best and best_score >= 0.5:
        return {
            "name": best.get("name"),
            "country": best.get("country"),
            "domain": (best.get("domains") or [None])[0]
        }
    return {}

def print_colored_text(text: str, color_code: str) -> None:
    print(f"\033[{color_code}m{text}\033[0m")


def print_horizontal_line(length: int) -> None:
    print('=' * length)

def print_loading_line(length: int) -> None:
    print_colored_text('=' * length + '|] Loading...[|' + '=' * length, 33)

def print_horizontal_small_line(length: int) -> None:
    print('-' * length)

def print_green_line(length: int) -> None:
    print_colored_text('=' * length, 32)

def print_yellow_line(length: int) -> None:
    print_colored_text('=' * length, 33)


