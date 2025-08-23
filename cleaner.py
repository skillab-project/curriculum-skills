import os
import json
import argparse
import time
import sys
import re
from typing import List, Dict, Any, Optional, Tuple
from rapidfuzz import fuzz

def deduplicate_titles(titles: list[str], threshold: int = 95) -> list[str]:
    """
    Remove near-duplicate degree titles using fuzzy matching.
    Keeps the first occurrence, removes later ones above the similarity threshold.
    """
    unique = []
    for t in titles:
        if not any(fuzz.ratio(t.lower(), u.lower()) >= threshold for u in unique):
            unique.append(t)
    return unique

def force_cut_length(title: str, max_len: int = 80) -> str:
    if not title:
        return title
    title = title.strip()
    if len(title) > max_len:
        title = title[:max_len].rsplit(" ", 1)[0]
    return title.strip()


def pre_strip_requirements(title: str) -> str:
    """
    Remove obvious requirement lists from degree title strings before sending to LLM.
    """
    if not title:
        return title

    MAX_DEGREE_LEN = 100

    if len(title) > MAX_DEGREE_LEN:
        title = title[:MAX_DEGREE_LEN].rsplit(" ", 1)[0]


    # Cut at common requirement words
    cut_words = [
        r"\bat least\b", r"\bfurthermore\b", r"\bincluding\b",
        r"\bmust have\b", r"\bshould consist\b", r"\bconsist of\b"
    ]
    pattern = re.compile("|".join(cut_words), re.IGNORECASE)
    m = pattern.search(title)
    if m:
        title = title[:m.start()].strip()

    # Cut if "credits" appears after first degree phrase
    m = re.search(r"\b\d+\s*credits?\b", title, flags=re.IGNORECASE)
    if m and m.start() > 30:
        title = title[:m.start()].strip()

    # Remove trailing punctuation
    title = re.sub(r"[\.,;:\-–]+$", "", title).strip()
    return title


def _lazy_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        print("Please install openai: pip install openai", file=sys.stderr)
        raise
    return OpenAI()

def _lazy_requests():
    try:
        import requests
    except Exception:
        print("Please install requests: pip install requests", file=sys.stderr)
        raise
    return requests


# =========================
# Prompt + Parsing
# =========================

SYSTEM_MSG = (
    "You are a precise academic classifier. "
    "Given a list of candidate academic degree titles, "
    "identify which are VALID academic degree titles."
)

USER_PROMPT_TEMPLATE = """
You will receive a JSON object: {{"items":[{{"id": 0, "title": "..." }}, ...]}}.

Task: For each item, decide if the string is a VALID academic degree title of ANY level
(acceptable: Bachelor's, Master's, Doctoral, Professional degrees; also PGCert/PGDip and similar).

Return exactly ONE JSON object with key "results" mapping each id to an object:
{{
  "results": {{
    "0": {{"valid": true, "normalized": "..." , "reason": "..."}},
    "1": {{"valid": false, "normalized": null, "reason": "..." }},
    ...
  }}
}}

Rules / Hints:
- A valid degree title should be short (typically under 10 words) and clearly identify the qualification name only.
- ACCEPT common types and formats (abbreviated or long form), e.g.:
  - Bachelor's: "BSc X", "BA in X", "Bachelor of Engineering"
  - Master's: "MSc X", "MA in X", "Master of Arts in X", "MRes X", "MBA", "LLM", "MArch", "MPH", "MEd", "MEng"
  - Doctoral/Research: "PhD in X", "DPhil", "EdD", "MD (Research)", "Doctor of Philosophy"
  - Postgraduate awards: "PGCert X", "PGDip in X", "Postgraduate Diploma in X"
  - Professional degrees (e.g., "MD", "JD", where applicable)
- REJECT non-degree text: marketing/breadcrumbs/navigation like "Apply", "Fees", "Entry Requirements",
  "Related Programmes", "Home", "Search", "Contact", "How to Apply", "A-Z".
  ALSO reject ANY degree names that make no sense! Like they are random series of words. Usually degrees make sense and are relevant to the course!
- REJECT department/school names alone without a degree type (e.g., "School of Computing").
- If VALID, provide a concise normalized title (Title Case + standard abbreviations if obvious),
  e.g., "MSc Data Science", "Bachelor of Arts in History", "PhD in Chemistry", "PGDip Marketing".
- If INVALID, set "normalized" to null and give a brief reason.
- DO NOT include course requirements, credit amounts, grade levels, or prerequisite details in the title.
- If the provided text contains subject lists, credit counts (e.g., "5 Credits"), or phrases like "including", "furthermore", "must have", 
  then it is NOT a degree title — reject it as invalid.
- You MUST ensure the cleaned degree title is concise and no longer than 80 characters.
- If valid, output only the degree title without any extra phrases or requirements.
- If invalid, set `normalized` to null and give a brief reason.


Only output the JSON object described (no extra text).
Items:
{payload}
""".strip()


def build_payload(titles: List[str]) -> Dict[str, Any]:
    return {"items": [{"id": i, "title": t} for i, t in enumerate(titles)]}


def call_openai(model: str, system_msg: str, user_msg: str, temperature: float = 0.0, max_retries: int = 4) -> str:
    client = _lazy_openai_client()
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


def call_ollama(model: str, user_msg: str, system_msg: Optional[str] = None, max_retries: int = 4) -> str:
    requests = _lazy_requests()
    url = "http://localhost:11434/api/generate"
    prompt = (system_msg + "\n\n" if system_msg else "") + user_msg
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=body, timeout=180)
            r.raise_for_status()
            j = r.json()
            return (j.get("response") or "").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2


MAX_FILENAME_LEN = 120

def safe_filename(path: str) -> str:
    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    if len(name) > MAX_FILENAME_LEN:
        name = name[:MAX_FILENAME_LEN].rsplit("_", 1)[0]
    return os.path.join(directory, name + ext)


def parse_llm_json(s: str) -> Dict[str, Any]:
    """
    Try to parse a strict JSON block from the model output.
    If it returns stray text, extract the first {...} block.
    """
    s = s.strip()
    # Quick path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: find first JSON object with basic bracket matching
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = s[start:end+1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    raise ValueError("Could not parse LLM JSON response")


# =========================
# Lightweight Regex Fallback
# =========================

# Quick sanity regex: contains a known degree token OR begins with "Master of" etc.
DEGREE_TOKEN_RX = re.compile(
    r"\b("
    r"BA|BSc|B\.?Eng|BEng|LLB|BFA|BEd|BS|AB|"
    r"MA|MSc|M\.?Eng|MEng|MBA|MPhil|MRes|MLitt|MFA|MEd|MPH|LLM|MS|"
    r"PhD|DPhil|EdD|MD|DDS|DVM|JD|DBA|ScD|DLitt|DLaw|EngD|PsyD|"
    r"PGDip|PGCert|Postgraduate\s+Diploma|Postgraduate\s+Certificate|"
    r"Associate\s+Degree|Certificate|Diploma|Advanced\s+Diploma|Graduate\s+Diploma"
    r")\b",
    re.IGNORECASE
)

BAD_UI_TOKENS = {
    "apply", "fees", "fee", "entry requirements", "requirements", "related programmes",
    "related programs", "related", "home", "search", "contact", "how to apply", "our", "my", "you", "your", "me", "the"
    "book an open day", "prospectus", "a-z", "course catalogue", "module catalogue", "project", "course", "student", "overview", "key", "we", "is", "will"
}

def heuristic_is_valid_any_degree(title: str) -> bool:
    if not title or len(title.strip()) < 2:
        return False
    low = title.lower()
    if any(tok in low for tok in BAD_UI_TOKENS):
        return False
    if not re.search(r"[A-Za-z]", title):
        return False
    if DEGREE_TOKEN_RX.search(title):
        return True
    if re.search(r"\b(Master|Masters|Bachelor|Doctor|Doctorate)\s+of\b", title, re.IGNORECASE):
        return True
    return False


# =========================
# Cleaning Logic
# =========================

def llm_validate_titles_any(
    titles: List[str],
    backend: str,
    model: str,
) -> List[Tuple[str, bool, Optional[str]]]:
    """
    Returns list of (original_title, is_valid, normalized_or_none).
    Falls back to regex if LLM output can't be parsed.
    """
    if not titles:
        return []

    payload = build_payload(titles)
    user_msg = USER_PROMPT_TEMPLATE.format(payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    titles = [pre_strip_requirements(t) for t in titles]
    try:
        if backend == "openai":
            raw = call_openai(model=model, system_msg=SYSTEM_MSG, user_msg=user_msg, temperature=0.0)
        elif backend == "ollama":
            raw = call_ollama(model=model, user_msg=user_msg, system_msg=SYSTEM_MSG)
        else:
            raise ValueError("Unsupported backend")
        parsed = parse_llm_json(raw)
        results = parsed.get("results", {})
        out: List[Tuple[str, bool, Optional[str]]] = []
        for i, t in enumerate(titles):
            r = results.get(str(i), {})
            is_valid = bool(r.get("valid", False))
            normalized = r.get("normalized")
            if is_valid and normalized:
                normalized = force_cut_length(normalized)
            if is_valid and not normalized:
                normalized = t
            out.append((t, is_valid, normalized if is_valid else None))
        return out
    except Exception as e:
        out: List[Tuple[str, bool, Optional[str]]] = []
        for t in titles:
            ok = heuristic_is_valid_any_degree(t)
            out.append((t, ok, t if ok else None))
        return out


def clean_file(
    in_path: str,
    out_path: str,
    backend: str,
    model: str,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Loads one JSON, validates degree_titles with the LLM, writes output.
    Returns summary info.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    titles = data.get("degree_titles", [])
    if not isinstance(titles, list):
        titles = []

    MAX_LESSON_NAME_LEN = 80

    if "lesson_name" in data and isinstance(data["lesson_name"], str):
        if len(data["lesson_name"]) > MAX_LESSON_NAME_LEN:
            data["lesson_name"] = data["lesson_name"][:MAX_LESSON_NAME_LEN].rsplit(" ", 1)[0]


    results = llm_validate_titles_any(titles, backend=backend, model=model)
    kept = [norm if norm else orig for (orig, ok, norm) in results if ok]

    summary = {
        "file": os.path.basename(in_path),
        "original_count": len(titles),
        "kept_count": len(kept),
        "removed_count": len(titles) - len(kept),
        "removed_examples": [orig for (orig, ok, norm) in results if not ok][:5],
    }

    kept = deduplicate_titles(kept, threshold=95)
    data["degree_titles"] = kept
    

    out_path = safe_filename(out_path)

    if dry_run:
        print(f"[DRY] {summary['file']}: kept {summary['kept_count']}/{summary['original_count']}, removed {summary['removed_count']}")
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK]  {summary['file']}: kept {summary['kept_count']}/{summary['original_count']}, removed {summary['removed_count']}")

    return summary


def iter_json_files(folder: str) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            if name.lower().endswith(".json"):
                name_noext, ext = os.path.splitext(name)
                if len(name_noext) > MAX_FILENAME_LEN:
                    base = name_noext[:MAX_FILENAME_LEN].rsplit("_", 1)[0]
                    name = base + ext
                files.append(os.path.join(root, name))
    return sorted(files)






# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser(description="Validate and clean degree_titles using an LLM (accept ANY degree type).")
    p.add_argument("--folder", required=True, help="Folder containing JSON files.")
    p.add_argument("--backend", choices=["openai", "ollama"], default="openai", help="LLM backend to use.")
    p.add_argument("--model", default=None, help="Model name (default: openai=gpt-4o-mini, ollama=llama3.1)")
    p.add_argument("--inplace", action="store_true", help="Write changes in-place (overwrite files).")
    p.add_argument("--outdir", default=None, help="Write cleaned JSON to this folder (ignored if --inplace).")
    p.add_argument("--dry-run", action="store_true", help="Run without writing files.")
    args = p.parse_args()

    folder = args.folder
    backend = args.backend
    model = args.model or ("gpt-4o-mini" if backend == "openai" else "llama3.1")

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    if not args.inplace:
        if not args.outdir:
            args.outdir = os.path.join(folder, "_cleaned")
        os.makedirs(args.outdir, exist_ok=True)

    files = iter_json_files(folder)
    if not files:
        print("No JSON files found.")
        return

    totals = {"files": 0, "titles_before": 0, "titles_after": 0, "removed": 0}
    for in_path in files:
        out_path = in_path if args.inplace else os.path.join(args.outdir, os.path.basename(in_path))
        summary = clean_file(in_path, out_path, backend=backend, model=model, dry_run=args.dry_run)
        totals["files"] += 1
        totals["titles_before"] += summary["original_count"]
        totals["titles_after"] += summary["kept_count"]
        totals["removed"] += summary["removed_count"]

    print("\n=== Summary ===")
    print(f"Files processed:   {totals['files']}")
    print(f"Titles before:     {totals['titles_before']}")
    print(f"Titles kept:       {totals['titles_after']}")
    print(f"Titles removed:    {totals['removed']}")

if __name__ == "__main__":
    main()
