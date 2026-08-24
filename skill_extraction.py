# -*- coding: utf-8 -*-
"""
Local (in-process) ESCO skill extraction for curriculum-skills.

Replaces the external `API_SKILL_EXTRACTOR_BASE_URL` service and the Tracker
name-resolution step with a self-contained pipeline, using the same approach as
the llm-skill-extraction project:

    course text  --(Mistral LLM, via llm_client)-->  skill phrases
    skill phrase --(SBERT all-MiniLM-L6-v2 cosine)-->  best ESCO skill

The ESCO taxonomy is read from the ESCO `skills_en.csv` already shipped with the
service (data/esco/skills_en.csv). Embeddings are computed once and cached to
disk so subsequent runs are fast.

Public API (kept shape-compatible with the old external helpers so the DB writer
is reused unchanged):
    extract_course_esco_matches(course) -> (course_id, title, {esco_uri: {categories}})
    resolve_uris(uris)                  -> {esco_uri: {"label","esco_id","level"}}
    warm()                              -> preload model + ESCO embeddings
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("skill_extraction")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
MATCH_THRESHOLD = float(os.getenv("ESCO_MATCH_THRESHOLD", "0.47"))
LLM_SKILL_MAX_CHARS = int(os.getenv("LLM_SKILL_MAX_CHARS", "9000"))

# Where to persist the ESCO embedding cache and the downloaded SBERT model.
# Default under longterm_storage so it survives container restarts (that volume
# is mounted in docker-compose).
_LONGTERM = os.getenv("LONGTERM_STORAGE_DIR", "longterm_storage")
ESCO_CACHE_DIR = os.getenv("ESCO_CACHE_DIR", os.path.join(_LONGTERM, "esco_cache"))

# Candidate locations for the ESCO skills CSV (first existing one wins).
_ESCO_CSV_CANDIDATES = [
    os.getenv("ESCO_SKILLS_CSV"),
    "data/skills_en.csv",        # docker: ./data/esco -> /app/data
    "data/esco/skills_en.csv",   # local checkout
    "/app/data/skills_en.csv",
]

# Course fields whose text is fed to the LLM extractor.
COURSE_TEXT_FIELDS = [
    "lesson_name", "degree_titles", "description", "objectives",
    "learning_outcomes", "course_content", "assessment", "prerequisites",
    "general_competences", "educational_material",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_text(value: Any) -> str:
    """Normalize a DB field (str / JSON string / list / dict) to plain text."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_to_text(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(_to_text(v) for v in value.values())
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", "ignore")
    s = str(value).strip()
    if s and s[0] in "[{" and s[-1] in "]}":
        try:
            return _to_text(json.loads(s))
        except Exception:
            return s
    return s


def _course_text(course: Dict[str, Any]) -> str:
    parts = [_to_text(course.get(f)) for f in COURSE_TEXT_FIELDS]
    return "\n".join(p for p in parts if p)


def _esco_id_from_uri(uri: str) -> Optional[str]:
    if not uri:
        return None
    m = re.search(r"/([0-9a-fA-F-]{8,})$", uri.strip())
    return m.group(1) if m else None


def _resolve_csv_path() -> Optional[str]:
    for c in _ESCO_CSV_CANDIDATES:
        if c and os.path.exists(c):
            return c
    return None


# ---------------------------------------------------------------------------
# LLM extraction (reuses the configured Mistral chat backend)
# ---------------------------------------------------------------------------
_SKILL_PROMPT = (
    "You extract the concrete skills, competences, tools, technologies, methods "
    "and knowledge areas taught or required by a university course, so they can "
    "be mapped to the ESCO skills taxonomy.\n\n"
    "Return ONLY a JSON object of the form {{\"skills\": [\"...\", \"...\"]}} where "
    "each entry is a short canonical skill phrase (2-6 words, no sentences, no "
    "duplicates, no numbering). If there are none, return {{\"skills\": []}}.\n\n"
    "COURSE TEXT:\n{text}"
)


def _parse_phrase_list(raw: str) -> List[str]:
    """Parse a list of skill phrases from an LLM response (object or array)."""
    if not raw:
        return []
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()

    data: Any = None
    try:
        data = json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s) or re.search(r"\[[\s\S]*\]", s)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None

    items: List[Any]
    if isinstance(data, dict):
        items = data.get("skills") or data.get("phrases") or data.get("items") or []
    elif isinstance(data, list):
        items = data
    else:
        return []

    out: List[str] = []
    seen: Set[str] = set()
    for it in items:
        if isinstance(it, dict):
            it = it.get("skill") or it.get("name") or it.get("label") or ""
        phrase = re.sub(r"\s+", " ", str(it or "")).strip()
        # drop a trailing "(context)" annotation if the model added one
        phrase = re.sub(r"\s*\([^)]*\)\s*$", "", phrase).strip()
        if not phrase or len(phrase) > 80:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(phrase)
    return out


def _llm_extract_skill_phrases(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        from llm_client import chat_generate
    except Exception as e:  # pragma: no cover
        logger.error("llm_client unavailable for skill extraction: %s", e)
        return []
    prompt = _SKILL_PROMPT.format(text=text[:LLM_SKILL_MAX_CHARS])
    try:
        raw = chat_generate(prompt, temperature=0.0, json_mode=True)
    except Exception as e:
        logger.warning("LLM skill-phrase extraction failed: %s", e)
        return []
    return _parse_phrase_list(raw)


# ---------------------------------------------------------------------------
# ESCO SBERT normalizer (singleton)
# ---------------------------------------------------------------------------
class _EscoNormalizer:
    def __init__(self) -> None:
        self._model = None
        self._embs = None                 # np.ndarray (N, d), L2-normalized
        self._uris: List[str] = []
        self._labels: List[str] = []
        self._levels: List[Optional[str]] = []
        self._uri_index: Dict[str, int] = {}
        self._ready = False
        self._error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    # ---- loading / indexing ----
    def _load(self) -> None:
        if self._ready or self._error:
            return
        with self._load_lock:
            if self._ready or self._error:
                return
            try:
                self._build()
                self._ready = True
            except Exception as e:  # pragma: no cover
                self._error = str(e)
                logger.exception("ESCO normalizer failed to initialise: %s", e)

    def _build(self) -> None:
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer

        csv_path = _resolve_csv_path()
        if not csv_path:
            raise FileNotFoundError(
                "ESCO skills CSV not found (looked in ESCO_SKILLS_CSV, "
                "data/skills_en.csv, data/esco/skills_en.csv)."
            )

        df = pd.read_csv(csv_path).fillna("")
        for col in ("preferredLabel", "conceptUri"):
            if col not in df.columns:
                raise ValueError(f"ESCO CSV missing required column: {col}")
        alt = df["altLabels"].astype(str) if "altLabels" in df.columns else pd.Series([""] * len(df))
        desc = df["description"].astype(str) if "description" in df.columns else pd.Series([""] * len(df))
        level_col = "reuseLevel" if "reuseLevel" in df.columns else ("skillType" if "skillType" in df.columns else None)

        self._labels = df["preferredLabel"].astype(str).tolist()
        self._uris = df["conceptUri"].astype(str).tolist()
        self._levels = (df[level_col].astype(str).tolist() if level_col else [""] * len(df))
        self._uri_index = {u: i for i, u in enumerate(self._uris)}

        texts = (
            df["preferredLabel"].astype(str) + ". "
            + alt.str.replace("\n", " ", regex=False) + ". "
            + desc
        ).tolist()

        cache_dir = Path(ESCO_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            mtime = int(os.path.getmtime(csv_path))
        except OSError:
            mtime = 0
        import hashlib
        key = hashlib.sha256(
            f"{EMBED_MODEL}|{os.path.abspath(csv_path)}|{mtime}|{len(df)}".encode("utf-8")
        ).hexdigest()[:16]
        emb_file = cache_dir / f"esco_emb_{key}.npy"

        self._model = SentenceTransformer(EMBED_MODEL, cache_folder=str(cache_dir / "models"))

        if emb_file.exists():
            self._embs = np.load(emb_file)
            logger.info("Loaded cached ESCO embeddings (%d skills) from %s", len(self._uris), emb_file)
        else:
            logger.info("Building ESCO embeddings for %d skills (one-time, may take a few minutes)…", len(texts))
            embs = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False,
            ).astype("float32")
            np.save(emb_file, embs)
            self._embs = embs
            logger.info("ESCO embeddings built and cached to %s", emb_file)

    # ---- querying ----
    def match(self, phrases: List[str], threshold: float = MATCH_THRESHOLD) -> List[Optional[Dict[str, Any]]]:
        self._load()
        if not phrases:
            return []
        if not self._ready:
            return [None] * len(phrases)
        import numpy as np
        with self._encode_lock:  # torch models are not safe for concurrent encode
            vecs = self._model.encode(
                phrases,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=False,
            ).astype("float32")
        sims = vecs @ self._embs.T  # (P, N); both sides normalized -> cosine
        best = sims.argmax(axis=1)
        out: List[Optional[Dict[str, Any]]] = []
        for i in range(len(phrases)):
            j = int(best[i])
            score = float(sims[i, j])
            if score >= threshold:
                uri = self._uris[j]
                out.append({
                    "uri": uri,
                    "label": self._labels[j],
                    "esco_id": _esco_id_from_uri(uri),
                    "level": (self._levels[j] or None),
                    "score": round(score, 4),
                })
            else:
                out.append(None)
        return out

    def meta_for(self, uris: Set[str]) -> Dict[str, Dict[str, Any]]:
        self._load()
        if not self._ready:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for u in uris:
            i = self._uri_index.get(u)
            if i is None:
                continue
            out[u] = {
                "label": self._labels[i],
                "esco_id": _esco_id_from_uri(u),
                "level": (self._levels[i] or None),
            }
        return out


_esco = _EscoNormalizer()


# ---------------------------------------------------------------------------
# Public API (drop-in for the old external helpers)
# ---------------------------------------------------------------------------
def extract_course_esco_matches(
    course: Dict[str, Any],
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[Any, str, Dict[str, Set[str]]]:
    """Extract ESCO skills for one course.

    Returns (course_id, title, {esco_uri: {categories}}) — the same shape the old
    `_extract_urls_for_course_all_fields` returned, so the existing DB writer is
    reused unchanged.
    """
    cid = course.get("course_id")
    title = course.get("lesson_name") or f"course_{cid}"
    uri_to_categories: Dict[str, Set[str]] = {}

    phrases = _llm_extract_skill_phrases(_course_text(course))
    if not phrases:
        return cid, title, uri_to_categories

    for phrase, m in zip(phrases, _esco.match(phrases, threshold)):
        if not m:
            continue
        uri_to_categories.setdefault(m["uri"], set()).add("course")
    return cid, title, uri_to_categories


def resolve_uris(uris: Set[str]) -> Dict[str, Dict[str, Any]]:
    """Resolve ESCO URIs to {label, esco_id, level} from the local taxonomy."""
    if not uris:
        return {}
    return _esco.meta_for(set(uris))


def warm() -> Dict[str, Any]:
    """Preload the model and ESCO embeddings; returns a small status dict."""
    _esco._load()
    return {
        "ready": _esco._ready,
        "error": _esco._error,
        "esco_skills": len(_esco._uris),
        "model": EMBED_MODEL,
        "threshold": MATCH_THRESHOLD,
    }
