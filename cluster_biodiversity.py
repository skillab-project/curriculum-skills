#!/usr/bin/env python3
"""
Cluster degrees/courses by skill similarity using the Biodiversity endpoint.

Usage examples:
  # Cluster DEGREES by Level-4 skills, KMeans 8 clusters
  python cluster_biodiversity.py \
    --base-url https://portal.skillab-project.eu/curriculum-skills \
    --theme "Computer Science" \
    --countries GR DE \
    --level 4 \
    --mode degree \
    --algo kmeans --k 8 \
    --out-csv clusters_degrees.csv \
    --out-summary clusters_degrees_summary.json

  # Cluster COURSES, DBSCAN with cosine distance
  python cluster_biodiversity.py \
    --base-url https://portal.skillab-project.eu/curriculum-skills \
    --level 4 \
    --mode course \
    --algo dbscan --eps 0.35 --min-samples 4 \
    --out-csv clusters_courses.csv

  # Offline demo against a saved biodiversity response
  python cluster_biodiversity.py --input-json response.json --mode degree --out-csv demo.csv

Notes:
- Expects Biodiversity API to support POST /biodiversity_analysis with paging:
    body = { countries, theme, level, degree_types, page, per_page }
  and a response like { page, per_page, total_pages, results: [...] }
- Results should include 'skills' (or 'level4_skills'/'unique_skills'), and for degrees a 'degree' dict.
- The script is resilient to slight schema differences.
"""

import argparse, json, ast, re, sys, math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter

import requests
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


# ---------- Utilities ----------

def _clean_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = re.sub(r"\s+", " ", str(x)).strip()
    return s or None

def _parse_degree_struct(d: Any) -> Tuple[Optional[str], Optional[str]]:
    """degree is often {"type":"BSc","title":"{'BSc': ['BSc Art and Education', ...]}"}"""
    deg_type, deg_title = None, None
    if isinstance(d, dict):
        deg_type = _clean_text(d.get("type"))
        title = d.get("title")
        if isinstance(title, str) and title.strip().startswith("{") and ":" in title:
            try:
                parsed = ast.literal_eval(title)
                if isinstance(parsed, dict):
                    if deg_type and deg_type in parsed and isinstance(parsed[deg_type], list) and parsed[deg_type]:
                        deg_title = parsed[deg_type][0]
                    else:
                        for _, v in parsed.items():
                            if isinstance(v, list) and v:
                                deg_title = v[0]
                                break
                if not deg_title:
                    deg_title = title
            except Exception:
                deg_title = title
        elif isinstance(title, str):
            deg_title = title
    return _clean_text(deg_type), _clean_text(deg_title)

def _skills_from_row(r: Dict[str, Any]) -> List[str]:
    skills = r.get("skills") or r.get("level4_skills") or r.get("unique_skills") or []
    out = []
    for s in skills:
        if isinstance(s, str):
            ss = _clean_text(s)
            if ss:
                out.append(ss)
    return sorted(set(out))

def _fetch_all_biodiversity(base_url: str,
                            countries: Optional[List[str]],
                            theme: Optional[str],
                            level: Optional[int],
                            degree_types: Optional[List[str]],
                            token: Optional[str],
                            per_page: int = 200) -> List[Dict[str, Any]]:
    s = requests.Session()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = token

    payload = {
        "countries": countries, "theme": theme, "level": level,
        "degree_types": degree_types, "page": 1, "per_page": per_page
    }
    rows: List[Dict[str, Any]] = []
    while True:
        r = s.post(base_url.rstrip("/") + "/biodiversity_analysis", json=payload, headers=headers, timeout=60)
        if r.status_code >= 400:
            raise SystemExit(f"HTTP {r.status_code} from biodiversity: {r.text[:300]}")
        data = r.json()
        page = int(data.get("page", payload["page"]))
        total_pages = int(data.get("total_pages", page))
        results = data.get("results") or data.get("data") or []
        rows.extend(results)
        if page >= total_pages:
            break
        payload["page"] = page + 1
    return rows

def _aggregate_to_degrees(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    bucket = {}
    for r in rows:
        country   = _clean_text(r.get("country"))
        university= _clean_text(r.get("university"))
        department= _clean_text(r.get("department"))
        deg_type, deg_title = _parse_degree_struct(r.get("degree") or {})
        key = (country, university, department, deg_type, deg_title)
        if key not in bucket:
            bucket[key] = {
                "country": country,
                "university": university,
                "department": department,
                "degree_type": deg_type,
                "degree_title": deg_title,
                "program_name": _clean_text(r.get("program_name") or r.get("course_title")),
                "skills": set(),
            }
        for s in _skills_from_row(r):
            bucket[key]["skills"].add(s)

    out = []
    for v in bucket.values():
        sk = sorted(v["skills"])
        out.append({
            **{k: v[k] for k in ("country","university","department","degree_type","degree_title","program_name")},
            "unique_skill_count": len(sk),
            "unique_skills": "|".join(sk),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df[(df["unique_skill_count"] > 0) & df["degree_type"].notna() & df["degree_title"].notna()].reset_index(drop=True)
    return df

def _aggregate_to_courses(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    items = []
    for r in rows:
        skills = _skills_from_row(r)
        if not skills:
            continue
        items.append({
            "country": _clean_text(r.get("country")),
            "university": _clean_text(r.get("university")),
            "department": _clean_text(r.get("department")),
            "course_title": _clean_text(r.get("course_title") or r.get("name") or r.get("title")),
            "unique_skill_count": len(skills),
            "unique_skills": "|".join(skills),
        })
    df = pd.DataFrame(items)
    if not df.empty:
        df = df[(df["unique_skill_count"] > 0) & df["course_title"].notna()].reset_index(drop=True)
    return df

def _tfidf(corpus: List[str]) -> Tuple[Any, TfidfVectorizer]:
    vec = TfidfVectorizer(token_pattern=r"[^|]+", lowercase=False)
    X = vec.fit_transform(corpus)
    return X, vec

def _kmeans(X, k: int, seed: int = 42) -> np.ndarray:
    k = max(1, int(k))
    model = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return model.fit_predict(X)

def _dbscan_cosine(X, eps: float = 0.35, min_samples: int = 4) -> np.ndarray:
    sim = cosine_similarity(X)
    dist = 1.0 - sim
    model = DBSCAN(eps=float(eps), min_samples=int(min_samples), metric="precomputed")
    return model.fit_predict(dist)

def _cluster_top_skills(X, labels, vectorizer, top_n: int = 12) -> Dict[int, List[Tuple[str, float, int]]]:
    feats = np.array(vectorizer.get_feature_names_out())
    info: Dict[int, List[Tuple[str, float, int]]] = {}
    for cid in sorted(set(int(l) for l in labels if int(l) != -1)):
        idx = np.where(labels == cid)[0]
        Xc = X[idx]
        avg = np.asarray(Xc.mean(axis=0)).ravel()
        top_idx = np.argsort(-avg)[:top_n]
        info[cid] = [(feats[j], float(avg[j]), 0) for j in top_idx if avg[j] > 0]
    return info

def _project_2d(X) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X.toarray())



def _pick_theme_endpoint(base: str, session: requests.Session, override: Optional[str]) -> str:
    if override:
        return override
    candidates = ["/theme-search", "/theme_search", "/themes/search", "/search/theme"]
    for c in candidates:
        url = base.rstrip("/") + c
        try:
            r = session.options(url, timeout=10)
            if r.status_code < 400:
                return c
            r = session.head(url, timeout=10)
            if r.status_code < 400:
                return c
        except requests.RequestException:
            pass
    return candidates[0]

def _theme_search_labels(base_url: str, token: Optional[str], cluster_top_skills: List[str], countries: Optional[List[str]], override_ep: Optional[str]) -> List[str]:
    s = requests.Session()
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = token
    ep = _pick_theme_endpoint(base_url, s, override_ep)
    url = base_url.rstrip("/") + ep
    payloads = [
        {"theme": " ".join(cluster_top_skills), "countries": countries},
        {"query": " ".join(cluster_top_skills), "countries": countries},
    ]
    for body in payloads:
        try:
            r = s.post(url, json=body, headers=headers, timeout=30)
            if r.status_code >= 400:
                continue
            data = r.json()
            cand = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("programs") or data.get("items") or data.get("results") or []
            else:
                items = []
            for it in items[:50]:
                for k in ("theme","topic","topic_identified","degree_title","department","program_name"):
                    v = it.get(k)
                    if isinstance(v, str):
                        vv = _clean_text(v)
                        if vv:
                            cand.append(vv)
            return cand
        except requests.RequestException:
            continue
    return []

def _label_from_candidates(candidates: List[str], top_skills: List[str]) -> Tuple[str, float]:
    """Pick a label via majority vote + skill token overlap scoring."""
    if not candidates:
        return _heuristic_label_from_skills(top_skills)
    norm = [re.sub(r"\\s+", " ", c.strip().lower()) for c in candidates if c and isinstance(c, str)]
    if not norm:
        return _heuristic_label_from_skills(top_skills)
    counts = Counter(norm)
    label, cnt = counts.most_common(1)[0]
    base_conf = cnt / max(1, len(norm))
    _, skill_conf = _heuristic_label_from_skills(top_skills)
    conf = min(1.0, 0.6*base_conf + 0.4*skill_conf)
    return label.title(), conf

def _heuristic_label_from_skills(top_skills: List[str]) -> Tuple[str, float]:
    """Fallback classifier: map keywords to a theme-like label."""
    text = " ".join([s.lower() for s in top_skills])
    rules = [
        ("Computer Science", ["python","machine learning","deep learning","algorithms","data structures","operating systems","networks","software","database","programming"]),
        ("Data Science", ["data analysis","data mining","statistics","visualisation","pandas","numpy","sql","big data"]),
        ("AI & ML", ["neural","deep learning","machine learning","nlp","computer vision","transformer"]),
        ("Cybersecurity", ["security","cryptography","malware","threat","forensics"]),
        ("Business & Economics", ["economics","marketing","finance","management","accounting","business"]),
        ("Logistics & Supply Chain", ["logistics","supply chain","inventory","transport","warehouse"]),
        ("Medicine", ["anatomy","physiology","clinic","medicine","surgery"]),
        ("Law", ["legal","law","arguments","jurisprudence"]),
        ("Education", ["teaching","curriculum","pedagogy","didactics","class"]),
        ("Art & Design", ["art","design","sketch","drawing","studio","creative"]),
        ("Architecture", ["buildings","architecture","urban","environment"]),
    ]
    best_label, best_score = "General", 0.15
    for label, kws in rules:
        score = sum(1 for k in kws if k in text) / max(1, len(kws))
        if score > best_score:
            best_label, best_score = label, score
    return best_label, float(min(1.0, best_score + 0.2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", help="Curriculum-Skills base URL (required unless --input-json used)")
    ap.add_argument("--token", help="Authorization header value, e.g. 'Bearer XXX'")
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--theme", default=None)
    ap.add_argument("--degree-types", nargs="*", default=None)
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--mode", choices=["degree","course"], default="degree", help="Aggregate per degree or per course before clustering")
    ap.add_argument("--algo", choices=["kmeans","dbscan"], default="kmeans")
    ap.add_argument("--k", type=int, default=8, help="k for kmeans")
    ap.add_argument("--eps", type=float, default=0.35, help="eps for dbscan (cosine distance)")
    ap.add_argument("--min-samples", type=int, default=4, help="min_samples for dbscan")
    ap.add_argument("--top-skills", type=int, default=12, help="Top skills per cluster in the summary")
    ap.add_argument("--project-2d", action="store_true", help="Add PCA(2D) projection columns")
    ap.add_argument("--input-json", help="Use a local biodiversity response (skips HTTP)")
    ap.add_argument("--out-csv", default="clusters.csv")
    ap.add_argument("--out-summary", default="clusters_summary.json")
    ap.add_argument("--label-with-theme", action="store_true", help="Call theme search to label clusters")
    ap.add_argument("--theme-endpoint", default=None, help="Override theme search endpoint path")
    args = ap.parse_args()

    if args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("results", payload if isinstance(payload, list) else [])
    else:
        if not args.base_url:
            raise SystemExit("--base-url is required when not using --input-json")
        rows = _fetch_all_biodiversity(args.base_url, args.countries, args.theme, args.level, args.degree_types, args.token)

    if args.mode == "degree":
        df = _aggregate_to_degrees(rows)
        id_cols = ["country","university","department","degree_type","degree_title","program_name"]
        label_col = "degree_title"
    else:
        df = _aggregate_to_courses(rows)
        id_cols = ["country","university","department","course_title"]
        label_col = "course_title"

    if df.empty or df.shape[0] < 2:
        raise SystemExit("Not enough items to cluster.")

    X, vec = _tfidf(df["unique_skills"].tolist())

    if args.algo == "kmeans":
        labels = _kmeans(X, args.k)
    else:
        labels = _dbscan_cosine(X, args.eps, args.min_samples)

    df["cluster"] = labels.astype(int)

    if args.project_2d:
        proj = _project_2d(X)
        df["x"] = proj[:,0]
        df["y"] = proj[:,1]

    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    top = _cluster_top_skills(X, labels, vec, args.top_skills)
    groups = []
    for cid, g in df.groupby("cluster"):
        if int(cid) == -1:
            members = g[id_cols].to_dict(orient="records")
            groups.append({"cluster": int(cid), "size": int(len(g)), "top_skills": [], "members": members})
        else:
            members = g[id_cols].to_dict(orient="records")
            groups.append({"cluster": int(cid), "size": int(len(g)), "top_skills": top.get(int(cid), []), "members": members, "label": (cluster_labels.get(int(cid), {}).get("label") if args.label_with_theme else None), "confidence": (cluster_labels.get(int(cid), {}).get("confidence") if args.label_with_theme else None)})


    cluster_labels = {}
    if args.label_with_theme:
        base = args.base_url
        for cid, g in df.groupby("cluster"):
            feats = np.array(vec.get_feature_names_out())
            idx = np.where(df["cluster"].values == cid)[0]
            Xc = X[idx]
            avg = np.asarray(Xc.mean(axis=0)).ravel()
            top_idx = np.argsort(-avg)[:12]
            top_skills = [feats[j] for j in top_idx if avg[j] > 0]
            cands = _theme_search_labels(base, args.token, top_skills, args.countries, args.theme_endpoint) if base else []
            label, conf = _label_from_candidates(cands, top_skills)
            cluster_labels[int(cid)] = {"label": label, "confidence": float(conf)}
        df["cluster_label"] = df["cluster"].map(lambda c: cluster_labels.get(int(c), {}).get("label"))
        df["cluster_confidence"] = df["cluster"].map(lambda c: cluster_labels.get(int(c), {}).get("confidence"))
        df.to_csv(args.out_csv, index=False, encoding="utf-8")

    summary = {
        "mode": args.mode,
        "algo": args.algo,
        "k": (args.k if args.algo == "kmeans" else None),
        "eps": (args.eps if args.algo == "dbscan" else None),
        "min_samples": (args.min_samples if args.algo == "dbscan" else None),
        "items": int(len(df)),
        "clusters": groups,
    }
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.out_csv} ({len(df)} rows) and {args.out_summary}")

if __name__ == "__main__":
    main()
