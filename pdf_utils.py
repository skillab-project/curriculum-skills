import os
import requests
import re
import glob
import json
import time
import fitz 
import shutil

from concurrent.futures import ThreadPoolExecutor
from gradio_client import Client
from thefuzz import fuzz, process


def extract_text_from_pdf(pdf_file_path: str) -> list[str]:
    """
    Extract plain text from each page using PyMuPDF.
    Returns a list of page strings (may be empty on scanned/secured PDFs).
    """
    page_texts: list[str] = []
    try:
        doc = fitz.open(pdf_file_path)
        try:
            def grab(pn: int) -> str:
                try:
                    return doc[pn].get_text("text") or ""
                except Exception:
                    return ""

            with ThreadPoolExecutor() as executor:
                page_texts = list(executor.map(grab, range(len(doc))))
        finally:
            doc.close()
    except Exception as e:
        print(f"[ERROR] extract_text_from_pdf failed: {e}")
        return []
    return page_texts



def _clean_pdf_text(s: str) -> str:
    s = re.sub(r'-\s*\n', '', s)
    s = re.sub(r'\s*\n+\s*', '\n', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()

def _extract_text_fallback(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        return pdfminer_extract_text(pdf_path) or ""
    except Exception:
        return ""

def extract_text_from_pdf_best(pdf_path: str) -> str:
    pages = extract_text_from_pdf(pdf_path)
    txt = " ".join(pages) if isinstance(pages, list) else str(pages or "")
    if len(txt.strip()) >= 1500:
        return txt
    alt = _extract_text_fallback(pdf_path)
    return alt if len(alt.strip()) > len(txt.strip()) else txt

def _file_pdf_path(pdf_name: str) -> str:
    if os.path.isabs(pdf_name) and os.path.exists(pdf_name):
        return pdf_name
    curriculum_folder = "curriculum"
    os.makedirs(curriculum_folder, exist_ok=True)
    matches = [f for f in os.listdir(curriculum_folder) if f.endswith(".pdf") and pdf_name in f]
    if not matches:
        raise HTTPException(status_code=404, detail="PDF not found")
    return os.path.join(curriculum_folder, matches[0])

def _extract_text_pdfminer(pdf_path: str) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        return pdfminer_extract_text(pdf_path) or ""
    except Exception:
        return ""

def _extract_text_ocr(pdf_path: str, max_pages=20, dpi=200) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""
    try:
        lang = os.getenv("TESS_LANGS", "eng+deu")
        images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
        texts = []
        for img in images:
            try:
                texts.append(pytesseract.image_to_string(img, lang=lang))
            except Exception:
                texts.append(pytesseract.image_to_string(img))
        return "\n".join(texts)
    except Exception:
        return ""

def _clean_pdf_text(s: str) -> str:
    s = re.sub(r'-\s*\n', '', s)
    s = re.sub(r'\s*\n+\s*', '\n', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s.strip()

def _detect_lang(text: str) -> str:
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def _translate_hf(text: str) -> str:
    import requests
    token = os.getenv("HF_API_TOKEN", "").strip()
    model = os.getenv("HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en")
    if not token:
        return text
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    out = []
    for ch in chunks:
        try:
            r = requests.post(url, headers=headers, json={"inputs": ch}, timeout=60)
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, list) and payload and "translation_text" in payload[0]:
                out.append(payload[0]["translation_text"])
            else:
                out.append(ch)
        except Exception:
            out.append(ch)
    return " ".join(out)

def _chunk_text(s: str, chunk_size=1200, overlap=150, limit=40000):
    s = s[:limit]
    chunks, i = [], 0
    while i < len(s):
        j = min(i + chunk_size, len(s))
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = max(j - overlap, j)
    return chunks

def _merge_ner(outputs):
    seen, merged = set(), []
    for item in outputs:
        key = (item.get("class_or_confidence"), item.get("token"))
        if key and item.get("token") and key not in seen:
            merged.append(item)
            seen.add(key)
    return merged
