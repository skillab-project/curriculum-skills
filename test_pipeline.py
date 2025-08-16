from playwright.sync_api import sync_playwright
from gradio_client import Client
from bs4 import BeautifulSoup
from langdetect import detect
import re
import json
import os
import string
from urllib.parse import urlparse
import trafilatura
import sys
import io
import requests

COOKIE_KEYWORDS = [
    "cookie", "consent", "gdpr", "privacy", "your choices", "manage preferences",
    "accept all", "allow all", "i agree", "agree to", "use of cookies"
]

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "").strip()
HF_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-mul-en"

def strip_cookie_banners(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.select("""
        #cookie-banner, .cookie-consent, .cookie-policy, .cookie, .cookies,
        #onetrust-banner-sdk, #onetrust-consent-sdk, .ot-sdk-container, .ot-sdk-row,
        #truste-consent-button, .truste_box_overlay,
        #CybotCookiebotDialog, #CybotCookiebotDialogBody, #CybotCookiebotDialogBodyLevelButtonAccept,
        .cmp-ui, .cmpbox, .qc-cmp2-container, .didomi-popup, .didomi-consent-popup
    """):
        el.decompose()
    TEXT_RE = re.compile(r"|".join(re.escape(k) for k in COOKIE_KEYWORDS), re.I)
    for el in list(soup.find_all(True)):
        try:
            bag = " ".join([
                str(el.get("id", "")),
                " ".join(el.get("class", []) or []),
                str(el.get("role", "")),
                (el.get_text(" ", strip=True)[:250] or "")
            ]).lower()
            if TEXT_RE.search(bag):
                el.decompose()
        except Exception:
            pass
    return str(soup)

def translate_text_api(text):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    api_url = f"https://api-inference.huggingface.co/models/{HF_TRANSLATION_MODEL}"
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    translated_chunks = []
    for chunk in chunks:
        response = requests.post(api_url, headers=headers, json={"inputs": chunk})
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0 and "translation_text" in result[0]:
                translated_chunks.append(result[0]["translation_text"])
            else:
                translated_chunks.append(chunk)
        else:
            translated_chunks.append(chunk)
    return " ".join(translated_chunks)

def extract_structured_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    text_blocks = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "span", "div"]):
        txt = tag.get_text(strip=True)
        if txt:
            text_blocks.append(txt)
    for row in soup.select("table tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        line = " | ".join(cells)
        if line:
            text_blocks.append(line)
    return "\n".join(text_blocks)

def is_table_based(html):
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all("table")) > 3

def detect_and_translate(text):
    try:
        lang = detect(text)
        if lang != "en":
            return translate_text_api(text)
        else:
            return text
    except Exception:
        return text

def extract_country_from_url(url):
    domain = urlparse(url).netloc.lower()
    country_map = {
        ".nl": "Netherlands",".gr": "Greece",".de": "Germany",".fr": "France",".it": "Italy",
        ".es": "Spain",".pt": "Portugal",".pl": "Poland",".fi": "Finland",".se": "Sweden",
        ".no": "Norway",".dk": "Denmark",".be": "Belgium",".at": "Austria",".cz": "Czech Republic",
        ".sk": "Slovakia",".ro": "Romania",".bg": "Bulgaria",".hu": "Hungary",".ch": "Switzerland",
        ".ie": "Ireland",".uk": "United Kingdom",".lu": "Luxembourg",".lt": "Lithuania",
        ".lv": "Latvia",".ee": "Estonia"
    }
    for tld, country in country_map.items():
        if domain.endswith(tld):
            return country
    if "rug.nl" in domain:
        return "Netherlands"
    elif "auth.gr" in domain:
        return "Greece"
    elif "uni-" in domain:
        return "Germany"
    return "Unknown"

def clean_text(text):
    noise_patterns = [
        r"skip to content", r"skip to navigation", r"login", r"search",
        r"you need to enable javascript.*?", r"page not found \(404\)",
        r"back to the homepage", r"\b(home|catalog|sitemap)\b"
    ]
    for pattern in noise_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def get_structured_text_from_url(url, return_html=False):
    def deduplicate_lines(text):
        seen = set()
        deduped = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower() not in seen:
                deduped.append(line)
                seen.add(line.lower())
        return "\n".join(deduped)

    def deduplicate_phrases(text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        seen = set()
        deduped = []
        for s in sentences:
            norm = s.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(s.strip())
        return " ".join(deduped)

    ws = os.getenv("BROWSERLESS_WS")
    if not ws:
        raise RuntimeError("BROWSERLESS_WS environment variable is required")

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(ws)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(3000)
        html = page.content()
        page.close()
        try:
            browser.close()
        except Exception:
            pass

    soup = BeautifulSoup(html, "html.parser")
    for el in soup.select('#cookie-banner, .cookie-consent, .cookie-policy'):
        el.decompose()
    html = strip_cookie_banners(html)

    narrative = trafilatura.extract(html, include_comments=False, include_tables=False)
    narrative = clean_text(narrative) if narrative else ""

    tables = soup.find_all("table")
    table_rows = []
    for table in tables:
        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if cells:
                table_rows.append(" | ".join(cells))
    structured_tables = "\n".join(table_rows)

    additional_kvs = extract_additional_key_values(soup)
    structured_data_block = clean_text("\n".join([structured_tables, additional_kvs]))
    plain_fallback = extract_filtered_text(soup)

    merged_parts = [narrative, structured_data_block, plain_fallback]
    merged_text = "\n\n".join(part for part in merged_parts if part)
    deduplicated_text = deduplicate_phrases(deduplicate_lines(merged_text))

    if return_html:
        return deduplicated_text, extract_title(html), html
    else:
        return deduplicated_text, extract_title(html)

def extract_additional_key_values(soup):
    kv_pairs = []
    for p in soup.find_all("p"):
        strong_or_b = p.find(["strong", "b"])
        if strong_or_b and strong_or_b.string:
            key = strong_or_b.get_text(strip=True).rstrip(":").strip()
            value = strong_or_b.next_sibling
            if value and isinstance(value, str):
                kv_pairs.append(f"{key} | {value.strip()}")
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            key = dt.get_text(strip=True)
            value = dd.get_text(strip=True)
            kv_pairs.append(f"{key} | {value}")
    for li in soup.find_all("li"):
        text = li.get_text(" ", strip=True)
        if ":" in text:
            parts = text.split(":", 1)
            if len(parts) == 2:
                kv_pairs.append(f"{parts[0].strip()} | {parts[1].strip()}")
    return "\n".join(kv_pairs)

def extract_filtered_text(soup):
    for tag in soup.select("script, style, nav, header, footer, aside, form"):
        tag.decompose()
    noise_keywords = ["menu", "nav", "header", "footer", "sidebar", "login", "search", "breadcrumb", "cookie"]
    to_remove = []
    for tag in soup.find_all(True):
        if not hasattr(tag, "attrs"):
            continue
        id_attr = tag.get("id", "")
        class_attr = tag.get("class", [])
        combined = ' '.join([id_attr] + class_attr).lower() if class_attr else str(id_attr).lower()
        if any(n in combined for n in noise_keywords):
            to_remove.append(tag)
    for tag in to_remove:
        tag.decompose()
    blocks = soup.find_all(["main", "article", "section"]) + soup.find_all("div", class_=re.compile(r"(main|content|course|description)", re.I))
    if blocks:
        texts = [b.get_text(" ", strip=True) for b in blocks if b.get_text(strip=True)]
        return "\n\n".join(texts)
    lines = (line.strip() for line in soup.get_text().splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

def extract_title(html):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    heading = ""
    for htag in ["h1", "h2", "h3"]:
        h = soup.find(htag)
        if h and h.get_text(strip=True):
            heading = h.get_text(strip=True)
            break
    return heading if len(heading) > 5 else title

def send_to_curricunlp(text):
    client = Client("marfoli/CurricuNLP")
    return client.predict(text, api_name="/predict")

def group_entities(ner_output):
    grouped = {}
    for item in ner_output:
        label = item.get("class_or_confidence")
        token = item.get("token", "").strip()
        if not label or not token:
            continue
        grouped.setdefault(label, []).append(token)
    cleaned = {}
    for label, tokens in grouped.items():
        deduped = []
        prev = None
        for tok in tokens:
            if tok != prev:
                deduped.append(tok)
            prev = tok
        if label == "professor":
            cleaned[label] = deduped
        else:
            cleaned[label] = " ".join(deduped).strip()
    return cleaned

def get_university_from_local_db(url, local_db_path="world_universities_and_domains.json"):
    domain = urlparse(url).netloc.lower()
    if not os.path.exists(local_db_path):
        return None
    try:
        with open(local_db_path, "r", encoding="utf-8") as f:
            universities = json.load(f)
    except Exception:
        return None
    for entry in universities:
        domains = entry.get("domains", [])
        for known_domain in domains:
            if domain == known_domain or domain.endswith(f".{known_domain}"):
                return entry.get("name")
    return None

def postprocess_fields(data, full_text=None, page_title=None):
    if full_text:
        section_map = {
            "learning outcomes": "learning_outcomes",
            "objectives": "objectives",
            "description": "description",
            "course content": "course_content",
            "content": "course_content",
            "year of study": "year",
            "year": "year",
            "last changed": None,
            "part of": None,
            "remarks": None
        }
        label_regex = r"(?i)(?P<label>" + "|".join(re.escape(l) for l in section_map) + r")\s*[:\-]?\s*"
        matches = list(re.finditer(label_regex, full_text))
        for i, match in enumerate(matches):
            label_text = match.group("label").lower()
            mapped_label = section_map.get(label_text)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            if mapped_label and (mapped_label not in data or len(data[mapped_label]) < 20):
                span = full_text[start:end].strip()
                data[mapped_label] = span
    return data

def save_to_json(data, filename="curriculum_output.json"):
    university = re.sub(r"[^\w\-]", "_", data.get("university", "UnknownUniversity"))
    output_dir = os.path.join("output", university)
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved extracted data to: {filepath}")

def clean_field_text(text):
    return re.sub(r'\s{2,}', ' ', text.strip().replace("\n", " "))

def remove_repeated_words(text):
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

def clean_fields(data, page_title=None, full_text=None):
    for key, value in data.items():
        if isinstance(value, str):
            value = remove_repeated_words(value)
            value = re.sub(r'\s{2,}', ' ', value).strip(" ,.;:-")
            data[key] = value
    if "language" in data:
        langs = re.findall(r"(English|German|Deutsch|Dutch|French|Greek|Spanish|Swedish|Finnish)", data["language"], re.IGNORECASE)
        if langs:
            data["language"] = ", ".join(sorted(set(l.title() for l in langs)))
        else:
            del data["language"]
    if "hours" in data:
        if not re.search(r"\d.*?(hpw|hour|hrs)", data["hours"], re.IGNORECASE):
            del data["hours"]
    if "lesson_name" not in data or len(data["lesson_name"]) < 5:
        if page_title and len(page_title.strip()) > 5:
            data["lesson_name"] = page_title.strip()
    if full_text:
        degree_titles = []
        degree_abbrs = [
            "BA", "MA", "BSc", "MSc", "MBA", "BBA", "MEng", "BEng", "LLB", "LLM",
            "Bachelor(?:['’s])?", "Master(?:['’s])?", "BAcc", "BFin", "MFin"
        ]
        pattern = r"\b(?P<type>" + "|".join(degree_abbrs) + r")\b\s*(?:of|in)?\s+(?P<field>[A-Za-z &/\-]{3,}(?:\s+[A-Za-z &/\-]{2,}){0,5})"
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            deg_type = match.group("type").strip()
            field = match.group("field").strip(string.punctuation + string.whitespace)
            field = re.split(r"\b(joint degree|Objectives|Add to|or|with|credit)\b", field, flags=re.IGNORECASE)[0].strip()
            deg_type_lower = deg_type.lower()
            if any(d in deg_type_lower for d in ["bachelor", "ba", "bsc", "bacc", "bfin", "beng", "bba", "b.a.", "llb"]):
                deg_type_norm = "BSc"
            elif any(d in deg_type_lower for d in ["master", "msc", "ma", "mba", "meng", "mfin", "llm"]):
                deg_type_norm = "MSc"
            else:
                continue
            full_title = f"{deg_type_norm} {field}"
            irrelevant_words = {"fees", "funding", "information", "overview", "course", "program", "credit", "add", "part"}
            if field and len(field.split()) <= 6 and not any(word.lower() in field.lower() for word in irrelevant_words):
                if len(field.split()) == 1 and len(field) < 4:
                    continue
                degree_titles.append((deg_type_norm, full_title))
        if degree_titles:
            degree_by_type = {"BSc": [], "MSc": []}
            for deg_type, title in degree_titles:
                if deg_type in degree_by_type:
                    degree_by_type[deg_type].append(title)
            degree_by_type = {k: sorted(list(set(v))) for k, v in degree_by_type.items() if v}
            if degree_by_type:
                data["degree_titles"] = degree_by_type
                data["msc_bsc"] = sorted(degree_by_type.keys())
        mand_opt_found = set()
        if re.search(r'\b(mandatory|compulsory|core course)\b', full_text, re.IGNORECASE):
            mand_opt_found.add("Mandatory")
        if re.search(r'\b(optional|option)\b', full_text, re.IGNORECASE):
            mand_opt_found.add("Optional")
        if re.search(r'\b(elective|choice)\b', full_text, re.IGNORECASE):
            mand_opt_found.add("Elective")
        if "mandatory" in data:
            mandatory_text = data["mandatory"].lower()
            if any(w in mandatory_text for w in ['mandatory', 'compulsory', 'yes', 'core']):
                mand_opt_found.add("Mandatory")
            if 'optional' in mandatory_text:
                mand_opt_found.add("Optional")
            if 'elective' in mandatory_text:
                mand_opt_found.add("Elective")
            del data["mandatory"]
        if mand_opt_found:
            data["mand_opt"] = sorted(list(mand_opt_found))
    if "website" in data:
        value = data["website"].strip()
        if not re.match(r"^(https?://|www\.)", value, re.IGNORECASE):
            del data["website"]
    if "year" in data:
        matches = re.findall(r"\b(?:Year\s*)?(1|2|3|4|19\d{2}|20\d{2}|2100)\b", data["year"], re.IGNORECASE)
        if matches:
            data["year"] = matches[0]
        else:
            del data["year"]
    if full_text and "year" not in data:
        match = re.search(r"\b(?:Year\s*)?(1|2|3|4|19\d{2}|20\d{2}|2100)\b", full_text, re.IGNORECASE)
        if match:
            data["year"] = match.group(1)
    if "professor" in data and isinstance(data["professor"], list):
        cleaned_profs = []
        for prof in data["professor"]:
            prof = re.sub(r"\b(Lecturer|Coordinator|Co-ordinator)\b", "", prof, flags=re.IGNORECASE).strip(" ,.;:-()")
            if len(prof) > 2:
                cleaned_profs.append(prof)
        data["professor"] = sorted(set(cleaned_profs))
    for field in ["description", "objectives"]:
        if field in data and len(data[field]) < 30:
            del data[field]
    fee_pattern = r"""
        (?:
            tuition\s+fee[s]?|fee[s]?|cost|price|amount
        )
        [\s:]*
        (?P<amount>
            (?:[€£$]\s?)?
            \d{1,3}(?:[.,\s]\d{3})*
            (?:[.,]\d{2})?
            (?:\s?(EUR|USD|GBP))?
        )
     """
    matches = re.findall(fee_pattern, full_text or "", flags=re.IGNORECASE | re.VERBOSE)
    fee_values = [m[0].strip() for m in matches if re.search(r'\d', m[0]) and len(m[0]) > 4]
    if fee_values:
        def normalize_fee(fee):
            fee = fee.replace("\u202f", "").replace(" ", "")
            fee = fee.replace(",", "") if fee.count(",") > 1 else fee
            return fee
        data["fee"] = sorted(set(normalize_fee(fee) for fee in fee_values))
    if full_text:
        ects_values = re.findall(r'\b(\d{1,3})\s*(?:ECTS|ects|credit|credits)\b', full_text)
        if ects_values:
            unique_sorted_ects = sorted(set(int(e) for e in ects_values))
            data["ects"] = unique_sorted_ects
    for k in list(data):
        if isinstance(data.get(k), str):
            data[k] = re.sub(r'\s{2,}', ' ', data[k].strip())
    return data

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import hashlib
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://www.gla.ac.uk/undergraduate/degrees/accountancy/"
    visible_text, page_title, raw_html = get_structured_text_from_url(url, return_html=True)
    visible_text = clean_field_text(visible_text)
    visible_text = detect_and_translate(visible_text)
    raw_ner_output = send_to_curricunlp(visible_text)
    lesson_names = [item["token"] for item in raw_ner_output if item.get("class_or_confidence") == "lesson_name"]
    lesson_names = sorted(set(lesson_names))
    if not lesson_names:
        lesson_names = [None]
    for i, lesson in enumerate(lesson_names):
        filtered_output = []
        for item in raw_ner_output:
            if lesson is None or item.get("class_or_confidence") != "lesson_name" or item["token"] == lesson:
                filtered_output.append(item)
        structured_data = group_entities(filtered_output)
        structured_data = postprocess_fields(structured_data, full_text=visible_text, page_title=page_title)
        structured_data = clean_fields(structured_data, page_title=page_title, full_text=visible_text)
        structured_data["url"] = url
        structured_data["country"] = extract_country_from_url(url)
        university_name = get_university_from_local_db(url)
        structured_data["university"] = university_name or "UnknownUniversity"
        lesson_name_safe = re.sub(r"[^\w\-]", "_", structured_data.get("lesson_name", f"NoLesson_{i}"))
        university_safe = re.sub(r"[^\w\-]", "_", university_name or "UnknownUniversity")
        filename = f"{university_safe}_{lesson_name_safe}.json"
        output_dir = os.path.join("output", university_safe)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        j = 1
        while os.path.exists(filepath):
            filename = f"{university_safe}_{lesson_name_safe}_{j}.json"
            filepath = os.path.join(output_dir, filename)
            j += 1
        save_to_json(structured_data, filename=filename)
