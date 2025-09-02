import json
import time
from datetime import datetime, timezone
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "APO/1.0 (+noncommercial; contact: admin@example.org)"}

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.text

# =========== OFICJALNE ===========

def fetch_isap_headlines() -> List[Dict]:
    # Prosty przykład; w praktyce dopasuj selektory do realnego HTML/atom (jeśli dostępny)
    url = "https://isap.sejm.gov.pl/isap.nsf/home.xsp"
    html = _safe_get(url)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a"):
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if not title or not href:
            continue
        # Filtr minimalny: słowa kluczowe
        if any(kw in title.lower() for kw in ["oświat", "edukac", "nauczyciel"]):
            out.append({
                "title": title,
                "link": href if href.startswith("http") else f"https://isap.sejm.gov.pl{href}",
                "date": datetime.utcnow().date().isoformat(),
                "source": "ISAP",
                "source_type": "official",
                "summary": ""
            })
    return out[:10]

def fetch_rcl_dz_u() -> List[Dict]:
    url = "https://dziennikustaw.gov.pl/"
    html = _safe_get(url)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a"):
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if not title or not href:
            continue
        if any(kw in title.lower() for kw in ["oświat", "edukac", "nauczyciel", "minister edukacji"]):
            out.append({
                "title": title,
                "link": href if href.startswith("http") else f"https://dziennikustaw.gov.pl{href}",
                "date": datetime.utcnow().date().isoformat(),
                "source": "Dziennik Ustaw (RCL)",
                "source_type": "official",
                "summary": ""
            })
    return out[:10]

def fetch_men_news() -> List[Dict]:
    url = "https://www.gov.pl/web/edukacja/aktualnosci"
    html = _safe_get(url)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for art in soup.select("article a"):
        title = art.get_text(strip=True)
        href = art.get("href", "")
        if not title or not href:
            continue
        out.append({
            "title": title,
            "link": href if href.startswith("http") else f"https://www.gov.pl{href}",
            "date": datetime.utcnow().date().isoformat(),
            "source": "MEN (gov.pl)",
            "source_type": "official",
            "summary": ""
        })
    return out[:10]

# =========== NIEOFICJALNE ===========

def fetch_infor_oswiata() -> List[Dict]:
    url = "https://www.infor.pl/prawo/oswiata/"
    html = _safe_get(url)
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for a in soup.select("a"):
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if not title or not href:
            continue
        if "/prawo/oswiata/" in href:
            out.append({
                "title": title,
                "link": href if href.startswith("http") else f"https://www.infor.pl{href}",
                "date": datetime.utcnow().date().isoformat(),
                "source": "Infor.pl – Prawo oświatowe",
                "source_type": "unofficial",
                "summary": ""
            })
    return out[:10]

# =========== ORKIESTRACJA ===========

def refresh_all() -> Dict:
    items: List[Dict] = []
    for fetcher in (fetch_isap_headlines, fetch_rcl_dz_u, fetch_men_news, fetch_infor_oswiata):
        try:
            items.extend(fetcher())
            time.sleep(0.5)  # grzeczne odpytywanie
        except Exception:
            # nie przerywaj całego procesu jeśli jedno źródło padnie
            continue

    # deduplikacja po (title, link)
    seen = set()
    uniq = []
    for it in items:
        key = (it["title"], it["link"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    return {
        "updated_at": _iso_now(),
        "items": uniq
    }