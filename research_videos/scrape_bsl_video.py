#!/usr/bin/env python3
import os
import re
import csv
import time
import string
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter, Retry

BASE = "https://bslsignbank.ucl.ac.uk"
SEARCH = BASE + "/dictionary/search"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_OUT = SCRIPT_DIR / "BSL_Videos"

WORD_HREF_RE = re.compile(r"^/dictionary/words/[^/]+-\d+\.html$", re.IGNORECASE)
PAGE_QS_RE = re.compile(r"[?&]page=(\d+)")
WORD_PAGE_NUM_RE = re.compile(r"(.+)-(\d+)\.html$", re.IGNORECASE)

def session_with_retries():
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD")
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-GB,en;q=0.9"
    })
    return s

def sanitize_folder(name: str) -> str:
    # keep alnum, dash, underscore, space → then tidy spaces to single underscores
    clean = re.sub(r"[^0-9A-Za-z _-]+", "", name).strip().lower()
    clean = re.sub(r"\s+", "_", clean)
    return clean or "unknown"

def get_letter_max_pages(soup: BeautifulSoup) -> int:
    max_page = 1
    for p in soup.find_all("p"):
        if "Jump to results page" in p.get_text(" ", strip=True):
            for a in p.find_all("a", href=True):
                m = PAGE_QS_RE.search(a["href"])
                if m:
                    max_page = max(max_page, int(m.group(1)))
            for s in p.find_all("strong"):
                txt = (s.get_text() or "").strip()
                if txt.isdigit():
                    max_page = max(max_page, int(txt))
            break
    return max_page

def parse_search_words(soup: BeautifulSoup):
    """Return list of absolute word URLs found in a letter results page."""
    container = soup.select_one("#searchresults")
    links = []
    if not container:
        return links
    for a in container.select("a[href]"):
        href = a["href"].strip()
        if WORD_HREF_RE.match(href):
            links.append(urljoin(BASE, href))
    return links

def fetch_soup(sess: requests.Session, url: str, **kw) -> BeautifulSoup:
    r = sess.get(url, timeout=(10, 60), **kw)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")

def get_word_name_and_max_variants(soup: BeautifulSoup) -> tuple[str, int]:
    """
    Read 'Matches for the word <em>WORD</em>' and the button group to find max variant pages.
    """
    word = "unknown"
    max_var = 1
    # Navbar area
    nav = soup.select_one("#signinfo")
    if nav:
        # word name
        em = nav.find("em")
        if em and em.get_text(strip=True):
            word = em.get_text(strip=True)
        # buttons/anchors with numbers
        for btn in nav.select(".btn-group a, .btn-group button"):
            txt = btn.get_text(strip=True)
            if txt.isdigit():
                max_var = max(max_var, int(txt))
    return word, max_var

def build_variant_urls(first_url: str, max_var: int) -> list[str]:
    """
    From '/dictionary/words/called-1.html' build all variant URLs up to max_var.
    """
    m = WORD_PAGE_NUM_RE.search(first_url)
    if not m:
        # fallback: just repeat first_url if we can't parse
        return [first_url]
    base_no_num = m.group(1)
    urls = []
    for i in range(1, max_var + 1):
        urls.append(f"{base_no_num}-{i}.html")
    return urls

def find_mp4_in_iframe(sess: requests.Session, iframe_src: str) -> str | None:
    """
    Given an iframe src like '/video/iframe/3848', open it and find a .mp4 URL.
    """
    iframe_url = urljoin(BASE, iframe_src)
    s_iframe = fetch_soup(sess, iframe_url)
    # common patterns: <source src="/media/...mp4" type="video/mp4"> or <video src="...mp4">
    for tag in s_iframe.find_all(["source", "video", "a"]):
        src = tag.get("src") or tag.get("href")
        if not src:
            continue
        if ".mp4" in src.lower():
            return urljoin(BASE, src)
    return None

def extract_mp4_url(sess: requests.Session, word_variant_url: str) -> str | None:
    """
    Open a word-variant page and return the actual mp4 URL by following the iframe.
    """
    soup = fetch_soup(sess, word_variant_url)
    # Find iframe
    iframe = soup.select_one("#videoiframe")
    if not iframe or not iframe.get("src"):
        return None
    return find_mp4_in_iframe(sess, iframe["src"])

def download_file(sess: requests.Session, url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Skip if already exists and > 0 bytes
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    ↳ Exists, skipping: {dest.name}")
        return
    with sess.get(url, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1_048_576):
                if chunk:
                    f.write(chunk)
        tmp.rename(dest)

def scrape_letter(sess: requests.Session, letter: str, delay=0.3):
    # First page: learn how many results pages
    print(f"[{letter}] Inspecting search pagination …")
    s1 = fetch_soup(sess, SEARCH, params={"query": letter, "page": 1})
    max_pages = get_letter_max_pages(s1)
    print(f"[{letter}] {max_pages} results page(s).")

    # Iterate results pages and collect word links
    word_links = parse_search_words(s1)
    print(f"[{letter}] Page 1 → {len(word_links)} words.")
    time.sleep(delay)

    for page in range(2, max_pages + 1):
        print(f"[{letter}] Fetching results page {page}/{max_pages} …")
        sp = fetch_soup(sess, SEARCH, params={"query": letter, "page": page})
        links = parse_search_words(sp)
        print(f"[{letter}] Page {page} → {len(links)} words.")
        word_links.extend(links)
        time.sleep(delay)

    # For each word link: find variants and download videos
    for idx, word_url in enumerate(word_links, start=1):
        try:
            soup_word = fetch_soup(sess, word_url)
        except requests.RequestException as e:
            print(f"[{letter}] #{idx} ERROR opening word page {word_url}: {e}")
            time.sleep(delay)
            continue

        word_name, max_var = get_word_name_and_max_variants(soup_word)
        safe = sanitize_folder(word_name)
        out_dir = ROOT_OUT / safe
        print(f"[{letter}] #{idx}/{len(word_links)} Word: {word_name!r} → {max_var} variant(s). Folder: {out_dir}")

        # Build all variant URLs from first URL
        variant_urls = build_variant_urls(word_url, max_var)

        for vnum, vurl in enumerate(variant_urls, start=1):
            mp4_url = None
            try:
                mp4_url = extract_mp4_url(sess, vurl)
            except requests.RequestException as e:
                print(f"    [var {vnum}] ERROR fetching {vurl}: {e}")
                time.sleep(delay)
                continue

            if not mp4_url:
                print(f"    [var {vnum}] No MP4 found at {vurl} (iframe/source missing).")
                time.sleep(delay)
                continue

            filename = f"{vnum:03d}.mp4"
            dest = out_dir / filename
            print(f"    [var {vnum}] Downloading → {filename}  ({mp4_url})")
            try:
                download_file(sess, mp4_url, dest)
            except requests.RequestException as e:
                print(f"    [var {vnum}] ERROR downloading MP4: {e}")
            time.sleep(delay)

def main():
    ROOT_OUT.mkdir(parents=True, exist_ok=True)
    sess = session_with_retries()

    # Loop A–Z. If you want a subset, e.g. letters = "ABC"
    letters = string.ascii_uppercase

    for letter in letters:
        try:
            scrape_letter(sess, letter, delay=0.3)
        except KeyboardInterrupt:
            print("\n[!] Interrupted by user.")
            break
        except Exception as e:
            print(f"[{letter}] Unhandled error: {e}")

    print("\n[✓] All done.")
    print(f"[→] Videos saved under: {ROOT_OUT}")

if __name__ == "__main__":
    main()

