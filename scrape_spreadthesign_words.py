#!/usr/bin/env python3
import csv
import re
import sys
import time
import argparse
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import requests
from bs4 import BeautifulSoup, NavigableString
from requests.adapters import HTTPAdapter, Retry


CATEGORY_URL = "https://spreadthesign.com/en.gb/search/by-category/398/sign-language-for-beginners/?q=&p=1"


def session_with_retries():
    s = requests.Session()
    retries = Retry(
        total=5, connect=5, read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD")
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; SpreadTheSignScraper/1.0)"
    })
    return s


def set_page(url: str, page: int) -> str:
    """Return url with p=page."""
    parts = list(urlparse(url))
    qs = parse_qs(parts[4], keep_blank_values=True)
    qs["p"] = [str(page)]
    parts[4] = urlencode(qs, doseq=True)
    return urlunparse(parts)


def extract_word_from_anchor(a_tag) -> str:
    """
    Pull only the visible word text (exclude the <small> POS).
    We take just the text nodes directly inside <a>.
    """
    texts = []
    for node in a_tag.contents:
        if isinstance(node, NavigableString):
            texts.append(str(node))
    word = "".join(texts).strip()
    # Normalise internal whitespace
    word = re.sub(r"\s+", " ", word)
    return word


def is_single_word(word: str) -> bool:
    """Single word = no whitespace characters."""
    return bool(word) and not re.search(r"\s", word)


def scrape_page(s: requests.Session, url: str):
    r = s.get(url, timeout=(10, 30))
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    rows = soup.select("div.search-result-title a[href]")
    results = []
    for a in rows:
        word = extract_word_from_anchor(a)
        if not is_single_word(word):
            continue
        href = a.get("href", "").strip()
        # Make absolute
        if href.startswith("/"):
            href = f"https://spreadthesign.com{href}"
        # Optional: grab POS if you need it later
        pos = a.find("small")
        pos_text = pos.get_text(strip=True) if pos else ""
        results.append({"word": word, "url": href, "pos": pos_text})
    return results


def main():
    ap = argparse.ArgumentParser(description="Scrape single-word entries from SpreadTheSign category pages.")
    ap.add_argument("--start", type=int, default=1, help="Start page (default: 1)")
    ap.add_argument("--end", type=int, default=293, help="End page inclusive (default: 293)")
    ap.add_argument("--base", type=str, default=CATEGORY_URL, help="Base URL with p=1 already set")
    ap.add_argument("-o", "--out", type=str, default="spreadthesign_words.csv", help="Output CSV file")
    ap.add_argument("--sleep", type=float, default=0.5, help="Delay between pages (seconds)")
    args = ap.parse_args()

    s = session_with_retries()

    total = 0
    seen = set()  # dedupe by (word, url)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "url", "pos", "page"])
        writer.writeheader()

        for p in range(args.start, args.end + 1):
            page_url = set_page(args.base, p)
            try:
                items = scrape_page(s, page_url)
            except requests.HTTPError as e:
                print(f"[{p}] HTTP error {e}. Stopping.", file=sys.stderr)
                break
            except Exception as e:
                print(f"[{p}] Error {e}. Skipping page.", file=sys.stderr)
                continue

            if not items:
                # If a page returns no results, assume we've reached the end.
                print(f"[{p}] 0 results. Stopping.")
                break

            # write rows
            page_count = 0
            for it in items:
                key = (it["word"].lower(), it["url"])
                if key in seen:
                    continue
                seen.add(key)
                it["page"] = p
                writer.writerow(it)
                page_count += 1
                total += 1

            print(f"[{p}] saved {page_count} (running total {total})")

            time.sleep(args.sleep)

    print(f"Done. Saved {total} single-word entries to {args.out}")


if __name__ == "__main__":
    main()

