#!/usr/bin/env python3
import csv
import os
import re
import time
import argparse
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup, NavigableString
from requests.adapters import HTTPAdapter, Retry


BASE_LIST_URL = "https://spreadthesign.com/en.gb/search/by-category/398/sign-language-for-beginners/?q=&p=1"
SITE_BASE = "https://spreadthesign.com"
MEDIA_TIMEOUT = (10, 120)  # connect, read


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
        "User-Agent": "Mozilla/5.0 (compatible; SpreadTheSignScraper/1.1)"
    })
    return s


def page_url(base_with_p, page_num):
    # base already has ?p=1; just replace the p value
    return re.sub(r"([?&])p=\d+", rf"\1p={page_num}", base_with_p)


def extract_word_from_anchor(a_tag) -> str:
    # Only take direct text nodes (exclude <small>)
    texts = []
    for node in a_tag.contents:
        if isinstance(node, NavigableString):
            texts.append(str(node))
    word = "".join(texts).strip()
    word = re.sub(r"\s+", " ", word)
    return word


def is_single_word(word: str) -> bool:
    # “More than 1 word” = contains any whitespace
    return bool(word) and not re.search(r"\s", word)


def sanitize_slug(s: str) -> str:
    # Safe-ish filename
    s = s.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "word"


def parse_id_and_slug_from_href(href: str):
    # Expect /en.gb/word/17424/mining/0/?q=mining  OR /en.gb/word/17424/mining/?q=mining
    m = re.search(r"/word/(\d+)/([^/]+)/", href)
    if m:
        return m.group(1), m.group(2)
    return None, None


def scrape_list_page(session: requests.Session, url: str):
    r = session.get(url, timeout=(10, 30))
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    anchors = soup.select("div.search-result-title a[href]")
    items = []
    for a in anchors:
        word = extract_word_from_anchor(a)
        if not is_single_word(word):
            continue
        href = a.get("href", "").strip()
        if not href:
            continue
        if href.startswith("/"):
            href = urljoin(SITE_BASE, href)
        pos = a.find("small")
        pos_text = pos.get_text(strip=True) if pos else ""
        items.append({"word": word, "href": href, "pos": pos_text})
    return items


def find_video_src_on_word_page(session: requests.Session, word_url: str) -> str | None:
    r = session.get(word_url, timeout=(10, 30))
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    v = soup.select_one("div.show-result video[src]")
    if v and v.has_attr("src"):
        return v["src"].strip()
    return None


def download_mp4(session: requests.Session, url: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=MEDIA_TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
    return out_path.stat().st_size


def main():
    ap = argparse.ArgumentParser(description="Scrape single-word entries, fetch video mp4, and write CSV.")
    ap.add_argument("--start", type=int, default=1, help="Start page (inclusive)")
    ap.add_argument("--end", type=int, default=293, help="End page (inclusive)")
    ap.add_argument("--base", type=str, default=BASE_LIST_URL, help="Category URL with p=1 present")
    ap.add_argument("-o", "--out", type=str, default="spreadthesign_words_with_video.csv", help="Output CSV")
    ap.add_argument("--media-dir", type=str, default="sts_media", help="Where to save mp4 files")
    ap.add_argument("--sleep", type=float, default=0.4, help="Delay between list pages (seconds)")
    ap.add_argument("--sleep-word", type=float, default=0.2, help="Delay between word pages (seconds)")
    args = ap.parse_args()

    session = session_with_retries()
    media_root = Path(args.media_dir)
    seen = set()  # dedupe by (word_lower, id) if id known else by href

    total_rows = 0
    total_downloaded = 0

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page", "word", "pos", "word_url",
                "video_url", "mp4_path", "mp4_bytes", "status"
            ],
        )
        writer.writeheader()

        for p in range(args.start, args.end + 1):
            list_url = page_url(args.base, p)
            try:
                items = scrape_list_page(session, list_url)
            except Exception as e:
                print(f"[Page {p}] ERROR listing: {e}")
                continue

            if not items:
                print(f"[Page {p}] 0 items — stopping.")
                break

            print(f"[Page {p}] {len(items)} items")

            for it in items:
                word = it["word"]
                word_url = it["href"]
                pos = it["pos"]

                _id, slug = parse_id_and_slug_from_href(urlparse(word_url).path)
                dedupe_key = (word.lower(), _id or word_url)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                video_url = None
                mp4_path = ""
                mp4_bytes = 0
                status = "ok"

                try:
                    video_url = find_video_src_on_word_page(session, word_url)
                    if video_url:
                        # Some video src are already absolute (media.spreadthesign.com)
                        if video_url.startswith("//"):
                            video_url = "https:" + video_url
                        elif video_url.startswith("/"):
                            # Safeguard, though videos are on media. host
                            video_url = urljoin(SITE_BASE, video_url)

                        # Filename: {id}_{slug or word}.mp4
                        fname_base = f"{_id or 'noid'}_{sanitize_slug(slug or word)}.mp4"
                        out_file = media_root / fname_base
                        # Avoid re-downloading if present
                        if not out_file.exists():
                            mp4_bytes = download_mp4(session, video_url, out_file)
                            total_downloaded += 1
                        else:
                            mp4_bytes = out_file.stat().st_size
                        mp4_path = str(out_file)
                    else:
                        status = "no_video"
                except Exception as e:
                    status = f"error:{e}"

                writer.writerow({
                    "page": p,
                    "word": word,
                    "pos": pos,
                    "word_url": word_url,
                    "video_url": video_url or "",
                    "mp4_path": mp4_path,
                    "mp4_bytes": mp4_bytes,
                    "status": status,
                })
                total_rows += 1
                time.sleep(args.sleep_word)

            time.sleep(args.sleep)

    print(f"Done. Rows: {total_rows}. Videos downloaded: {total_downloaded}. CSV: {args.out}. Media dir: {media_root}")


if __name__ == "__main__":
    main()

