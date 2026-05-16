"""
scrape_shl.py — Standalone scraper for SHL Individual Test Solutions
Fetches all pages of:
    https://www.shl.com/solutions/products/product-catalog/?start=N&type=1
Then enriches each item with its detail page (description, job levels, languages).
Writes catalog.json.

Usage:
    python scrape_shl.py

Requires: requests, beautifulsoup4, lxml
"""

import json, time, re, sys
import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
LIST_URL = BASE + "/solutions/products/product-catalog/?start={start}&type=1"
OUT_FILE = "catalog.json"
PAGE_SIZE = 12
DELAY = 1.0  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    resp = session.get(url, headers=HEADERS, timeout=25)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def parse_list_page(soup: BeautifulSoup) -> list[dict]:
    """Extract assessment rows from a catalog list page."""
    items = []
    # SHL renders a table; rows typically have class "catalogue-row" or similar
    # Selector strategy: find all <tr> that contain an <a> to a product page
    rows = soup.select("table tbody tr")

    for row in rows:
        a_tag = row.find("a", href=True)
        if not a_tag:
            continue

        href = a_tag["href"]
        if not href.startswith("/"):
            continue  # skip external links

        name = a_tag.get_text(strip=True)
        url = BASE + href

        # Test type icons — SHL marks them with filled/empty glyphs or CSS classes
        # Columns after the name: Remote Testing | Adaptive/IRT | Test Types (A/B/C/D/E/K/M/P/S)
        tds = row.find_all("td")
        test_types = []
        remote = False
        adaptive = False

        for idx, td in enumerate(tds):
            cell_text = td.get_text(strip=True)
            # Detect filled indicator (● or similar Unicode, or a non-empty class)
            has_dot = bool(re.search(r"[●•✓✔]", cell_text))

            # Column 0 = name, 1 = remote testing, 2 = adaptive/IRT, 3+ = type letters
            if idx == 1 and has_dot:
                remote = True
            elif idx == 2 and has_dot:
                adaptive = True
            elif idx >= 3:
                # Some pages use single letters A-S; others use filled/empty glyphs per column
                if re.match(r"^[A-Z]$", cell_text):
                    test_types.append(cell_text)
                elif has_dot:
                    # Try to derive type from column header
                    pass

        # Fallback: scrape type from CSS classes on <span> inside cells
        for span in row.select("span[class]"):
            cls = " ".join(span.get("class", []))
            m = re.search(r"\b([ABCDEKMS])\b", cls.upper())
            if m and m.group(1) not in test_types:
                test_types.append(m.group(1))

        primary = test_types[0] if test_types else "A"

        items.append({
            "name": name,
            "url": url,
            "test_type": primary,
            "test_types": test_types,
            "remote_testing": remote,
            "adaptive": adaptive,
            "description": "",
            "job_levels": [],
            "languages": [],
        })

    return items


def enrich(item: dict, session: requests.Session) -> dict:
    """Fetch the product detail page and fill in description/job_levels/languages."""
    try:
        soup = get_soup(item["url"], session)

        # ── Description ──────────────────────────────────────────────────────
        # Try meta description first (most reliable)
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and meta.get("content"):
            item["description"] = meta["content"][:700]

        # Also try first substantial paragraph in content area
        for sel in [
            ".product-catalogue-detail__description p",
            ".product-description p",
            "article p",
            "main p",
        ]:
            paras = soup.select(sel)
            if paras:
                text = paras[0].get_text(strip=True)
                if len(text) > 60:
                    item["description"] = text[:700]
                    break

        # ── Job levels ───────────────────────────────────────────────────────
        # Look for a section labeled "Job Level" or "Levels"
        label = soup.find(
            lambda t: t.name in ("dt", "th", "strong", "p", "span")
            and re.search(r"job.?level", t.get_text(), re.I)
        )
        if label:
            sib = label.find_next(["dd", "td", "p", "span"])
            if sib:
                raw = sib.get_text(separator=", ")
                item["job_levels"] = [s.strip() for s in re.split(r"[,\n]", raw) if s.strip()]

        # ── Languages ────────────────────────────────────────────────────────
        lang_label = soup.find(
            lambda t: t.name in ("dt", "th", "strong", "p", "span")
            and re.search(r"language", t.get_text(), re.I)
        )
        if lang_label:
            sib = lang_label.find_next(["dd", "td", "p", "span"])
            if sib:
                raw = sib.get_text(separator=", ")
                item["languages"] = [s.strip() for s in re.split(r"[,\n]", raw) if s.strip()]

        # ── Test type from page (more reliable than list page) ───────────────
        type_label = soup.find(
            lambda t: t.name in ("dt", "th", "strong", "p", "span")
            and re.search(r"test.?type|assessment.?type", t.get_text(), re.I)
        )
        if type_label:
            sib = type_label.find_next(["dd", "td", "p", "span"])
            if sib:
                raw = sib.get_text(separator=" ")
                found = re.findall(r"\b([ABCDEKMS])\b", raw.upper())
                if found:
                    item["test_types"] = list(dict.fromkeys(found))  # dedupe, preserve order
                    item["test_type"] = item["test_types"][0]

    except Exception as e:
        print(f"  [warn] enrich failed for {item['name']}: {e}")

    return item


def main():
    session = requests.Session()
    all_items: list[dict] = []
    seen_urls: set[str] = set()

    print("=== SHL Catalog Scraper ===\n")
    print("Phase 1: list pages")

    start = 0
    while True:
        url = LIST_URL.format(start=start)
        print(f"  GET {url}")

        try:
            soup = get_soup(url, session)
        except Exception as e:
            print(f"  [error] {e} — stopping pagination")
            break

        items = parse_list_page(soup)

        if not items:
            print(f"  No items at start={start}, pagination complete.")
            break

        new = [i for i in items if i["url"] not in seen_urls]
        if not new:
            print(f"  All items seen at start={start}, stopping.")
            break

        for i in new:
            seen_urls.add(i["url"])

        all_items.extend(new)
        print(f"  +{len(new)} items (total {len(all_items)})")
        start += PAGE_SIZE
        time.sleep(DELAY)

    if not all_items:
        print("\n[warn] No items scraped. Check the page structure or try later.")
        sys.exit(1)

    print(f"\nPhase 2: enriching {len(all_items)} detail pages")
    for idx, item in enumerate(all_items):
        print(f"  [{idx+1}/{len(all_items)}] {item['name'][:60]}")
        all_items[idx] = enrich(item, session)
        time.sleep(DELAY * 0.5)

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(all_items)} assessments → {OUT_FILE}")


if __name__ == "__main__":
    main()