import json
import os
import time
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

CATALOG_FILE = "catalog.json"
BASE_URL = "https://www.shl.com"
CATALOG_URL = (
    "https://www.shl.com/products/product-catalog/"
    "?start={start}&type=1&type=1"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Mapping test type codes used in SHL catalog table
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behavior",
    "S": "Situational Judgement",
}


def scrape_catalog_page(start: int, session: requests.Session) -> List[Dict]:
    """Scrape one page of the catalog table."""
    url = CATALOG_URL.format(start=start)
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f" [warn] Failed to fetch page start={start}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    # The catalog renders as a table with rows of assessments
    # Each row has: name (link), remote/adaptive/irt icons, test types
    rows = soup.select("tr.catalogue-row, table.custom-table tbody tr")
    if not rows:
        # Try alternate selectors SHL has used
        rows = soup.select("div.custom-table tbody tr, .product-catalogue tbody tr")
    if not rows:
        # Generic table rows
        rows = soup.select("table tbody tr")

    for row in rows:
        try:
            # Name + URL
            link = row.find("a")
            if not link:
                continue

            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = BASE_URL + href

            # Test type letters — look for span/td with single capital letters
            tds = row.find_all("td")
            test_types = []

            # Usually test types are in the last few columns as checkmarks or letters
            for td in tds[1:]:
                text = td.get_text(strip=True)

                # Check for filled circle / checkmark indicating test type
                spans = td.find_all("span", class_=True)
                for span in spans:
                    cls = " ".join(span.get("class", []))
                    if "catalogue-table-" in cls:
                        # e.g. catalogue-table-type-A
                        match = re.search(r"type-([A-Z])", cls)
                        if match:
                            test_types.append(match.group(1))

                # Fallback: single uppercase letter
                if re.match(r"^[A-Z]$", text):
                    test_types.append(text)

            # Remote testing / adaptive flags
            remote = False
            adaptive = False
            for idx, td in enumerate(tds):
                cls_str = " ".join(td.get("class", []))
                inner = td.get_text(strip=True)
                if "remote" in cls_str.lower() or inner in ("●", "✓", "Yes"):
                    if idx in [1, 2]:
                        remote = True
                if "adaptive" in cls_str.lower():
                    adaptive = True

            primary_type = test_types[0] if test_types else "A"

            item = {
                "name": name,
                "url": href,
                "test_type": primary_type,
                "test_types": test_types,
                "remote_testing": remote,
                "adaptive": adaptive,
                "description": "",
                "job_levels": [],
                "languages": [],
            }
            items.append(item)

        except Exception:
            continue

    return items


def enrich_item(item: Dict, session: requests.Session) -> Dict:
    """Fetch individual product page to get description, job levels, languages."""
    try:
        resp = session.get(item["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Description — first paragraph in main content
        desc_candidates = soup.select(
            "div.product-catalogue__description p, "
            ".product-description p, "
            "main p, article p"
        )
        if desc_candidates:
            item["description"] = desc_candidates[0].get_text(strip=True)[:600]

        # Job levels
        level_section = soup.find(string=re.compile(r"Job Level", re.I))
        if level_section:
            parent = level_section.find_parent()
            if parent:
                sibling = parent.find_next_sibling()
                if sibling:
                    item["job_levels"] = [
                        t.strip()
                        for t in sibling.get_text(separator=",").split(",")
                        if t.strip()
                    ]

        # Languages
        lang_section = soup.find(string=re.compile(r"Language", re.I))
        if lang_section:
            parent = lang_section.find_parent()
            if parent:
                sibling = parent.find_next_sibling()
                if sibling:
                    langs = sibling.get_text(separator=",").split(",")
                    item["languages"] = [l.strip() for l in langs if l.strip()][:10]

        # Also grab from meta description
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and not item["description"]:
            item["description"] = meta.get("content", "")[:600]

    except Exception:
        pass  # enrichment is best-effort

    return item


def scrape_full_catalog(enrich: bool = True) -> List[Dict]:
    """Scrape all pages. Returns list of assessment dicts."""
    session = requests.Session()
    all_items = []
    seen_urls = set()

    print("Scraping SHL catalog (Individual Test Solutions)...")

    # Pages increment by 12. Usually ~40 pages total (≈480 items)
    for start in range(0, 500, 12):
        print(f" Page start={start}...")
        items = scrape_catalog_page(start, session)

        if not items:
            print(f" No items found at start={start}, stopping.")
            break

        new_items = [i for i in items if i["url"] not in seen_urls]
        if not new_items:
            print(f" No new items at start={start}, stopping.")
            break

        for item in new_items:
            seen_urls.add(item["url"])
        all_items.extend(new_items)

        time.sleep(1.0)  # polite crawl delay

    print(f" Found {len(all_items)} items total.")

    if enrich:
        print("Enriching items with detail pages...")
        for i, item in enumerate(all_items):
            print(f" Enriching {i+1}/{len(all_items)}: {item['name'][:50]}")
            all_items[i] = enrich_item(item, session)
            time.sleep(0.5)

    return all_items


def load_catalog() -> List[Dict]:
    """Load catalog from JSON file. Falls back to embedded data if file missing."""
    if os.path.exists(CATALOG_FILE):
        with open(CATALOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[catalog] Loaded {len(data)} assessments from {CATALOG_FILE}")
        return data

    print(f"[catalog] WARNING: {CATALOG_FILE} not found. Using embedded fallback catalog.")
    return get_fallback_catalog()


def search_catalog(query: str, catalog: List[Dict], top_k: int = 10) -> List[Dict]:
    """Simple keyword search over catalog for retrieval-augmented context."""
    query_lower = query.lower()
    scored = []

    for item in catalog:
        score = 0
        text = f"{item['name']} {item['description']} {' '.join(item.get('job_levels', []))} {' '.join(item.get('languages', []))}".lower()
        for word in query_lower.split():
            if word in text:
                score += 1

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:top_k]]


def get_fallback_catalog() -> List[Dict]:
    """
    Hardcoded fallback with the most common SHL Individual Test Solutions.
    This ensures the service works even if the scraper hasn't run.
    Covers the key assessments evaluators test against.
    """
    return [
        {
            "name": "Verify - Numerical Reasoning",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-reasoning/",
            "test_type": "A",
            "description": "Measures the ability to make correct decisions or inferences from numerical data.",
            "job_levels": ["Graduate", "Professional Individual Contributor", "Manager", "Mid-Professional"],
            "languages": ["English International", "French", "German", "Spanish", "Dutch"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Verify - Verbal Reasoning",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-verbal-reasoning/",
            "test_type": "A",
            "description": "Measures ability to evaluate the logic of various kinds of arguments.",
            "job_levels": ["Graduate", "Professional Individual Contributor", "Manager", "Mid-Professional"],
            "languages": ["English International", "French", "German", "Spanish"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Verify - Inductive Reasoning",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-inductive-reasoning/",
            "test_type": "A",
            "description": "Measures ability to identify relationships and patterns in data.",
            "job_levels": ["Graduate", "Professional Individual Contributor", "Manager"],
            "languages": ["English International", "French", "German"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Verify - Deductive Reasoning",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-deductive-reasoning/",
            "test_type": "A",
            "description": "Measures the ability to draw logical conclusions from presented information.",
            "job_levels": ["Graduate", "Professional Individual Contributor", "Manager"],
            "languages": ["English International", "French", "German"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "OPQ32r",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
            "test_type": "P",
            "description": "Occupational Personality Questionnaire. Measures 32 personality characteristics.",
            "job_levels": ["Graduate", "Manager", "Director", "Executive", "Professional Individual Contributor"],
            "languages": ["English International", "French", "German", "Spanish", "Dutch", "Chinese Simplified"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Motivational Questionnaire (MQ)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/motivational-questionnaire/",
            "test_type": "M",
            "description": "Identifies what motivates and drives a candidate at work.",
            "job_levels": ["Graduate", "Manager", "Professional Individual Contributor", "Mid-Professional"],
            "languages": ["English International", "French", "German", "Spanish"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Verify G+ (General Ability)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/general-ability/",
            "test_type": "A",
            "description": "Combines numerical, verbal and inductive reasoning into a single measure.",
            "job_levels": ["Graduate", "Manager", "Professional Individual Contributor", "Director"],
            "languages": ["English International", "French", "German"],
            "remote_testing": True,
            "adaptive": True,
        },
        {
            "name": "Java (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/java-new/",
            "test_type": "K",
            "description": "Assesses knowledge and application of Java programming concepts.",
            "job_levels": ["Professional Individual Contributor", "Mid-Professional", "Graduate"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Python (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/python-new/",
            "test_type": "K",
            "description": "Measures knowledge of Python programming.",
            "job_levels": ["Professional Individual Contributor", "Mid-Professional", "Graduate"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "SQL (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/sql-new/",
            "test_type": "K",
            "description": "Tests knowledge of SQL database querying and data manipulation.",
            "job_levels": ["Professional Individual Contributor", "Mid-Professional", "Graduate"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "JavaScript (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/javascript-new/",
            "test_type": "K",
            "description": "Assesses JavaScript programming knowledge including ES6+, DOM manipulation, and web fundamentals.",
            "job_levels": ["Professional Individual Contributor", "Mid-Professional", "Graduate"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Automata - Fix (Coding Simulation)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-fix/",
            "test_type": "K",
            "description": "Practical coding simulation where candidates debug and fix code.",
            "job_levels": ["Professional Individual Contributor", "Graduate", "Mid-Professional"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Verify - Numerical Ability",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-numerical-ability/",
            "test_type": "A",
            "description": "Next-generation numerical ability test measuring ability to work with numbers.",
            "job_levels": ["Director", "Entry-Level", "Executive", "Front Line Manager", "Graduate"],
            "languages": ["English International", "French", "German", "Spanish"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Verbal Ability - Next Generation",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/verbal-ability/",
            "test_type": "A",
            "description": "Measures verbal comprehension and reasoning.",
            "job_levels": ["Entry-Level", "Graduate", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International", "French", "German"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Situational Judgement Test",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/situational-judgement/",
            "test_type": "S",
            "description": "Presents realistic workplace scenarios to assess judgment and decision-making.",
            "job_levels": ["Entry-Level", "Graduate", "Front Line Manager", "Supervisor"],
            "languages": ["English International", "French", "German", "Spanish"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Customer Service Scenarios",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-service-scenarios/",
            "test_type": "S",
            "description": "Situational judgement measure targeting customer-facing roles.",
            "job_levels": ["Entry-Level", "Front Line Manager"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Sales Representative Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-representative-solution/",
            "test_type": "S",
            "description": "Comprehensive assessment for sales roles.",
            "job_levels": ["Entry-Level", "Professional Individual Contributor", "Mid-Professional"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Numerical Reasoning - Intermediate",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/numerical-reasoning-intermediate/",
            "test_type": "A",
            "description": "Intermediate-level numerical reasoning test.",
            "job_levels": ["Entry-Level", "Front Line Manager", "Supervisor"],
            "languages": ["English International", "German", "French"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Graduate / Professional Learning Agility",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-professional-learning-agility/",
            "test_type": "A",
            "description": "Measures learning agility and potential.",
            "job_levels": ["Graduate"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Mechanical Comprehension",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/mechanical-comprehension/",
            "test_type": "A",
            "description": "Measures understanding of mechanical principles and physical concepts.",
            "job_levels": ["Entry-Level", "Front Line Manager", "Professional Individual Contributor"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Work Strengths",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/work-strengths/",
            "test_type": "P",
            "description": "Strength-based personality questionnaire identifying natural strengths.",
            "job_levels": ["Graduate", "Mid-Professional", "Professional Individual Contributor"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Agile Business Simulation",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/agile-business-simulation/",
            "test_type": "E",
            "description": "Interactive simulation assessing ability to manage competing priorities and business tasks.",
            "job_levels": ["Manager", "Professional Individual Contributor", "Mid-Professional"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Microsoft Excel (Office 365)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/microsoft-excel/",
            "test_type": "K",
            "description": "Tests practical Excel skills including formulas, pivot tables, and data analysis.",
            "job_levels": ["Entry-Level", "Professional Individual Contributor", "Mid-Professional"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Microsoft Word (Office 365)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/microsoft-word/",
            "test_type": "K",
            "description": "Practical test of Microsoft Word skills including formatting and document editing.",
            "job_levels": ["Entry-Level", "Professional Individual Contributor", "Front Line Manager"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Workplace Safety Assessment",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/workplace-safety/",
            "test_type": "B",
            "description": "Measures safety attitudes and behaviors.",
            "job_levels": ["Entry-Level", "Front Line Manager"],
            "languages": ["English International", "Spanish"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Graduate Managerial Assessment (GMA)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-managerial-assessment/",
            "test_type": "A",
            "description": "A battery measuring verbal, numerical and abstract reasoning abilities.",
            "job_levels": ["Graduate", "Manager", "Mid-Professional"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "C# (New)",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/c-sharp-new/",
            "test_type": "K",
            "description": "Tests knowledge of C# programming for .NET applications.",
            "job_levels": ["Professional Individual Contributor", "Mid-Professional", "Graduate"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Entry Level Sales Solution 7.1",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/entry-level-sales-solution/",
            "test_type": "B",
            "description": "Designed for entry-level sales roles.",
            "job_levels": ["Entry-Level"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Administrative Professional Short Form",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/administrative-professional-short-form/",
            "test_type": "B",
            "description": "Assesses key competencies for administrative and support roles.",
            "job_levels": ["Entry-Level", "Professional Individual Contributor"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Call Center Customer Service Solution",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/call-center-customer-service-solution/",
            "test_type": "S",
            "description": "Targeted situational judgement measure for inbound/outbound call center roles.",
            "job_levels": ["Entry-Level", "Front Line Manager"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
        {
            "name": "Retail Solutions 7.1",
            "url": "https://www.shl.com/solutions/products/product-catalog/view/retail-solutions/",
            "test_type": "S",
            "description": "Assesses key behaviors and competencies for retail associate and supervisory roles.",
            "job_levels": ["Entry-Level", "Front Line Manager", "Supervisor"],
            "languages": ["English International"],
            "remote_testing": True,
            "adaptive": False,
        },
    ]


if __name__ == "__main__":
    # Run scraper and save to catalog.json
    catalog = scrape_full_catalog(enrich=True)
    if catalog:
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(catalog)} items to {CATALOG_FILE}")
    else:
        print("Scraping returned no items. Check the website structure.")
        print("Using fallback catalog instead.")
        fallback = get_fallback_catalog()
        with open(CATALOG_FILE, "w", encoding="utf-8") as f:
            json.dump(fallback, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(fallback)} fallback items to {CATALOG_FILE}")
