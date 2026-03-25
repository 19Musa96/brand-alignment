"""Wikipedia data access for the PoC using the official REST API only."""

import os
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List, Any, Tuple

import requests

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"

# Reuse a single session for connection pooling (HTTP keep-alive)
_session = requests.Session()


class DisambiguationError(Exception):
    """Raised when a Wikipedia lookup resolves to a disambiguation page.

    Attributes:
        entity_name: The original entity name the user entered.
        candidates: List of (title, description) tuples parsed from the page.
    """

    def __init__(self, entity_name: str, candidates: List[Tuple[str, str]]):
        self.entity_name = entity_name
        self.candidates = candidates
        titles = [t for t, _ in candidates[:5]]
        super().__init__(
            f"'{entity_name}' is ambiguous. Did you mean: {', '.join(titles)}?"
        )


def _parse_disambiguation_candidates(title: str) -> List[Tuple[str, str]]:
    """Fetch a disambiguation page and return candidate (title, description) pairs.

    Uses the Action API to get the page HTML, then extracts internal links
    from list items which is the standard disambiguation page format.
    """
    params = {
        "action": "parse",
        "page": title,
        "prop": "links|text",
        "format": "json",
        "redirects": 1,
    }
    resp = _session.get(
        WIKIPEDIA_SEARCH_URL,
        params=params,
        headers=_build_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    parse_data = data.get("parse", {})
    page_text = parse_data.get("text", {}).get("*", "")

    # Extract list items: disambiguation pages use <li> with <a> links
    candidates: List[Tuple[str, str]] = []
    # Match <li> blocks containing at least one internal wiki link
    li_pattern = re.compile(r"<li>(.*?)</li>", re.DOTALL)
    link_pattern = re.compile(r'<a[^>]+href="/wiki/([^"#]+)"[^>]*>([^<]+)</a>')

    for li_match in li_pattern.finditer(page_text):
        li_html = li_match.group(1)
        links = link_pattern.findall(li_html)
        if not links:
            continue
        # First link in the <li> is the candidate article
        raw_title, display_text = links[0]
        candidate_title = urllib.parse.unquote(raw_title).replace("_", " ")
        # Skip meta/help links
        if candidate_title.startswith(("Wikipedia:", "Help:", "Category:", "Template:", "Special:")):
            continue
        # Build description from remaining text (strip HTML tags)
        description = re.sub(r"<[^>]+>", "", li_html).strip()
        # Truncate long descriptions
        if len(description) > 150:
            description = description[:147] + "..."
        candidates.append((candidate_title, description))

    return candidates


def _build_headers() -> Dict[str, str]:
    """Return headers with a required User-Agent for polite Wikipedia access."""
    user_agent = os.getenv("WIKIPEDIA_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            "WIKIPEDIA_USER_AGENT environment variable is required for Wikipedia API requests."
        )
    return {"User-Agent": user_agent}


def _is_likely_brand(entity_name: str) -> bool:
    """
    Detect if an entity name likely refers to a brand/company.
    Uses simple heuristics for PoC.
    """
    name_lower = entity_name.lower()
    
    # Common brand indicators
    brand_keywords = [
        "pizza", "burger", "restaurant", "cafe", "coffee", 
        "motors", "automotive", "corporation", "inc", "ltd", "llc",
        "company", "technologies", "tech", "systems"
    ]
    
    # Check for brand keywords
    if any(kw in name_lower for kw in brand_keywords):
        return True
    
    # Check if capitalized and short (likely brand name)
    words = entity_name.split()
    if len(words) <= 3 and any(w[0].isupper() for w in words if w):
        # Common single-word brands
        common_brands = ["tesla", "apple", "amazon", "google", "microsoft", "meta", "domino", "dominos"]
        if any(brand in name_lower for brand in common_brands):
            return True
    
    return False


def _is_likely_public_figure(entity_name: str) -> bool:
    """
    Detect if an entity name likely refers to a public figure/person.
    Uses simple heuristics for PoC.
    """
    name_lower = entity_name.lower()
    words = entity_name.split()
    
    # Single capitalized surname (e.g., "Putin", "Obama", "Trump")
    if len(words) == 1 and entity_name[0].isupper():
        # Common surnames that are likely public figures
        common_surnames = [
            "putin", "obama", "trump", "biden", "musk", "bezos", "gates",
            "zuckerberg", "jobs", "swift", "beyonce", "rihanna", "kardashian",
            "lebron", "ronaldo", "messi", "oprah", "winfrey"
        ]
        if name_lower in common_surnames:
            return True
        # Any single capitalized word is potentially a surname
        if len(name_lower) > 3:  # Filter out very short words
            return True
    
    # Two-word names (likely person: "Steve Jobs", "Bill Gates")
    if len(words) == 2 and all(w[0].isupper() for w in words if w):
        # Exclude if it looks like a brand
        if not _is_likely_brand(entity_name):
            return True
    
    return False


def _normalize_public_figure_name(name: str) -> List[str]:
    """
    Generate search variants for a public figure name.
    Returns list of search queries to try in order.
    """
    variants = [name]
    
    # Add common full names for well-known surnames
    name_lower = name.lower()
    surname_map = {
        "putin": "Vladimir Putin",
        "obama": "Barack Obama",
        "trump": "Donald Trump",
        "biden": "Joe Biden",
        "musk": "Elon Musk",
        "bezos": "Jeff Bezos",
        "gates": "Bill Gates",
        "zuckerberg": "Mark Zuckerberg",
        "jobs": "Steve Jobs",
        "swift": "Taylor Swift",
    }
    
    if name_lower in surname_map:
        variants.insert(0, surname_map[name_lower])
    
    return variants


def _normalize_brand_name(name: str) -> List[str]:
    """
    Generate search variants for a brand name.
    Returns list of search queries to try in order.
    """
    variants = [name]
    
    # Add company suffixes
    if not any(suffix in name.lower() for suffix in ["inc", "corp", "ltd", "llc", "company"]):
        variants.append(f"{name} company")
        variants.append(f"{name} Inc.")
        variants.append(f"{name} corporation")
    
    # Handle common misspellings/variations
    name_lower = name.lower()
    if "domino" in name_lower:
        variants.insert(0, "Domino's Pizza")
    if name_lower == "apple":
        variants.insert(0, "Apple Inc.")
    if name_lower == "tesla":
        variants.insert(0, "Tesla, Inc.")
    
    return variants


def _detect_entity_type(description: str) -> str:
    """
    Infer entity type from Wikipedia description (first line).
    Returns: 'person', 'organization', or 'other'.
    """
    desc_lower = description.lower()
    
    # EXPLICIT REJECT KEYWORDS (concepts, objects, food)
    reject_keywords = [
        "fruit", "plant", "food", "object", "concept", "building",
        "place", "city", "town", "species", "genus", "album", "song",
        "film", "movie", "book", "novel"
    ]
    if any(kw in desc_lower for kw in reject_keywords):
        return "other"
    
    # PERSON KEYWORDS (human individuals only)
    person_keywords = [
        "born", "politician", "actor", "actress", "singer", "footballer",
        "businessman", "businesswoman", "entrepreneur", "athlete",
        "musician", "director", "writer", "author", "scientist",
        "engineer", "artist", "producer", "rapper", "player",
        "coach", "model", "celebrity", "personality"
    ]
    
    # ORGANIZATION KEYWORDS (companies, brands)
    organization_keywords = [
        "company", "corporation", "inc.", "ltd.", "llc",
        "technology", "manufacturer", "brand", "organization",
        "automotive", "restaurant", "chain", "multinational",
        "conglomerate", "enterprise", "business", "firm",
        "retailer", "developer", "publisher", "operator"
    ]

    person_score = sum(1 for kw in person_keywords if kw in desc_lower)
    organization_score = sum(1 for kw in organization_keywords if kw in desc_lower)

    if organization_score > person_score and organization_score > 0:
        return "organization"
    elif person_score > 0:
        return "person"
    else:
        return "other"


def _search_wikipedia_with_type(
    query: str,
    expected_type: Optional[str] = None,
    is_brand: bool = False,
    is_public_figure: bool = False,
) -> str:
    """
    Search Wikipedia and return best match with STRICT type enforcement.
    expected_type: 'person' or 'organization' - REQUIRED.
    If is_brand=True, tries multiple search variants with company suffixes.
    If is_public_figure=True, tries common full names and prefers person pages.
    """
    # Generate search variants based on type
    if is_brand:
        search_queries = _normalize_brand_name(query)
    elif is_public_figure:
        search_queries = _normalize_public_figure_name(query)
    else:
        search_queries = [query]
    
    for search_query in search_queries:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": search_query,
            "format": "json",
            "srlimit": 10,  # Increased to find better matches
        }
        response = _session.get(
            WIKIPEDIA_SEARCH_URL,
            params=params,
            headers=_build_headers(),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("query", {}).get("search", [])
        
        if not results:
            continue  # Try next variant
        
        # STRICT TYPE ENFORCEMENT
        if expected_type:
            for result in results:
                snippet = result.get("snippet", "")
                detected_type = _detect_entity_type(snippet)
                if detected_type == expected_type:
                    return result["title"]
            # No match found for this search variant, continue to next
            continue
        
        # Legacy path (when no expected_type specified)
        if is_brand:
            for result in results:
                snippet = result.get("snippet", "")
                detected_type = _detect_entity_type(snippet)
                if detected_type == "organization":
                    return result["title"]
        elif is_public_figure:
            for result in results:
                snippet = result.get("snippet", "")
                detected_type = _detect_entity_type(snippet)
                if detected_type == "person":
                    return result["title"]
        
        # Fallback to top result if no type filtering
        return results[0]["title"]
    
    # No results found for any variant
    if expected_type == "person":
        raise ValueError(
            f"Could not resolve '{query}' as a person. "
            "Please provide a full name or more details."
        )
    elif expected_type == "organization":
        raise ValueError(
            f"Could not resolve '{query}' as a brand/organization. "
            "Please be more specific."
        )
    elif is_public_figure:
        raise ValueError(
            f"Could not confidently resolve this public figure: '{query}'. "
            "Please provide more details (e.g., full name)."
        )
    raise ValueError(f"No Wikipedia articles found for '{query}'.")


def _looks_like_logo(url: str) -> bool:
    """Return True if the file name suggests a logo/wordmark/seal."""
    name = url.lower()
    return any(keyword in name for keyword in ["logo", "wordmark", "seal"])


def _is_disallowed_person_type(description: str) -> bool:
    """
    Check if description indicates a non-person entity that should be rejected.
    Used for direct page validation when expected_type is 'person'.
    """
    desc_lower = description.lower()
    disallowed_keywords = [
        "band", "group", "fictional", "character", "myth",
        "mythology", "comic", "novel", "tv series", "film",
        "album", "song title"
    ]
    return any(kw in desc_lower for kw in disallowed_keywords)


def _add_image_candidate(candidates: List[Dict[str, Any]], url: Optional[str], width: Optional[int], height: Optional[int]) -> None:
    if not url:
        return
    key = (url, width, height)
    if any((c.get("url"), c.get("width"), c.get("height")) == key for c in candidates):
        return
    candidates.append({"url": url, "width": width, "height": height})


def _pick_best_image(candidates: List[Dict[str, Any]], is_brand: bool, is_public_figure: bool) -> Optional[str]:
    if not candidates:
        return None

    def area(item: Dict[str, Any]) -> int:
        w, h = item.get("width"), item.get("height")
        if not w or not h:
            return 0
        return int(w) * int(h)

    if is_brand:
        logo_first = [c for c in candidates if _looks_like_logo(c.get("url", ""))]
        if logo_first:
            return max(logo_first, key=area).get("url")
        brand_photos = [c for c in candidates if c.get("width") and c.get("height") and int(c["width"]) >= int(c["height"])]
        if brand_photos:
            return max(brand_photos, key=area).get("url")
    if is_public_figure:
        portraits = [c for c in candidates if c.get("height") and c.get("width") and int(c["height"]) >= int(c["width"])]
        if portraits:
            return max(portraits, key=area).get("url")

    return max(candidates, key=area).get("url") or candidates[0].get("url")


def _fetch_best_image(title: str, is_brand: bool, is_public_figure: bool, summary_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Collect image candidates from REST summary and PageImages; prefer logos for brands and portraits for people."""
    candidates: List[Dict[str, Any]] = []

    if summary_data:
        thumb = summary_data.get("thumbnail") or {}
        _add_image_candidate(candidates, thumb.get("source"), thumb.get("width"), thumb.get("height"))
        original = summary_data.get("originalimage") or {}
        _add_image_candidate(candidates, original.get("source"), original.get("width"), original.get("height"))

    # Use Action API pageimages for more options (allows logo/portrait selection)
    params = {
        "action": "query",
        "titles": title,
        "prop": "pageimages",
        "piprop": "original|thumbnail",
        "pithumbsize": 800,
        "format": "json",
        "redirects": 1,
        "pilicense": "any",
    }
    try:
        resp = _session.get(
            WIKIPEDIA_SEARCH_URL,
            params=params,
            headers=_build_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            thumb = page.get("thumbnail") or {}
            _add_image_candidate(candidates, thumb.get("source"), thumb.get("width"), thumb.get("height"))
            original = page.get("original") or {}
            _add_image_candidate(candidates, original.get("source"), original.get("width"), original.get("height"))
    except Exception:
        # Image retrieval is best-effort; ignore failures
        pass

    return _pick_best_image(candidates, is_brand, is_public_figure)


def _fetch_article_sections(title: str) -> str:
    """
    Fetch clean, plain-text content (lead + body) limited in size.

    Uses the Action API extracts with plaintext. If empty, falls back to REST summary.
    """
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "exintro": False,  # allow body content
    }
    response = _session.get(
        WIKIPEDIA_SEARCH_URL,
        params=params,
        headers=_build_headers(),
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"Could not fetch article for '{title}'.")
    page_id = list(pages.keys())[0]
    page = pages[page_id]

    extract = page.get("extract", "")
    if not extract:
        # Fallback to REST summary
        encoded_title = urllib.parse.quote(title)
        url = WIKIPEDIA_SUMMARY_URL.format(title=encoded_title)
        resp_summary = _session.get(url, headers=_build_headers(), timeout=10)
        if resp_summary.status_code == 200:
            extract = resp_summary.json().get("extract", "")
    if not extract:
        raise ValueError(f"Article for '{title}' has no content.")

    # Truncate to reasonable size for embeddings
    max_length = 2500
    if len(extract) > max_length:
        extract = extract[:max_length]

    return extract


def get_entity_text(entity_name: str, expected_type: Optional[str] = None) -> Dict[str, Optional[str]]:
    """
    Fetch Wikipedia content for an entity with STRICT type enforcement.

    Args:
        entity_name: The entity to search for
        expected_type: REQUIRED - either 'person' or 'organization'
                      'person' → ONLY human individuals
                      'organization' → ONLY companies/brands

    Returns a dict with:
    - title: resolved article title
    - text: plain-text content (lead + body truncated)
    - image_url: best-effort logo (brands) or portrait (people)
    """

    if not entity_name or not entity_name.strip():
        raise ValueError("entity_name must be a non-empty string.")

    entity_name = entity_name.strip()
    
    # Determine search strategy based on expected type
    is_brand = expected_type == "organization" or _is_likely_brand(entity_name)
    is_public_figure = expected_type == "person" or (_is_likely_public_figure(entity_name) if not is_brand else False)
    
    title = entity_name
    summary_data = None

    # Try direct lookup first
    encoded_title = urllib.parse.quote(title)
    url = WIKIPEDIA_SUMMARY_URL.format(title=encoded_title)
    response = _session.get(url, headers=_build_headers(), timeout=10)

    direct_lookup_failed = False
    if response.status_code != 200:
        direct_lookup_failed = True
    else:
        data = response.json()
        page_type = data.get("type")
        if page_type == "disambiguation":
            candidates = _parse_disambiguation_candidates(title)
            if candidates:
                raise DisambiguationError(entity_name, candidates)
            # If parsing returned nothing, fall back to search
            direct_lookup_failed = True
        elif page_type == "missing":
            direct_lookup_failed = True
        else:
            # Validate that direct result matches expected_type
            if expected_type:
                extract = data.get("extract", "")
                if expected_type == "person":
                    # For persons: accept by default, reject only disallowed
                    if _is_disallowed_person_type(extract):
                        direct_lookup_failed = True
                    else:
                        summary_data = data
                elif expected_type == "organization":
                    # For organizations: strict keyword validation required
                    detected = _detect_entity_type(extract)
                    if detected != "organization":
                        direct_lookup_failed = True
                    else:
                        summary_data = data
                else:
                    summary_data = data
            else:
                summary_data = data

    # If direct lookup fails, search with STRICT type enforcement
    if direct_lookup_failed:
        title = _search_wikipedia_with_type(
            entity_name,
            expected_type=expected_type,
            is_brand=is_brand,
            is_public_figure=is_public_figure,
        )
        # Fetch summary for resolved title
        encoded_title = urllib.parse.quote(title)
        url = WIKIPEDIA_SUMMARY_URL.format(title=encoded_title)
        resp_summary = _session.get(url, headers=_build_headers(), timeout=10)
        if resp_summary.status_code == 200:
            summary_data = resp_summary.json()

    # Fetch article text and image concurrently
    resolved_title = summary_data.get("title", title) if summary_data else title
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_extract = executor.submit(_fetch_article_sections, title)
        future_image = executor.submit(
            _fetch_best_image, resolved_title, is_brand, is_public_figure, summary_data
        )
        extract = future_extract.result()
        image_url = future_image.result()

    return {
        "title": resolved_title,
        "text": extract,
        "image_url": image_url,
    }


def get_entity_text_by_title(
    title: str, expected_type: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """Fetch Wikipedia content for a *resolved* article title.

    Unlike ``get_entity_text`` this skips disambiguation/search logic and
    goes straight to fetching the article.  Used after the user picks a
    candidate from the disambiguation picker.
    """
    if not title or not title.strip():
        raise ValueError("title must be a non-empty string.")

    title = title.strip()
    is_brand = expected_type == "organization"
    is_public_figure = expected_type == "person"

    encoded_title = urllib.parse.quote(title)
    url = WIKIPEDIA_SUMMARY_URL.format(title=encoded_title)
    resp = _session.get(url, headers=_build_headers(), timeout=10)
    summary_data = resp.json() if resp.status_code == 200 else None

    # Fetch article text and image concurrently
    resolved_title = summary_data.get("title", title) if summary_data else title
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_extract = executor.submit(_fetch_article_sections, title)
        future_image = executor.submit(
            _fetch_best_image, resolved_title, is_brand, is_public_figure, summary_data
        )
        extract = future_extract.result()
        image_url = future_image.result()

    return {
        "title": resolved_title,
        "text": extract,
        "image_url": image_url,
    }
