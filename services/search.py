"""
Academic Paper Search Service.
Fetches papers concurrently from ArXiv and Semantic Scholar,
normalizes, deduplicates, and ranks them by relevance and recency.
"""

import asyncio
import difflib
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import httpx
from pydantic import BaseModel, Field
import xmltodict

logger = logging.getLogger("academic_assistant.search")

ARXIV_API_URL = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API_URL = "https://api.openalex.org/works"
TARGET_ARXIV_CATEGORIES = "(cat:cs.DC OR cat:cs.SE OR cat:cs.AI OR cat:cs.AR)"


class AsyncRateLimiter:
    """
    Async concurrency throttling and rate-limiter enforcing a maximum concurrency
    and a minimum time interval between consecutive outbound requests.
    Uses lazy initialization to safely bind to the running asyncio event loop.
    """
    def __init__(self, min_interval: float = 1.0, concurrency: int = 1):
        self.min_interval = min_interval
        self.concurrency = concurrency
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock: Optional[asyncio.Lock] = None
        self._last_call: float = 0.0

    @property
    def semaphore(self) -> asyncio.Semaphore:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._semaphore_loop = current_loop
        elif current_loop is not None and getattr(self, "_semaphore_loop", None) is not current_loop:
            self._semaphore = asyncio.Semaphore(self.concurrency)
            self._semaphore_loop = current_loop
        return self._semaphore

    @property
    def lock(self) -> asyncio.Lock:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._lock is None:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        elif current_loop is not None and getattr(self, "_lock_loop", None) is not current_loop:
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        return self._lock

    async def __aenter__(self):
        await self.semaphore.acquire()
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            wait_time = self.min_interval - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_call = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class SearchCache:
    """
    Thread-safe in-memory TTL cache for academic search results.
    Normalizes query strings to minimize duplicate API calls and provides
    instant (sub-millisecond) responses for repeated or equivalent queries.
    """
    def __init__(self, ttl: float = 3600.0):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[List["Paper"], Dict[str, Any], float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def normalize_key(query: str) -> str:
        """
        Normalize query string:
        - Convert to lowercase
        - Strip non-alphanumeric punctuation (except hyphen and underscore)
        - Collapse whitespace
        Example: 'Distributed Checkpointing (HPC)!' -> 'distributed checkpointing hpc'
        """
        cleaned = re.sub(r"[^\w\s\-]", " ", query.lower()).strip()
        tokens = cleaned.split()
        return " ".join(tokens)

    def get(self, query: str) -> Optional[Tuple[List["Paper"], Dict[str, Any]]]:
        """
        Retrieve unexpired cached papers and metadata.
        Returns (papers, metadata) with metadata['cache_hit'] = True, or None if expired/not found.
        """
        key = self.normalize_key(query)
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                papers, metadata, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    # Return deep copies to prevent external mutation of cache items
                    copied_papers = [p.model_copy(deep=True) for p in papers]
                    copied_meta = dict(metadata)
                    copied_meta["cache_hit"] = True
                    return copied_papers, copied_meta
                else:
                    # Entry expired, evict
                    del self._cache[key]
        return None

    def set(self, query: str, papers: List["Paper"], metadata: Dict[str, Any]) -> None:
        """Store papers and search metadata with TTL timestamp."""
        key = self.normalize_key(query)
        if not key:
            return
        with self._lock:
            cached_papers = [p.model_copy(deep=True) for p in papers]
            self._cache[key] = (cached_papers, dict(metadata), time.time())

    def clear(self) -> None:
        """Evict all cached entries."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


# Global singletons for cache and rate limiters
search_cache = SearchCache(ttl=3600.0)
semantic_scholar_limiter = AsyncRateLimiter(min_interval=1.0, concurrency=1)
arxiv_limiter = AsyncRateLimiter(min_interval=0.5, concurrency=2)
openalex_limiter = AsyncRateLimiter(min_interval=0.2, concurrency=4)

# Global shared HTTP client with connection pooling
_shared_http_client: Optional[httpx.AsyncClient] = None
_shared_client_loop: Optional[asyncio.AbstractEventLoop] = None


def get_shared_http_client() -> httpx.AsyncClient:
    """
    Retrieve or lazily initialize the shared HTTP client with keepalive connection pooling.
    Guarantees reuse across all search providers, Kroki, and OpenAI-compatible SDKs.
    Automatically detects if the client was bound to a previously closed event loop and refreshes it.
    """
    global _shared_http_client, _shared_client_loop
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    recreate = False
    if _shared_http_client is None or _shared_http_client.is_closed:
        recreate = True
    elif current_loop is not None and (_shared_client_loop is None or _shared_client_loop.is_closed() or _shared_client_loop is not current_loop):
        recreate = True

    if recreate:
        _shared_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            timeout=30.0,
            follow_redirects=True,
        )
        _shared_client_loop = current_loop
    return _shared_http_client


def set_shared_http_client(client: Optional[httpx.AsyncClient]) -> None:
    """Set or inject the shared HTTP client (managed via FastAPI lifespan)."""
    global _shared_http_client
    _shared_http_client = client


async def close_shared_http_client() -> None:
    """Safely close the shared HTTP client connection pool on app shutdown."""
    global _shared_http_client
    if _shared_http_client is not None and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
    _shared_http_client = None



class Paper(BaseModel):
    """Normalized academic paper metadata."""
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    institutions: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    tldr: Optional[str] = None
    pdf_url: Optional[str] = None
    citation_count: Optional[int] = None
    source: str = "Unknown"  # "ArXiv", "Semantic Scholar", "OpenAlex", "ArXiv & OpenAlex", etc.
    primary_category: Optional[str] = None
    url: Optional[str] = None
    code_url: Optional[str] = None


def clean_academic_query(raw_query: str) -> str:
    """
    Extract high-signal technical keywords suitable for academic search engines.
    Strips parenthetical examples, conversational framing, punctuation, and non-essential filler words.
    Example:
      "Recent advances and research gaps in distributed checkpointing and fault tolerance
       for cloud-based HPC scientific simulations (such as reactor or thermal digital twins)"
      -> "distributed checkpointing fault tolerance cloud HPC simulations"
    """
    if not raw_query:
        return "distributed systems"

    # 1. Strip content inside parentheses and brackets (e.g. "(such as reactor or thermal digital twins)")
    cleaned = re.sub(r"\([^)]*\)", " ", raw_query)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)

    # 2. Strip conversational, survey, and filler phrases
    filler_phrases = [
        r"\brecent advances (and|&) (research )?gaps (in|for|on)\b",
        r"\b(recent|current|latest) advances (in|for|on)\b",
        r"\bresearch gaps (in|for|on)\b",
        r"\bstate of the art (in|for|on)\b",
        r"\bliterature review (of|on|for)\b",
        r"\bsurvey (of|on|for)\b",
        r"\b(can you|please) (find|search|show|tell me about|explain)\b",
        r"\bwhat (is|are|were|do you know about)\b",
        r"\btell me (about|everything about)\b",
        r"\bsearch for (papers on|articles about|research on)?\b",
        r"\blooking for (papers on|articles on|research on)?\b",
        r"\bi need (papers on|literature on|research on)?\b",
        r"\bprovide a (summary of|literature review of)?\b",
        r"\bsuch as\b",
        r"\bfor example\b",
        r"\be\.g\.\b",
        r"\bi\.e\.\b",
        r"\bfor my thesis\b",
        r"\bfor a research paper\b",
        r"\bin academic literature\b",
        r"\bfor graduate research\b",
    ]

    for pat in filler_phrases:
        cleaned = re.sub(pat, " ", cleaned, flags=re.IGNORECASE)

    # 3. Remove non-alphanumeric punctuation except hyphens and spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", cleaned)
    tokens = cleaned.split()

    # 4. Stopwords & generic non-discriminative academic filler
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "about", "as", "into", "through", "during", "before", "after",
        "above", "below", "under", "between", "out", "off", "over", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
        "will", "just", "should", "now", "recent", "advances", "advancements", "current",
        "latest", "techniques", "methods", "methodologies", "approaches", "gaps", "gap",
        "research", "study", "studies", "overview", "survey", "thesis", "paper", "papers",
        "article", "articles", "work", "works", "scientific", "based", "using", "used", "via"
    }

    filtered_tokens = []
    for t in tokens:
        # Strip trailing -based (e.g. cloud-based -> cloud)
        clean_t = re.sub(r"-based$", "", t, flags=re.IGNORECASE).strip("-")
        if clean_t and clean_t.lower() not in stopwords and len(clean_t) > 1:
            filtered_tokens.append(clean_t)

    # If filtering leaves tokens, return them joined; otherwise fallback to original tokens
    if filtered_tokens:
        return " ".join(filtered_tokens)
    elif tokens:
        return " ".join(tokens)
    return raw_query.strip()


def normalize_title(title: str) -> str:
    """Normalize paper title for fuzzy similarity comparison."""
    if not title:
        return ""
    cleaned = re.sub(r"[^a-z0-9]", "", title.lower())
    return cleaned


def title_similarity(title_a: str, title_b: str) -> float:
    """Compute string similarity ratio between two titles."""
    norm_a = normalize_title(title_a)
    norm_b = normalize_title(title_b)
    if not norm_a or not norm_b:
        return 0.0
    return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()


async def fetch_arxiv_papers(
    keywords: str,
    client: httpx.AsyncClient,
    max_results: int = 10,
    timeout: float = 12.0,
    author: Optional[str] = None,
    categories: Optional[List[str]] = None
) -> List[Paper]:
    """
    Fetch papers from ArXiv API filtered to CS categories or custom categories.
    Uses essential terms combined with AND to maximize precision.
    Supports author filtering (au:...) and parses author affiliation metadata.
    """
    papers: List[Paper] = []
    tokens = keywords.split() if keywords else []
    
    if not tokens and not author:
        return papers

    # Determine category clause
    if categories and len(categories) > 0:
        clean_cats = [c.strip() for c in categories if c.strip()]
        cat_clause = "(" + " OR ".join(f"cat:{c}" for c in clean_cats) + ")" if clean_cats else TARGET_ARXIV_CATEGORIES
    else:
        cat_clause = TARGET_ARXIV_CATEGORIES

    # Build search terms clause
    if tokens:
        if len(tokens) >= 4:
            arxiv_terms = tokens[:4]
        elif len(tokens) >= 2:
            arxiv_terms = tokens[:len(tokens)]
        else:
            arxiv_terms = tokens
        term_clause = "all:(" + " AND ".join(arxiv_terms) + ")"
    else:
        term_clause = ""

    # Integrate author filter
    query_parts = []
    if term_clause:
        query_parts.append(term_clause)
    if author and author.strip():
        query_parts.append(f'au:"{author.strip()}"')
    if cat_clause:
        query_parts.append(cat_clause)

    search_query = " AND ".join(query_parts)

    effective_max = min(max(max_results, 1), 100)

    async def _execute_arxiv_query(query_str: str) -> List[dict]:
        params = {
            "search_query": query_str,
            "start": 0,
            "max_results": effective_max,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with arxiv_limiter:
            response = await client.get(
                ARXIV_API_URL,
                params=params,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "AxiomResearchAI/1.0 (academic-assistant@gemini.ai)"}
            )
        if response.status_code != 200:
            logger.warning("ArXiv API returned status %d: %s", response.status_code, response.text[:200])
            return []
        parsed = xmltodict.parse(response.text)
        entries_data = parsed.get("feed", {}).get("entry", [])
        if isinstance(entries_data, dict):
            return [entries_data]
        elif isinstance(entries_data, list):
            return entries_data
        return []

    try:
        entries = await _execute_arxiv_query(search_query)

        # Automatic fallback: if 0 results returned and we had >= 3 terms, retry with top 2 core terms
        if not entries and len(tokens) > 2:
            relaxed_terms = tokens[:2]
            relaxed_term_clause = f"all:({' AND '.join(relaxed_terms)})"
            relaxed_parts = [relaxed_term_clause]
            if author and author.strip():
                relaxed_parts.append(f'au:"{author.strip()}"')
            if cat_clause:
                relaxed_parts.append(cat_clause)
            relaxed_query = " AND ".join(relaxed_parts)
            logger.info("ArXiv returned 0 results for %r. Retrying with relaxed terms: %r", search_query, relaxed_query)
            entries = await _execute_arxiv_query(relaxed_query)

        for entry in entries:
            # 1. Title
            raw_title = entry.get("title", "")
            title = re.sub(r"\s+", " ", str(raw_title)).strip()
            if not title:
                continue

            # 2. Authors & Affiliations
            authors_data = entry.get("author", [])
            if isinstance(authors_data, dict):
                authors_data = [authors_data]
            authors = []
            affiliations = []
            for a in authors_data:
                if isinstance(a, dict):
                    if "name" in a:
                        authors.append(a["name"])
                    aff = a.get("arxiv:affiliation")
                    if aff:
                        aff_text = aff.get("#text", "") if isinstance(aff, dict) else str(aff)
                        aff_clean = aff_text.strip()
                        if aff_clean and aff_clean not in affiliations:
                            affiliations.append(aff_clean)
                elif isinstance(a, str):
                    authors.append(a)

            # 3. Year
            published = entry.get("published", "")
            year = None
            if published and len(published) >= 4 and published[:4].isdigit():
                year = int(published[:4])

            # 4. Abstract / Summary
            raw_summary = entry.get("summary", "")
            abstract = re.sub(r"\s+", " ", str(raw_summary)).strip()

            # 5. ID & Links (HTML & PDF)
            entry_id = entry.get("id", "")
            links = entry.get("link", [])
            if isinstance(links, dict):
                links = [links]

            pdf_url = None
            abs_url = entry_id

            for link in links:
                if not isinstance(link, dict):
                    continue
                rel = link.get("@rel")
                link_type = link.get("@type")
                link_title = link.get("@title", "").lower()
                href = link.get("@href")

                if link_title == "pdf" or link_type == "application/pdf":
                    pdf_url = href
                elif rel == "alternate" and href:
                    abs_url = href

            # Fallback derivation for PDF URL
            if not pdf_url and abs_url and "/abs/" in abs_url:
                pdf_url = abs_url.replace("/abs/", "/pdf/")
                if not pdf_url.endswith(".pdf"):
                    pdf_url += ".pdf"

            # Primary Category
            prim_cat_elem = entry.get("arxiv:primary_category")
            prim_cat = None
            if isinstance(prim_cat_elem, dict):
                prim_cat = prim_cat_elem.get("@term")

            # Clean ID
            paper_id = entry_id.split("/abs/")[-1] if "/abs/" in entry_id else entry_id

            papers.append(
                Paper(
                    id=f"arxiv:{paper_id}",
                    title=title,
                    authors=authors,
                    institutions=affiliations,
                    year=year,
                    abstract=abstract,
                    tldr=None,
                    pdf_url=pdf_url,
                    citation_count=None,
                    source="ArXiv",
                    primary_category=prim_cat,
                    url=abs_url,
                )
            )

    except Exception as e:
        logger.error("Error during ArXiv fetch: %s", str(e), exc_info=True)

    return papers


async def fetch_semantic_scholar_papers(
    keywords: str,
    client: httpx.AsyncClient,
    api_key: Optional[str] = None,
    limit: int = 10,
    timeout: float = 12.0,
    **kwargs: Any
) -> List[Paper]:
    """
    Fetch papers from Semantic Scholar API with citation counts, TLDRs, and OpenAccess PDFs.
    If SEMANTIC_SCHOLAR_API_KEY is provided (via parameter or environment variable),
    it is passed in headers as {"x-api-key": SEMANTIC_SCHOLAR_API_KEY}.
    Otherwise, gracefully falls back to direct unauthenticated public requests.
    Encapsulated in a robust try/except block with graceful fallback on HTTP 429 or errors.
    """
    papers: List[Paper] = []
    effective_key = (api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "") or kwargs.get("key") or "").strip()

    headers = {
        "User-Agent": "AxiomResearchAI/1.0 (academic-assistant@gemini.ai)"
    }
    if effective_key:
        headers["x-api-key"] = effective_key

    tokens = keywords.split()
    # Build candidate queries: primary query, then relaxed queries if multi-word
    candidate_queries = []
    if len(tokens) >= 4:
        candidate_queries.append(" ".join(tokens[:4]))
        candidate_queries.append(" ".join(tokens[:3]))
        candidate_queries.append(" ".join(tokens[:2]))
    elif len(tokens) >= 2:
        candidate_queries.append(keywords)
        candidate_queries.append(" ".join(tokens[:2]))
    else:
        candidate_queries.append(keywords)

    seen_queries = set()
    queries_to_try = []
    for q in candidate_queries:
        q_clean = q.strip()
        if q_clean and q_clean not in seen_queries:
            seen_queries.add(q_clean)
            queries_to_try.append(q_clean)

    raw_papers = []
    max_retries = 2
    backoff_delays = [2.0, 4.0]

    try:
        for current_query in queries_to_try:
            params = {
                "query": current_query,
                "fields": "title,authors,year,abstract,tldr,openAccessPdf,citationCount,url",
                "limit": limit,
            }

            response = None
            for attempt in range(max_retries + 1):
                try:
                    async with semantic_scholar_limiter:
                        response = await client.get(
                            SEMANTIC_SCHOLAR_API_URL,
                            params=params,
                            headers=headers,
                            timeout=timeout,
                            follow_redirects=True,
                        )
                except httpx.TimeoutException:
                    logger.warning("Semantic Scholar request timed out for %r.", current_query)
                    break
                except Exception as e:
                    logger.warning("Semantic Scholar network error: %s", str(e))
                    break

                if response is not None and response.status_code == 429:
                    if attempt < max_retries:
                        backoff = backoff_delays[attempt]
                        logger.warning(
                            "Semantic Scholar API rate limit encountered (HTTP 429). Retrying in %.1fs (attempt %d/%d)...",
                            backoff,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    else:
                        logger.warning(
                            "Semantic Scholar API 429 persists after %d retries. Silently falling back to ArXiv results.",
                            max_retries,
                        )
                        return papers

                # Not 429: break the retry loop
                break

            if response is None:
                continue

            if response.status_code != 200:
                logger.warning(
                    "Semantic Scholar API returned non-200 status %d: %s. Continuing with ArXiv results.",
                    response.status_code,
                    response.text[:200]
                )
                return papers

            data = response.json()
            items = data.get("data", [])
            if items:
                raw_papers = items
                break
            else:
                logger.info("Semantic Scholar returned 0 results for %r. Trying next relaxed query...", current_query)

        for item in raw_papers:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            # Authors
            raw_authors = item.get("authors", [])
            authors = [a.get("name", "") for a in raw_authors if isinstance(a, dict) and a.get("name")]

            # Year
            year = item.get("year")
            if not isinstance(year, int):
                year = None

            # Abstract
            abstract = (item.get("abstract") or "").strip()

            # TLDR
            tldr_data = item.get("tldr")
            tldr = None
            if isinstance(tldr_data, dict):
                tldr = tldr_data.get("text")

            # Open Access PDF
            pdf_data = item.get("openAccessPdf")
            pdf_url = None
            if isinstance(pdf_data, dict):
                pdf_url = pdf_data.get("url")

            # Citations
            citation_count = item.get("citationCount")

            # Paper ID & URL
            paper_id = item.get("paperId", "")
            paper_url = item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)

            papers.append(
                Paper(
                    id=f"ss:{paper_id}" if paper_id else f"ss:{abs(hash(title))}",
                    title=title,
                    authors=authors,
                    year=year,
                    abstract=abstract,
                    tldr=tldr,
                    pdf_url=pdf_url,
                    citation_count=citation_count,
                    source="Semantic Scholar",
                    primary_category=None,
                    url=paper_url,
                )
            )

    except httpx.TimeoutException:
        logger.warning("Semantic Scholar request timed out after %.1fs. Gracefully continuing with ArXiv results.", timeout)
    except Exception as e:
        logger.warning("Error during Semantic Scholar fetch: %s. Gracefully continuing with ArXiv results.", str(e))

    return papers


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstruct full-text abstract from OpenAlex abstract_inverted_index.
    The index maps words to lists of 0-based token positions:
    {"In": [0], "this": [1], "paper": [2], ...}
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pos_word_pairs: List[Tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    pos_word_pairs.append((pos, word))
    pos_word_pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pos_word_pairs)


async def fetch_openalex_papers(
    keywords: str,
    client: httpx.AsyncClient,
    limit: int = 10,
    timeout: float = 12.0,
    author: Optional[str] = None,
    institution: Optional[str] = None
) -> List[Paper]:
    """
    Fetch academic papers from OpenAlex works endpoint with has_fulltext:true.
    Reconstructs inverted index abstracts, extracts citations, authors, institutions, and open-access PDFs.
    Scales up to 100 papers per page.
    """
    papers: List[Paper] = []
    clean_kw = keywords.strip() if keywords else ""

    filter_parts = ["has_fulltext:true"]
    if author and author.strip():
        filter_parts.append(f"raw_author_name.search:{author.strip()}")
    if institution and institution.strip():
        filter_parts.append(f"raw_affiliation_strings.search:{institution.strip()}")

    if not clean_kw and len(filter_parts) == 1:
        return papers

    params = {
        "filter": ",".join(filter_parts),
        "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,open_access,cited_by_count",
        "per-page": min(max(limit, 5), 100),
    }
    if clean_kw:
        params["search"] = clean_kw

    headers = {
        "User-Agent": "AxiomResearch/1.0 (mailto:team@swmplabs.org)",
        "Accept": "application/json",
    }

    try:
        async with openalex_limiter:
            response = await client.get(
                OPENALEX_API_URL,
                params=params,
                headers=headers,
                timeout=timeout
            )

        if response.status_code != 200:
            logger.warning("OpenAlex API returned status %d: %s", response.status_code, response.text[:200])
            return papers

        data = response.json()
        items = data.get("results", [])

        for item in items:
            title = (item.get("title") or "").strip()
            if not title:
                continue

            # Authors & Institutions
            authorships = item.get("authorships", [])
            authors = []
            institutions = []
            for a in authorships:
                if isinstance(a, dict):
                    author_obj = a.get("author", {})
                    if isinstance(author_obj, dict) and author_obj.get("display_name"):
                        authors.append(author_obj["display_name"])
                    for inst in a.get("institutions", []):
                        if isinstance(inst, dict) and inst.get("display_name"):
                            iname = inst["display_name"].strip()
                            if iname and iname not in institutions:
                                institutions.append(iname)

            # Year
            year = item.get("publication_year")
            if not isinstance(year, int):
                year = None

            # Abstract reconstructed from inverted index
            abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

            # Open Access PDF / URL
            oa_data = item.get("open_access", {})
            pdf_url = None
            if isinstance(oa_data, dict) and oa_data.get("oa_url"):
                pdf_url = oa_data.get("oa_url")

            # Citations
            citation_count = item.get("cited_by_count")

            # Paper ID & Canonical URL
            raw_id = item.get("id") or ""
            work_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
            doi = item.get("doi")
            canonical_url = doi or raw_id or pdf_url

            papers.append(
                Paper(
                    id=f"openalex:{work_id}" if work_id else f"openalex:{abs(hash(title))}",
                    title=title,
                    authors=authors,
                    institutions=institutions,
                    year=year,
                    abstract=abstract,
                    tldr=None,
                    pdf_url=pdf_url,
                    citation_count=citation_count,
                    source="OpenAlex",
                    primary_category=None,
                    url=canonical_url,
                    code_url=None,
                )
            )

    except httpx.TimeoutException:
        logger.warning("OpenAlex request timed out after %.1fs.", timeout)
    except Exception as e:
        logger.warning("Error during OpenAlex fetch: %s", str(e))

    return papers


async def fetch_entity_papers(
    entity_type: str,
    name: str,
    limit: int = 50,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 15.0
) -> List[Paper]:
    """
    Dedicated endpoint logic to query ALL works by a specific researcher or affiliated with an institution
    without requiring a topic inquiry.
    - If entity_type == 'author':
      Query: https://api.openalex.org/works?filter=raw_author_name.search:{name}&sort=cited_by_count:desc&per_page={limit}
    - If entity_type == 'institution':
      Query: https://api.openalex.org/works?filter=authorships.institutions.display_name.search:{name}&sort=publication_year:desc,cited_by_count:desc&per_page={limit}
      (With automatic fallback to raw_affiliation_strings.search:{name} if rejected by OpenAlex).
    Returns standardized Paper list with authors, institutions, citations, and PDF links.
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return []

    entity_type_clean = entity_type.strip().lower()
    effective_limit = min(max(limit, 1), 100)

    # Check search cache first
    cache_key = f"entity:{entity_type_clean}:{clean_name.lower()}:{effective_limit}"
    cached = search_cache.get(cache_key)
    if cached is not None:
        cached_papers, _ = cached
        logger.info("Search cache HIT for entity %s %r (%d papers)", entity_type_clean, clean_name, len(cached_papers))
        return cached_papers

    http_client = client or get_shared_http_client()
    headers = {
        "User-Agent": "AxiomResearch/1.0 (mailto:team@swmplabs.org)",
        "Accept": "application/json",
    }

    papers: List[Paper] = []

    def _parse_openalex_item(item: dict) -> Optional[Paper]:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        authorships = item.get("authorships", [])
        authors = []
        institutions = []
        for a in authorships:
            if isinstance(a, dict):
                author_obj = a.get("author", {})
                if isinstance(author_obj, dict) and author_obj.get("display_name"):
                    authors.append(author_obj["display_name"])
                for inst in a.get("institutions", []):
                    if isinstance(inst, dict) and inst.get("display_name"):
                        iname = inst["display_name"].strip()
                        if iname and iname not in institutions:
                            institutions.append(iname)
        year = item.get("publication_year")
        if not isinstance(year, int):
            year = None
        abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
        oa_data = item.get("open_access", {})
        pdf_url = oa_data.get("oa_url") if isinstance(oa_data, dict) else None
        citation_count = item.get("cited_by_count")
        raw_id = item.get("id") or ""
        work_id = raw_id.split("/")[-1] if "/" in raw_id else raw_id
        doi = item.get("doi")
        canonical_url = doi or raw_id or pdf_url
        return Paper(
            id=f"openalex:{work_id}" if work_id else f"openalex:{abs(hash(title))}",
            title=title,
            authors=authors,
            institutions=institutions,
            year=year,
            abstract=abstract,
            tldr=None,
            pdf_url=pdf_url,
            citation_count=citation_count,
            source="OpenAlex",
            primary_category=None,
            url=canonical_url,
            code_url=None
        )

    if entity_type_clean == "author":
        params = {
            "filter": f"raw_author_name.search:{clean_name}",
            "sort": "cited_by_count:desc",
            "per-page": effective_limit,
            "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,open_access,cited_by_count"
        }
        try:
            async with openalex_limiter:
                resp = await http_client.get(OPENALEX_API_URL, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    p = _parse_openalex_item(item)
                    if p:
                        papers.append(p)
            else:
                logger.warning("OpenAlex author search returned status %d: %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("Error fetching author works from OpenAlex: %s", str(e))

    elif entity_type_clean == "institution":
        candidate_filters = [
            f"authorships.institutions.display_name.search:{clean_name}",
            f"raw_affiliation_strings.search:{clean_name}"
        ]
        for filt in candidate_filters:
            params = {
                "filter": filt,
                "sort": "publication_year:desc,cited_by_count:desc",
                "per-page": effective_limit,
                "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,open_access,cited_by_count"
            }
            try:
                async with openalex_limiter:
                    resp = await http_client.get(OPENALEX_API_URL, params=params, headers=headers, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    if results:
                        for item in results:
                            p = _parse_openalex_item(item)
                            if p:
                                papers.append(p)
                        break
            except Exception as e:
                logger.warning("OpenAlex query with filter %r failed: %s", filt, str(e))

    if papers:
        await discover_paper_code_urls(papers, http_client)
        search_cache.set(cache_key, papers, {"entity_type": entity_type_clean, "name": clean_name, "count": len(papers)})

    return papers


def extract_github_url_from_text(text: str) -> Optional[str]:
    """Scan text (abstract, summary, comments) for GitHub repository URLs."""
    if not text:
        return None
    match = re.search(r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", text)
    if match:
        url = match.group(0).rstrip(".,)>]\"'")
        return url
    return None


async def discover_paper_code_urls(
    papers: List[Paper],
    client: httpx.AsyncClient,
    timeout: float = 3.0
) -> None:
    """
    Check retrieved papers for associated open-source implementation repositories.
    1. First inspects inline paper abstracts and notes for GitHub repository URLs.
    2. For ArXiv papers without inline code links, queries repository endpoints
       (Papers with Code / Hugging Face / repository lookups) concurrently.
    Populates paper.code_url in place.
    """
    # 1. Immediate regex extraction from abstracts
    for p in papers:
        found_url = extract_github_url_from_text(p.abstract) or extract_github_url_from_text(p.tldr or "")
        if found_url:
            p.code_url = found_url

    # 2. For papers without code_url, perform lightweight asynchronous lookup
    async def check_repo(p: Paper):
        if p.code_url:
            return
        
        # Check if paper has an ArXiv ID
        arxiv_id = None
        if "arxiv:" in p.id.lower():
            arxiv_id = p.id.split("arxiv:")[-1].strip()
        elif p.pdf_url and "arxiv.org" in p.pdf_url:
            match = re.search(r"(\d{4}\.\d{4,5})", p.pdf_url)
            if match:
                arxiv_id = match.group(1)
        elif p.url and "arxiv.org" in p.url:
            match = re.search(r"(\d{4}\.\d{4,5})", p.url)
            if match:
                arxiv_id = match.group(1)

        if not arxiv_id:
            return

        # Query Hugging Face / Papers with Code repository metadata endpoint
        try:
            hf_url = f"https://huggingface.co/api/papers/{arxiv_id}"
            resp = await client.get(hf_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("summary", "")
                found_url = extract_github_url_from_text(summary)
                if found_url:
                    p.code_url = found_url
                    return
        except Exception:
            pass

    lookup_tasks = [check_repo(p) for p in papers if not p.code_url]
    if lookup_tasks:
        await asyncio.gather(*lookup_tasks, return_exceptions=True)


def deduplicate_and_rank(
    arxiv_papers: List[Paper],
    ss_papers: List[Paper],
    openalex_papers: Optional[List[Paper]] = None,
    top_k: int = 6,
    similarity_threshold: float = 0.85,
    author: Optional[str] = None,
    institution: Optional[str] = None
) -> List[Paper]:
    """
    Deduplicate papers across ArXiv, Semantic Scholar, and OpenAlex by normalized title similarity.
    Merges metadata when duplicates occur (preserving open-access PDFs, TLDRs, institutions, highest citation count, and code_url).
    Ranks candidates by:
      0. Author or Institution match (if filters specified)
      1. Presence of implementation code (code_url present gets highest priority)
      2. Recency (2025/2026 prioritized over older papers)
      3. Academic citation count and keyword match quality
    Returns strictly the top_k requested papers (up to 100).
    """
    merged_papers: List[Paper] = []
    openalex_list = openalex_papers or []

    def merge_candidate(candidate: Paper):
        for existing in merged_papers:
            sim = title_similarity(existing.title, candidate.title)
            if sim >= similarity_threshold:
                # Merge sources
                if candidate.source not in existing.source:
                    existing.source = f"{existing.source} & {candidate.source}"

                # Merge citation count (keep highest)
                if candidate.citation_count is not None:
                    if existing.citation_count is None or candidate.citation_count > existing.citation_count:
                        existing.citation_count = candidate.citation_count

                # Merge TLDR
                if not existing.tldr and candidate.tldr:
                    existing.tldr = candidate.tldr

                # Merge PDF URL
                if not existing.pdf_url and candidate.pdf_url:
                    existing.pdf_url = candidate.pdf_url

                # Merge Code URL
                if not existing.code_url and candidate.code_url:
                    existing.code_url = candidate.code_url

                # Merge Year (favor candidate if existing year is missing)
                if not existing.year and candidate.year:
                    existing.year = candidate.year

                # Merge authors if candidate has more complete list
                if len(candidate.authors) > len(existing.authors):
                    existing.authors = candidate.authors

                # Merge institutions
                for inst in candidate.institutions:
                    if inst not in existing.institutions:
                        existing.institutions.append(inst)

                # Merge abstract if candidate has longer text
                if len(candidate.abstract) > len(existing.abstract):
                    existing.abstract = candidate.abstract

                return

        # Not matched: append copy
        merged_papers.append(candidate.model_copy(deep=True))

    for p in arxiv_papers:
        merge_candidate(p)
    for p in ss_papers:
        merge_candidate(p)
    for p in openalex_list:
        merge_candidate(p)

    # Priority Ranking Heuristic:
    author_clean = author.strip().lower() if author else ""
    inst_clean = institution.strip().lower() if institution else ""

    def ranking_score(p: Paper) -> Tuple[int, int, int, int]:
        author_match = 1 if (author_clean and any(author_clean in a.lower() for a in p.authors)) else 0
        inst_match = 1 if (inst_clean and any(inst_clean in i.lower() for i in p.institutions)) else 0
        has_code = 1 if p.code_url else 0
        year_score = p.year if p.year is not None else 0
        citation_score = p.citation_count if p.citation_count is not None else 0
        return (author_match + inst_match, has_code, year_score, citation_score)

    merged_papers.sort(key=ranking_score, reverse=True)
    return merged_papers[:top_k]


async def search_academic_papers(
    inquiry: str,
    semantic_scholar_api_key: Optional[str] = None,
    top_k: int = 6,
    use_cache: bool = True,
    client: Optional[httpx.AsyncClient] = None,
    author: Optional[str] = None,
    institution: Optional[str] = None,
    categories: Optional[List[str]] = None,
    **kwargs: Any
) -> Tuple[List[Paper], Dict[str, Any]]:
    """
    High-level concurrent academic search orchestrator.
    Extracts keywords, launches concurrent requests via asyncio.gather to ArXiv,
    Semantic Scholar, and OpenAlex. Checks and populates thread-safe in-memory TTL search cache.
    Discovers implementation code URLs, prioritizes ranking, and seamlessly compensates quota
    if any single provider fails or hits rate limits.
    Supports scaling up to 100 papers, author filtering, institution filtering, and category selection.
    """
    cache_key = f"{inquiry}|author:{author or ''}|inst:{institution or ''}|cat:{','.join(categories) if categories else ''}|k:{top_k}"
    # 1. Check TTL Cache first
    if use_cache:
        cached_result = search_cache.get(cache_key)
        if cached_result is not None:
            cached_papers, cached_meta = cached_result
            logger.info("Search cache HIT for %r (%d papers returned from cache)", cache_key, len(cached_papers))
            return cached_papers[:top_k], cached_meta

    keywords = clean_academic_query(inquiry) if inquiry else ""
    logger.info("Search cache MISS. Executing academic search for %r (author=%r, inst=%r) -> Keywords: %r", inquiry, author, institution, keywords)

    effective_ss_key = (
        semantic_scholar_api_key
        or kwargs.get("api_key")
        or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        or ""
    ).strip()

    metadata: Dict[str, Any] = {
        "raw_inquiry": inquiry,
        "extracted_keywords": keywords,
        "author_filter": author,
        "institution_filter": institution,
        "categories_filter": categories,
        "arxiv_count": 0,
        "ss_count": 0,
        "openalex_count": 0,
        "arxiv_status": "pending",
        "ss_status": "pending",
        "openalex_status": "pending",
        "errors": [],
        "cache_hit": False,
    }

    target_fetch_count = min(max(top_k, 10), 100)
    http_client = client or get_shared_http_client()

    # Run ArXiv, Semantic Scholar, and OpenAlex in parallel with return_exceptions=True
    arxiv_task = fetch_arxiv_papers(
        keywords,
        http_client,
        max_results=target_fetch_count,
        author=author,
        categories=categories
    )
    ss_task = fetch_semantic_scholar_papers(
        keywords,
        http_client,
        api_key=effective_ss_key or None,
        limit=target_fetch_count
    )
    openalex_task = fetch_openalex_papers(
        keywords,
        http_client,
        limit=target_fetch_count,
        author=author,
        institution=institution
    )

    results = await asyncio.gather(arxiv_task, ss_task, openalex_task, return_exceptions=True)

    # Handle ArXiv result
    arxiv_papers: List[Paper] = []
    if isinstance(results[0], Exception):
        err = f"ArXiv error: {str(results[0])}"
        logger.warning(err)
        metadata["errors"].append(err)
        metadata["arxiv_status"] = "failed"
    else:
        arxiv_papers = results[0]
        metadata["arxiv_count"] = len(arxiv_papers)
        metadata["arxiv_status"] = "success" if arxiv_papers else "empty"

    # Handle Semantic Scholar result
    ss_papers: List[Paper] = []
    if isinstance(results[1], Exception):
        err = f"Semantic Scholar error: {str(results[1])}"
        logger.warning(err)
        metadata["errors"].append(err)
        metadata["ss_status"] = "failed"
    else:
        ss_papers = results[1]
        metadata["ss_count"] = len(ss_papers)
        metadata["ss_status"] = "success" if ss_papers else "empty"

    # Handle OpenAlex result
    openalex_papers: List[Paper] = []
    if isinstance(results[2], Exception):
        err = f"OpenAlex error: {str(results[2])}"
        logger.warning(err)
        metadata["errors"].append(err)
        metadata["openalex_status"] = "failed"
    else:
        openalex_papers = results[2]
        metadata["openalex_count"] = len(openalex_papers)
        metadata["openalex_status"] = "success" if openalex_papers else "empty"

    # Discover code repository URLs for all candidates
    all_candidates = arxiv_papers + ss_papers + openalex_papers
    await discover_paper_code_urls(all_candidates, http_client)

    # Deduplicate, merge metadata, prioritize code & recency, take top k
    final_papers = deduplicate_and_rank(
        arxiv_papers,
        ss_papers,
        openalex_papers,
        top_k=top_k,
        author=author,
        institution=institution
    )
    metadata["final_count"] = len(final_papers)

    # Populate cache
    if use_cache:
        search_cache.set(cache_key, final_papers, metadata)

    return final_papers, metadata
