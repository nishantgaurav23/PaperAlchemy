"""arXiv API client with rate limiting and retry logic."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import urlencode

import feedparser
import httpx
from src.exceptions import ArxivAPIError
from src.schemas.arxiv import ArxivPaper

logger = logging.getLogger(__name__)


class ArxivClient:
    """Async arXiv API client with rate limiting, retry, and PDF download.

    Respects arXiv API guidelines:
    - Minimum 3s delay between requests
    - Exponential backoff on errors
    - Proper User-Agent header
    """

    def __init__(
        self,
        base_url: str = "https://export.arxiv.org/api/query",
        rate_limit_delay: float = 3.0,
        max_results: int = 100,
        search_category: str = "cs.AI",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        cache_dir: str = "data/arxiv_pdfs",
    ) -> None:
        self.base_url = base_url
        self.rate_limit_delay = max(rate_limit_delay, 3.0)
        self.max_results = max_results
        self.search_category = search_category
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_time: float | None = None
        self.user_agent = "PaperAlchemy/1.0 (Academic Research)"

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect the minimum delay between requests."""
        if self._last_request_time is not None:
            loop = asyncio.get_running_loop()
            elapsed = loop.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)

        self._last_request_time = asyncio.get_running_loop().time()

    # ------------------------------------------------------------------
    # HTTP with retry
    # ------------------------------------------------------------------

    async def _make_request(self, url: str) -> str:
        """GET request with exponential backoff retry.

        Retries on: 429, 503, timeout, connection errors.
        Raises immediately on other non-200 status codes.
        """
        for attempt in range(self.max_retries):
            await self._wait_for_rate_limit()

            try:
                async with httpx.AsyncClient(timeout=self.timeout) as http:
                    response = await http.get(url, headers={"User-Agent": self.user_agent})

                if response.status_code == 200:
                    return response.text

                if response.status_code == 429:
                    wait = self.retry_delay * (2**attempt) * 10
                    logger.warning("Rate limited (429), waiting %.1fs", wait)
                    await asyncio.sleep(wait)
                    continue

                if response.status_code == 503:
                    wait = self.retry_delay * (2**attempt)
                    logger.warning("Service unavailable (503), retry %d/%d", attempt + 1, self.max_retries)
                    await asyncio.sleep(wait)
                    continue

                raise ArxivAPIError(
                    detail=f"HTTP {response.status_code}: {response.text}",
                    context={"url": url, "status_code": response.status_code},
                )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.warning("Request error: %s, retry %d/%d", exc, attempt + 1, self.max_retries)
                await asyncio.sleep(self.retry_delay * (2**attempt))
                continue

        raise ArxivAPIError(detail=f"Failed after {self.max_retries} retries", context={"url": url})

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _build_query(
        self,
        category: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        search_query: str | None = None,
        *,
        skip_default_category: bool = False,
    ) -> str:
        """Build arXiv API search query string.

        Args:
            category: arXiv category filter (e.g. "cs.AI"). Pass empty string to skip.
            from_date: Start date filter (YYYYMMDD).
            to_date: End date filter (YYYYMMDD).
            search_query: Free-text search query.
            skip_default_category: If True, don't fall back to self.search_category
                when category is None. Used for web search where the user wants
                to search across ALL categories.
        """
        parts: list[str] = []

        cat = category if skip_default_category else (category or self.search_category)
        if cat:
            # Support comma-separated categories: "cs.AI,cs.LG" → "(cat:cs.AI OR cat:cs.LG)"
            cats = [c.strip() for c in cat.split(",") if c.strip()]
            if len(cats) == 1:
                parts.append(f"cat:{cats[0]}")
            elif cats:
                cat_query = " OR ".join(f"cat:{c}" for c in cats)
                parts.append(f"({cat_query})")

        if from_date and to_date:
            # arXiv API requires YYYYMMDDHHMM format for submittedDate
            fd = from_date.ljust(12, "0")[:12]
            td = to_date[:8].ljust(12, "0")[:8] + "2359"
            parts.append(f"submittedDate:[{fd} TO {td}]")
        elif from_date:
            fd = from_date.ljust(12, "0")[:12]
            parts.append(f"submittedDate:[{fd} TO 999912312359]")
        elif to_date:
            td = to_date[:8].ljust(12, "0")[:8] + "2359"
            parts.append(f"submittedDate:[000001010000 TO {td}]")

        if search_query:
            # Phrase-match on title and abstract for best relevance.
            # ti: boosts exact title matches (e.g. "attention is all you need"),
            # abs: catches topic searches where keywords appear in abstract.
            escaped = search_query.replace('"', '\\"')
            parts.append(f'(ti:"{escaped}" OR abs:"{escaped}")')

        return " AND ".join(parts) if parts else "cat:cs.AI"

    # ------------------------------------------------------------------
    # Entry parsing
    # ------------------------------------------------------------------

    def _parse_entry(self, entry: dict) -> ArxivPaper:
        """Parse a feedparser entry into an ArxivPaper."""
        arxiv_id = entry.get("id", "").split("/abs/")[-1]
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

        if not arxiv_id:
            raise ValueError("Missing arXiv ID")

        authors = [a.get("name", "") for a in entry.get("authors", [])]
        categories = [t.get("term", "") for t in entry.get("tags", [])]
        published = entry.get("published", "")
        updated = entry.get("updated", "")

        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=entry.get("title", "").replace("\n", " ").strip(),
            authors=authors,
            abstract=entry.get("summary", "").replace("\n", " ").strip(),
            categories=categories,
            published_date=published,
            updated_date=updated if updated != published else None,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        )

    # ------------------------------------------------------------------
    # Fetch papers
    # ------------------------------------------------------------------

    async def fetch_papers(
        self,
        max_results: int | None = None,
        category: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        search_query: str | None = None,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        start: int = 0,
        *,
        skip_default_category: bool = False,
    ) -> list[ArxivPaper]:
        """Fetch papers from the arXiv API.

        Returns:
            List of parsed ArxivPaper objects.
        """
        max_results = max_results or self.max_results
        query = self._build_query(
            category=category,
            from_date=from_date,
            to_date=to_date,
            search_query=search_query,
            skip_default_category=skip_default_category,
        )

        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        url = f"{self.base_url}?{urlencode(params)}"

        logger.info("Fetching papers: %s (max=%d)", query, max_results)
        response_text = await self._make_request(url)

        feed = feedparser.parse(response_text)
        if feed.bozo and feed.bozo_exception:
            logger.warning("Feed parsing warning: %s", feed.bozo_exception)

        papers: list[ArxivPaper] = []
        for entry in feed.entries:
            try:
                papers.append(self._parse_entry(entry))
            except Exception as exc:
                logger.warning("Skipping malformed entry: %s", exc)
                continue

        logger.info("Fetched %d papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # PDF download
    # ------------------------------------------------------------------

    async def download_pdf(
        self,
        arxiv_id: str,
        pdf_url: str,
        *,
        force: bool = False,
    ) -> Path | None:
        """Download a PDF to the local cache directory.

        Args:
            arxiv_id: arXiv paper ID (used for filename).
            pdf_url: Direct PDF download URL.
            force: Re-download even if cached.

        Returns:
            Path to the downloaded PDF, or None on failure.
        """
        safe_id = arxiv_id.replace("/", "_")
        pdf_path = self.cache_dir / f"{safe_id}.pdf"

        if pdf_path.exists() and not force:
            logger.debug("PDF cached: %s", pdf_path)
            return pdf_path

        logger.info("Downloading PDF: %s", arxiv_id)
        await self._wait_for_rate_limit()

        try:
            async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
                response = await http.get(pdf_url, headers={"User-Agent": self.user_agent})

            if response.status_code != 200:
                logger.warning("PDF download failed: HTTP %d", response.status_code)
                return None

            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                logger.warning("Unexpected content type: %s", content_type)
                return None

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > 50 * 1024 * 1024:
                logger.warning("PDF too large: %s bytes", content_length)
                return None

            # Write to temp then rename (atomic)
            temp_path = pdf_path.with_suffix(".tmp")
            temp_path.write_bytes(response.content)

            # Verify PDF magic bytes
            if response.content[:5] != b"%PDF-":
                logger.warning("Invalid PDF file: %s", arxiv_id)
                temp_path.unlink(missing_ok=True)
                return None

            temp_path.rename(pdf_path)
            logger.info("PDF downloaded: %s", pdf_path.name)
            return pdf_path

        except httpx.TimeoutException:
            logger.warning("PDF download timeout: %s", arxiv_id)
            return None
        except Exception as exc:
            logger.error("PDF download error: %s", exc)
            return None
