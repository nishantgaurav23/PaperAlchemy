"""arXiv API client with rate limiting and retry logic."""                                                                       
                                                                                                                                   
import asyncio                                                                                                                   
import logging                                                                                                                   
import re                                                                                                                        
from pathlib import Path                                                                                                         
from typing import Optional                                                                                                      
from urllib.parse import urlencode                                                                                               
                                                                                                                                
import aiohttp                                                                                                                   
import feedparser                                                                                                                
from aiohttp import ClientTimeout, ClientError                                                                                   
                                                                                                                                
from src.schemas.arxiv.paper import ArxivPaper                                                                                   
                                                                                                                                
logger = logging.getLogger(__name__)                                                                                             
                                                                                                                                
                                                                                                                                
class ArxivClientError(Exception):                                                                                               
    """Base exception for arXiv client errors."""                                                                                
    pass                                                                                                                         
                                                                                                                                
                                                                                                                                
class ArxivRateLimitError(ArxivClientError):                                                                                     
    """Rate limit exceeded."""                                                                                                   
    pass                                                                                                                         
                                                                                                                                
                                                                                                                                
class ArxivAPIError(ArxivClientError):                                                                                           
    """API returned an error."""                                                                                                 
    pass                                                                                                                         
                                                                                                                                
                                                                                                                                
class ArxivClient:                                                                                                               
    """                                                                                                                          
    arXiv API client with rate limiting, retry logic, and PDF download.                                                          
                                                                                                                                
    Respects arXiv API guidelines:                                                                                               
    - 3 second delay between requests                                                                                            
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
    ):                                                                                                                           
        """                                                                                                                      
        Initialize arXiv client.                                                                                                 
                                                                                                                                
        Args:                                                                                                                    
            base_url: arXiv API base URL                                                                                         
            rate_limit_delay: Seconds between requests (min 3.0)                                                                 
            max_results: Default max results per query                                                                           
            search_category: Default arXiv category                                                                              
            timeout: Request timeout in seconds                                                                                  
            max_retries: Max retry attempts on failure                                                                           
            retry_delay: Initial retry delay (exponential backoff)                                                               
            cache_dir: Directory for PDF cache                                                                                   
        """                                                                                                                      
        self.base_url = base_url                                                                                                 
        self.rate_limit_delay = max(rate_limit_delay, 3.0)  # Enforce minimum                                                    
        self.max_results = max_results                                                                                           
        self.search_category = search_category                                                                                   
        self.timeout = timeout                                                                                                   
        self.max_retries = max_retries                                                                                           
        self.retry_delay = retry_delay                                                                                           
        self.cache_dir = Path(cache_dir)                                                                                         
                                                                                                                                
        # Track last request time for rate limiting                                                                              
        self._last_request_time: Optional[float] = None                                                                          
                                                                                                                                
        # Create cache directory                                                                                                 
        self.cache_dir.mkdir(parents=True, exist_ok=True)                                                                        
                                                                                                                                
        # User agent (required by arXiv)                                                                                         
        self.user_agent = "PaperAlchemy/1.0 (Academic Research; mailto:nishantgaurav23@email.com)"                                          
                                                                                                                                
    async def _wait_for_rate_limit(self) -> None:                                                                                
        """Wait if needed to respect rate limit."""                                                                              
        if self._last_request_time is not None:                                                                                  
            elapsed = asyncio.get_event_loop().time() - self._last_request_time                                                  
            if elapsed < self.rate_limit_delay:                                                                                  
                wait_time = self.rate_limit_delay - elapsed                                                                      
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")                                                         
                await asyncio.sleep(wait_time)                                                                                   
                                                                                                                                
        self._last_request_time = asyncio.get_event_loop().time()                                                                
                                                                                                                                
    async def _make_request(                                                                                                     
        self,                                                                                                                    
        url: str,                                                                                                                
        headers: Optional[dict] = None                                                                                           
    ) -> str:                                                                                                                    
        """                                                                                                                      
        Make HTTP request with retry logic.                                                                                      
                                                                                                                                
        Args:                                                                                                                    
            url: Request URL                                                                                                     
            headers: Optional headers                                                                                            
                                                                                                                                
        Returns:                                                                                                                 
            Response text                                                                                                        
                                                                                                                                
        Raises:                                                                                                                  
            ArxivAPIError: On unrecoverable errors                                                                               
            ArxivRateLimitError: On rate limit (429)                                                                             
        """                                                                                                                      
        headers = headers or {}                                                                                                  
        headers["User-Agent"] = self.user_agent                                                                                  
                                                                                                                                
        timeout = ClientTimeout(total=self.timeout)                                                                              
                                                                                                                                
        for attempt in range(self.max_retries):                                                                                  
            await self._wait_for_rate_limit()                                                                                    
                                                                                                                                
            try:                                                                                                                 
                async with aiohttp.ClientSession(timeout=timeout) as session:                                                    
                    async with session.get(url, headers=headers) as response:                                                    
                        if response.status == 200:                                                                               
                            return await response.text()                                                                         
                                                                                                                                
                        elif response.status == 429:                                                                             
                            # Rate limited - wait longer                                                                         
                            wait_time = self.retry_delay * (2 ** attempt) * 10                                                   
                            logger.warning(f"Rate limited (429), waiting {wait_time}s")                                          
                            await asyncio.sleep(wait_time)                                                                       
                            continue                                                                                             
                                                                                                                                
                        elif response.status == 503:                                                                             
                            # Service unavailable - retry with backoff                                                           
                            wait_time = self.retry_delay * (2 ** attempt)                                                        
                            logger.warning(f"Service unavailable (503), retry {attempt + 1}/{self.max_retries}")                 
                            await asyncio.sleep(wait_time)                                                                       
                            continue                                                                                             
                                                                                                                                
                        else:                                                                                                    
                            raise ArxivAPIError(f"HTTP {response.status}: {await response.text()}")                              
                                                                                                                                
            except asyncio.TimeoutError:                                                                                         
                logger.warning(f"Request timeout, retry {attempt + 1}/{self.max_retries}")                                       
                await asyncio.sleep(self.retry_delay * (2 ** attempt))                                                           
                continue                                                                                                         
                                                                                                                                
            except ClientError as e:                                                                                             
                logger.warning(f"Client error: {e}, retry {attempt + 1}/{self.max_retries}")                                     
                await asyncio.sleep(self.retry_delay * (2 ** attempt))                                                           
                continue                                                                                                         
                                                                                                                                
        raise ArxivAPIError(f"Failed after {self.max_retries} retries")                                                          
                                                                                                                                
    def _build_query(                                                                                                            
        self,                                                                                                                    
        category: Optional[str] = None,                                                                                          
        from_date: Optional[str] = None,                                                                                         
        to_date: Optional[str] = None,                                                                                           
        search_query: Optional[str] = None,                                                                                      
    ) -> str:                                                                                                                    
        """                                                                                                                      
        Build arXiv search query string.                                                                                         
                                                                                                                                
        Args:                                                                                                                    
            category: arXiv category (e.g., 'cs.AI')                                                                             
            from_date: Start date YYYYMMDD                                                                                       
            to_date: End date YYYYMMDD                                                                                           
            search_query: Additional search terms                                                                                
                                                                                                                                
        Returns:                                                                                                                 
            Query string for arXiv API                                                                                           
        """                                                                                                                      
        parts = []                                                                                                               
                                                                                                                                
        # Category filter                                                                                                        
        cat = category or self.search_category                                                                                   
        if cat:                                                                                                                  
            parts.append(f"cat:{cat}")                                                                                           
                                                                                                                                
        # Date range filter                                                                                                      
        if from_date and to_date:                                                                                                
            # arXiv date query format: submittedDate:[YYYYMMDD TO YYYYMMDD]                                                      
            parts.append(f"submittedDate:[{from_date} TO {to_date}]")                                                            
        elif from_date:                                                                                                          
            parts.append(f"submittedDate:[{from_date} TO 99991231]")                                                             
        elif to_date:                                                                                                            
            parts.append(f"submittedDate:[00000101 TO {to_date}]")                                                               
                                                                                                                                
        # Additional search terms                                                                                                
        if search_query:                                                                                                         
            parts.append(f"all:{search_query}")                                                                                  
                                                                                                                                
        return " AND ".join(parts) if parts else "cat:cs.AI"                                                                     
                                                                                                                                
    def _parse_entry(self, entry: dict) -> ArxivPaper:                                                                           
        """                                                                                                                      
        Parse feedparser entry to ArxivPaper.                                                                                    
                                                                                                                                
        Args:                                                                                                                    
            entry: Feedparser entry dict                                                                                         
                                                                                                                                
        Returns:                                                                                                                 
            ArxivPaper schema                                                                                                    
        """                                                                                                                      
        # Extract arXiv ID from URL                                                                                              
        arxiv_id = entry.get("id", "").split("/abs/")[-1]                                                                        
        # Remove version suffix for consistency                                                                                  
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id)                                                                                
                                                                                                                                
        # Extract authors                                                                                                        
        authors = [author.get("name", "") for author in entry.get("authors", [])]                                                
                                                                                                                                
        # Extract categories                                                                                                     
        categories = [tag.get("term", "") for tag in entry.get("tags", [])]                                                      
                                                                                                                                
        # Parse dates                                                                                                            
        published = entry.get("published", "")                                                                                   
        updated = entry.get("updated", "")                                                                                       
                                                                                                                                
        # Build PDF URL                                                                                                          
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"                                                                        
                                                                                                                                
        return ArxivPaper(                                                                                                       
            arxiv_id=arxiv_id,                                                                                                   
            title=entry.get("title", "").replace("\n", " ").strip(),                                                             
            authors=authors,                                                                                                     
            abstract=entry.get("summary", "").replace("\n", " ").strip(),                                                        
            categories=categories,                                                                                               
            published_date=published,                                                                                            
            updated_date=updated if updated != published else None,                                                              
            pdf_url=pdf_url,                                                                                                     
        )                                                                                                                        
                                                                                                                                
    async def fetch_papers(                                                                                                      
        self,                                                                                                                    
        max_results: Optional[int] = None,                                                                                       
        category: Optional[str] = None,                                                                                          
        from_date: Optional[str] = None,                                                                                         
        to_date: Optional[str] = None,                                                                                           
        search_query: Optional[str] = None,                                                                                      
        sort_by: str = "submittedDate",                                                                                          
        sort_order: str = "descending",                                                                                          
        start: int = 0,                                                                                                          
    ) -> list[ArxivPaper]:                                                                                                       
        """                                                                                                                      
        Fetch papers from arXiv API.                                                                                             
                                                                                                                                
        Args:                                                                                                                    
            max_results: Maximum papers to fetch                                                                                 
            category: arXiv category filter                                                                                      
            from_date: Start date YYYYMMDD format                                                                                
            to_date: End date YYYYMMDD format                                                                                    
            search_query: Additional search terms                                                                                
            sort_by: Sort field (submittedDate, relevance, lastUpdatedDate)                                                      
            sort_order: Sort order (ascending, descending)                                                                       
            start: Starting index for pagination                                                                                 
                                                                                                                                
        Returns:                                                                                                                 
            List of ArxivPaper objects                                                                                           
                                                                                                                                
        Raises:                                                                                                                  
            ArxivAPIError: On API errors                                                                                         
        """                                                                                                                      
        max_results = max_results or self.max_results                                                                            
                                                                                                                                
        # Build query                                                                                                            
        query = self._build_query(                                                                                               
            category=category,                                                                                                   
            from_date=from_date,                                                                                                 
            to_date=to_date,                                                                                                     
            search_query=search_query,                                                                                           
        )                                                                                                                        
                                                                                                                                
        # Build URL                                                                                                              
        params = {                                                                                                               
            "search_query": query,                                                                                               
            "start": start,                                                                                                      
            "max_results": max_results,                                                                                          
            "sortBy": sort_by,                                                                                                   
            "sortOrder": sort_order,                                                                                             
        }                                                                                                                        
        url = f"{self.base_url}?{urlencode(params)}"                                                                             
                                                                                                                                
        logger.info(f"Fetching papers: {query} (max={max_results})")                                                             
                                                                                                                                
        # Make request                                                                                                           
        response_text = await self._make_request(url)                                                                            
                                                                                                                                
        # Parse response                                                                                                         
        feed = feedparser.parse(response_text)                                                                                   
                                                                                                                                
        if feed.bozo and feed.bozo_exception:                                                                                    
            logger.warning(f"Feed parsing warning: {feed.bozo_exception}")                                                       
                                                                                                                                
        # Convert entries to ArxivPaper objects                                                                                  
        papers = []                                                                                                              
        for entry in feed.entries:                                                                                               
            try:                                                                                                                 
                paper = self._parse_entry(entry)                                                                                 
                papers.append(paper)                                                                                             
            except Exception as e:                                                                                               
                logger.warning(f"Failed to parse entry: {e}")                                                                    
                continue                                                                                                         
                                                                                                                                
        logger.info(f"Fetched {len(papers)} papers")                                                                             
        return papers                                                                                                            
                                                                                                                                
    async def download_pdf(                                                                                                      
        self,                                                                                                                    
        paper: ArxivPaper,                                                                                                       
        force: bool = False                                                                                                      
    ) -> Optional[Path]:                                                                                                         
        """                                                                                                                      
        Download PDF for a paper.                                                                                                
                                                                                                                                
        Args:                                                                                                                    
            paper: ArxivPaper object                                                                                             
            force: Re-download even if cached                                                                                    
                                                                                                                                
        Returns:                                                                                                                 
            Path to downloaded PDF or None on failure                                                                            
        """                                                                                                                      
        # Sanitize filename                                                                                                      
        safe_id = paper.arxiv_id.replace("/", "_")                                                                               
        pdf_path = self.cache_dir / f"{safe_id}.pdf"                                                                             
                                                                                                                                
        # Check cache                                                                                                            
        if pdf_path.exists() and not force:                                                                                      
            logger.debug(f"PDF cached: {pdf_path}")                                                                              
            return pdf_path                                                                                                      
                                                                                                                                
        logger.info(f"Downloading PDF: {paper.arxiv_id}")                                                                        
                                                                                                                                
        await self._wait_for_rate_limit()                                                                                        
                                                                                                                                
        try:                                                                                                                     
            timeout = ClientTimeout(total=60)  # Longer timeout for PDFs                                                         
            headers = {"User-Agent": self.user_agent}                                                                            
                                                                                                                                
            async with aiohttp.ClientSession(timeout=timeout) as session:                                                        
                async with session.get(paper.pdf_url, headers=headers) as response:                                              
                    if response.status != 200:                                                                                   
                        logger.warning(f"PDF download failed: HTTP {response.status}")                                           
                        return None                                                                                              
                                                                                                                                
                    # Check content type                                                                                         
                    content_type = response.headers.get("Content-Type", "")                                                      
                    if "pdf" not in content_type.lower():                                                                        
                        logger.warning(f"Unexpected content type: {content_type}")                                               
                        return None                                                                                              
                                                                                                                                
                    # Check size (max 50MB)                                                                                      
                    content_length = response.headers.get("Content-Length")                                                      
                    if content_length and int(content_length) > 50 * 1024 * 1024:                                                
                        logger.warning(f"PDF too large: {content_length} bytes")                                                 
                        return None                                                                                              
                                                                                                                                
                    # Download to temp file then rename (atomic)                                                                 
                    temp_path = pdf_path.with_suffix(".tmp")                                                                     
                                                                                                                                
                    with open(temp_path, "wb") as f:                                                                             
                        async for chunk in response.content.iter_chunked(8192):                                                  
                            f.write(chunk)                                                                                       
                                                                                                                                
                    # Verify PDF magic bytes                                                                                     
                    with open(temp_path, "rb") as f:                                                                             
                        magic = f.read(5)                                                                                        
                        if magic != b"%PDF-":                                                                                    
                            logger.warning(f"Invalid PDF file: {paper.arxiv_id}")                                                
                            temp_path.unlink()                                                                                   
                            return None                                                                                          
                                                                                                                                
                    # Atomic rename                                                                                              
                    temp_path.rename(pdf_path)                                                                                   
                                                                                                                                
                    logger.info(f"PDF downloaded: {pdf_path.name}")                                                              
                    return pdf_path                                                                                              
                                                                                                                                
        except asyncio.TimeoutError:                                                                                             
            logger.warning(f"PDF download timeout: {paper.arxiv_id}")                                                            
            return None                                                                                                          
                                                                                                                                
        except Exception as e:                                                                                                   
            logger.error(f"PDF download error: {e}")                                                                             
            return None                                                                                                          
                                                                                                                                
    async def fetch_and_download(                                                                                                
        self,                                                                                                                    
        max_results: Optional[int] = None,                                                                                       
        category: Optional[str] = None,                                                                                          
        from_date: Optional[str] = None,                                                                                         
        to_date: Optional[str] = None,                                                                                           
        download_pdfs: bool = True,                                                                                              
    ) -> list[tuple[ArxivPaper, Optional[Path]]]:                                                                                
        """                                                                                                                      
        Fetch papers and optionally download PDFs.                                                                               
                                                                                                                                
        Args:                                                                                                                    
            max_results: Maximum papers to fetch                                                                                 
            category: arXiv category filter                                                                                      
            from_date: Start date YYYYMMDD                                                                                       
            to_date: End date YYYYMMDD                                                                                           
            download_pdfs: Whether to download PDFs                                                                              
                                                                                                                                
        Returns:                                                                                                                 
            List of (ArxivPaper, pdf_path) tuples                                                                                
        """                                                                                                                      
        papers = await self.fetch_papers(                                                                                        
            max_results=max_results,                                                                                             
            category=category,                                                                                                   
            from_date=from_date,                                                                                                 
            to_date=to_date,                                                                                                     
        )                                                                                                                        
                                                                                                                                
        results = []                                                                                                             
        for paper in papers:                                                                                                     
            pdf_path = None                                                                                                      
            if download_pdfs:                                                                                                    
                pdf_path = await self.download_pdf(paper)                                                                        
            results.append((paper, pdf_path))                                                                                    
                                                                                                                                
        return results