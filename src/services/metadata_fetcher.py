import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from dateutil import parser as date_parser

from src.schemas.arxiv.paper import ArxivPaper, PaperCreate, PDFContent
from src.services.arxiv.client import ArxivClient
from src.services.pdf_parser.service import PDFParserService
from src.repositories.paper import PaperRepository

logger = logging.getLogger(__name__)

class MetadataFetcherError(Exception):
    """Base exception for metadata fetcher errors."""
    pass

class MetadataFetcher:
    """
    Orchestrates the complete paper ingestion pipeline

    Pipeline flow:
    1. Fetch papers from arXiv API
    2. Download PDFs (optional)
    3. Parse PDFs with Docling (optional)
    4. Store in PostgresSQL database

    Features:
    - Greacefull error handling (continues on individual failures)
    - Detailed metrics and logging
    - Configurable processing options
    - Transcation management
    """

    def __init__(
            self,
            arxiv_client: ArxivClient,
            pdf_parser: PDFParserService
    ):
        """
        Inititialize metadata fetcher.

        Args:
            arxiv_client: arXiv API client
            pdf_parser: PDF parsing service
        """
        self.arxiv_client = arxiv_client
        self.pdf_parser = pdf_parser

    def _convert_to_paper_create(
            self,
            arxiv_paper: ArxivPaper,
            pdf_content: Optional[PDFContent] = None,
    ) -> PaperCreate:
        """
        Convert ArxivPaper to PaperCreate schema

        Args:
            arxiv_paper: Paper from arXiv API
            pdf_content: Parsed PDF content (optional)

        Returns:
            PaperCreate schema for database
        
        """
        # Parse published date
        if isinstance(arxiv_paper.published_date, str):
            published_date = date_parser.parse(arxiv_paper.published_date)
        else:
            published_date = arxiv_paper.published_date

        # Parse update date
        updated_date = None
        if arxiv_paper.updated_date:
            if isinstance(arxiv_paper.updated_date, str):
                updated_date = date_parser.parse(arxiv_paper.updated_date)
            else:
                updated_date = arxiv_paper.updated_date

        # Determine parsing status
        if pdf_content:
            parsing_status = "success"
            raw_text = pdf_content.raw_text
            sections = [
                {"title": s.title, "content": s.content, "level": s.level}
                for s in pdf_content.sections
            ]
        else:
            parsing_status = "pending"
            raw_text = None
            sections = None

        return PaperCreate(
            arxiv_id=arxiv_paper.arxiv_id,
            title=arxiv_paper.title,
            authors=arxiv_paper.authors,
            abstract=arxiv_paper.abstract,
            categories=arxiv_paper.categories,
            published_date=published_date,
            updated_date=updated_date,
            pdf_url=arxiv_paper.pdf_url,
            pdf_content=raw_text,
            sections=sections,
            parsing_status=parsing_status
        )
    
    async def fetch_and_process_papers(
            self,
            max_results: int = 10,
            category: Optional[str] = None,
            from_date: Optional[str] = None,
            to_date: Optional[str] = None,
            process_pdfs: bool = True,
            store_to_db: bool = True,
            db_session: Optional[Session] = None
    ) -> dict:
        """
        Fetch and process papers from arXiv.

        Args:
            max_results: Maximum papers to fetch
            category: arXiv category filter
            from_date: Start date (YYYYMMDD)
            to_date: End date (YYYYMMDD)
            process_pdfs: Whether to download and parse PDFs
            store_to_db: Whether to store in database
            db_session: Database session (required if store_to_db=True)

        Returns:
            Dict with processing metrics:
            - paper_fetched: Number of fetched from arxiv
            - pdfs_downloaded: Number of PDFs downloaded
            - pdfs_parsed: Number successfully parsed
            - papers_stored: Number stored in database
            - processing_time: Number stored in database
            - errors: List of error messages
        """
        start_time = time.time()

        results = {
            "papers_fetched": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "papers_stored": 0,
            "processing_time": 0.0,
            "errors": [],
            "papers": [],
        }

        # Validate parrameters
        if store_to_db and db_session is None:
            raise MetadataFetcherError("db_session required when store_to_db=True")
        
        try:
            # Step 1: Fetch papers from arXiv
            logger.info(f"Fetching up to {max_results} papers from arXiv...")

            papers = await self.arxiv_client.fetch_papers(
                max_results=max_results,
                category=category,
                from_date=from_date,
                to_date=to_date
            )

            results["papers_fetched"] = len(papers)
            logger.info(f"Fetched {len(papers)} papers from arXiv")

            if not papers:
                results["processing_time"] = time.time() - start_time
                return results
            
            # Step 2-4: Process each paper
            for i, paper in enumerate(papers, 1):
                logger.info(f"Processing paper {i}/{len(papers)}: {paper.arxiv_id}")

                pdf_content = None

                try:
                    # Step 2: Download PDF (optional)
                    if process_pdfs:
                        pdf_path = await self.arxiv_client.download_pdf(paper)

                        if pdf_path:
                            results["pdfs_downloaded"] += 1

                            # Step 3: Parse PDF
                            pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                            if pdf_content:
                                results["pdfs_parsed"] += 1
                                logger.info(
                                    f"Parsed {paper.arxiv_id}: "
                                    f"{len(pdf_content.sections)} sections"
                                )
                            else:
                                logger.warning(f"PDF parsing failed: {paper.arxiv_id}")
                        else:
                            logger.warning(f"PDF download failed: {paper.arxiv_id}")
                    
                    # Step 4: Store in database (optional)
                    if store_to_db and db_session:
                        paper_create = self._convert_to_paper_create(paper, pdf_content)

                        repo = PaperRepository(db_session)
                        stored_paper = repo.upsert(paper_create)

                        if stored_paper:
                            results["papers_stored"] += 1
                            results["papers"].append(stored_paper.arxiv_id)
                            logger.info(f"Stored paper: {paper.arxiv_id} (ID: {stored_paper.id})")

                except Exception as e:
                    error_msg = f"Error processing {paper.arxiv_id}: {str(e)}"                                                   
                    logger.error(error_msg)                                                                                      
                    results["errors"].append(error_msg)                                                                          
                    # Continue processing other papers                                                                           
                    continue

            # Commit transaction if storing to DB                                                                                
            if store_to_db and db_session:                                                                                       
                db_session.commit()                                                                                              
                logger.info(f"Committed {results['papers_stored']} papers to database")

        except Exception as e:                                                                                                   
            error_msg = f"Pipeline error: {str(e)}"                                                                              
            logger.error(error_msg)                                                                                              
            results["errors"].append(error_msg)                                                                                  
                                                                                                                                   
            # Rollback on error                                                                                                  
            if store_to_db and db_session:                                                                                       
                db_session.rollback()                

        results["processing_time"] = time.time() - start_time

        # Log summary
        logger.info(
            f"Pipeline complete: "
            f"fetched={results['papers_fetched']}, "
            f"downloaded={results['pdfs_downloaded']}, "
            f"parsed={results['pdfs_parsed']}, "
            f"stored={results['papers_stored']}, "
            f"errors={len(results['errors'])}, "
            f"time={results['processing_time']:.1f}s"
        )

        return results
    
    async def processing_pending_papers(
            self,
            db_session: Session,
            limit: int = 100,
    ) -> dict:
        """
        Process papers that are pending PDF parsing.

        Args:
            db_session: Database session
            limit: Maximum papers to process

        Returns:
            Dict with proessing metrics
        """
        start_time = time.time()

        results = {
            "papers_processed": 0,
            "pdfs_downloaded": 0,
            "pdfs_parsed": 0,
            "parse_failures": 0,
            "processing_time": 0.0,
            "errors": [],
        }

        try:
            repo = PaperRepository(db_session)
            pending_papers = repo.get_pending_parsing(limit=limit)

            logger.info(f"Found {len(pending_papers)} papers pending parsing")

            for paper in pending_papers:
                results["papers_processed"] += 1

                try:
                    # Download PDF
                    from src.schemas.arxiv.paper import ArxivPaper

                    arxiv_paper = ArxivPaper(
                        arxiv_id=paper.arxiv_id,
                        title=paper.title,
                        authors=paper.authors,
                        abstract=paper.abstract,
                        categories=paper.categories,
                        published_date=paper.published_date,
                        pdf_url=paper.pdf_url
                    )

                    pdf_path = await self.arxiv_client.download_pdf(arxiv_paper)

                    if pdf_path:
                        results["pdfs_downloaded"] += 1

                        # Parse PDF
                        pdf_content = await self.pdf_parser.parse_pdf(pdf_path)

                        if pdf_content:
                            results["pdfs_parsed"] += 1

                            # Update paper with parsed content
                            sections = [
                                {"title": s.title, "content": s.content, "level": s.level}
                                for s in pdf_content.sections
                            ]

                            repo.update_parsing_status(
                                arxiv_id=paper.arxiv_id,
                                status="success",
                                pdf_content=pdf_content.raw_text,
                                sections=sections,
                            )
                        else:
                            results['parse_failures'] += 1
                            repo.update_parsing_status(
                                arxiv_id=paper.arxiv_id,
                                status="failed",
                                error="PDF parsing failed"
                            )

                except Exception as e:
                    error_msg = f"Error processing {paper.arxiv_id}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

                    repo.update_parsing_status(
                        arxiv_id=paper.arxiv_id,
                        status="failed",
                        error=str(e)
                    )
            db_session.commit()
        
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            db_session.rollback()

        results["processing_time"] = time.time() - start_time

        logger.info(
            f"Pending processing complete: "
            f"processed={results['papers_processed']}, "
            f"parsed={results['pdfs_parsed']}, "
            f"failures={results['parse_failures']}"
        )

        return results
    

def make_metadata_fetcher(
        arxiv_client: Optional[ArxivClient] = None,
        pdf_parser: Optional[PDFParserService] = None,
) -> MetadataFetcher:
    """
    Create metadata fetcher with default client or provided dependencies.

    Args:
        arxiv_client: Optional arXiv client (creates default if None)
        pdf_parser: Optional PDF parser (Creates default if None)

    Returns:
        MetadataFetcher instance
    """

    if arxiv_client is None:
        from src.services.arxiv.factory import make_arxiv_client
        arxiv_client = make_arxiv_client()

    if pdf_parser is None:
        from src.services.pdf_parser.factory import make_pdf_parser_service
        pdf_parser = make_pdf_parser_service()

    return MetadataFetcher(
        arxiv_client=arxiv_client,
        pdf_parser=pdf_parser
    )
