"""PDF parsing service using Docling."""                                                                                         
                                                                                                                                   
import asyncio                                                                                                                   
import logging                                                                                                                   
import time                                                                                                                      
from concurrent.futures import ThreadPoolExecutor                                                                                
from pathlib import Path                                                                                                         
from typing import Optional                                                                                                      
                                                                                                                                
from src.schemas.arxiv.paper import PDFContent, Section                                                                          
                                                                                                                                
logger = logging.getLogger(__name__)                                                                                             
                                                                                                                                
                                                                                                                                
class PDFParserError(Exception):                                                                                                 
    """Base exception for PDF parser errors."""                                                                                  
    pass                                                                                                                         
                                                                                                                                
                                                                                                                                
class PDFParserTimeoutError(PDFParserError):                                                                                     
    """Parsing timed out."""                                                                                                     
    pass                                                                                                                         
                                                                                                                                
                                                                                                                                
class PDFParserService:                                                                                                          
    """                                                                                                                          
    PDF parsing service using Docling for structured content extraction.                                                         
                                                                                                                                
    Features:                                                                                                                    
    - Extracts sections, tables, figures                                                                                         
    - Configurable page limits                                                                                                   
    - Timeout protection                                                                                                         
    - Graceful error handling                                                                                                    
    """                                                                                                                          
                                                                                                                                
    def __init__(                                                                                                                
        self,                                                                                                                    
        max_pages: int = 50,                                                                                                     
        max_file_size_mb: int = 50,                                                                                              
        timeout: int = 120,                                                                                                      
    ):                                                                                                                           
        """                                                                                                                      
        Initialize PDF parser service.                                                                                           
                                                                                                                                
        Args:                                                                                                                    
            max_pages: Maximum pages to process                                                                                  
            max_file_size_mb: Maximum file size in MB                                                                            
            timeout: Parsing timeout in seconds                                                                                  
        """                                                                                                                      
        self.max_pages = max_pages                                                                                               
        self.max_file_size_mb = max_file_size_mb                                                                                 
        self.timeout = timeout                                                                                                   
        self._executor = ThreadPoolExecutor(max_workers=2)                                                                       
        self._converter = None                                                                                                   
                                                                                                                                
    def _get_converter(self):                                                                                                    
        """Lazy initialization of Docling converter."""                                                                          
        if self._converter is None:                                                                                              
            try:                                                                                                                 
                from docling.document_converter import DocumentConverter                                                         
                self._converter = DocumentConverter()                                                                            
                logger.info("Docling converter initialized")                                                                     
            except ImportError:                                                                                                  
                logger.error("Docling not installed. Run: uv add docling")                                                       
                raise PDFParserError("Docling not installed")                                                                    
        return self._converter                                                                                                   
                                                                                                                                
    def _validate_file(self, pdf_path: Path) -> None:                                                                            
        """                                                                                                                      
        Validate PDF file before parsing.                                                                                        
                                                                                                                                
        Args:                                                                                                                    
            pdf_path: Path to PDF file                                                                                           
                                                                                                                                
        Raises:                                                                                                                  
            PDFParserError: If validation fails                                                                                  
        """                                                                                                                      
        if not pdf_path.exists():                                                                                                
            raise PDFParserError(f"File not found: {pdf_path}")                                                                  
                                                                                                                                
        if not pdf_path.suffix.lower() == ".pdf":                                                                                
            raise PDFParserError(f"Not a PDF file: {pdf_path}")                                                                  
                                                                                                                                
        # Check file size                                                                                                        
        size_mb = pdf_path.stat().st_size / (1024 * 1024)                                                                        
        if size_mb > self.max_file_size_mb:                                                                                      
            raise PDFParserError(f"File too large: {size_mb:.1f}MB > {self.max_file_size_mb}MB")                                 
                                                                                                                                
        # Check PDF magic bytes                                                                                                  
        with open(pdf_path, "rb") as f:                                                                                          
            magic = f.read(5)                                                                                                    
            if magic != b"%PDF-":                                                                                                
                raise PDFParserError(f"Invalid PDF file: {pdf_path}")                                                            
                                                                                                                                
    def _parse_sync(self, pdf_path: Path) -> PDFContent:                                                                         
        """                                                                                                                      
        Synchronous PDF parsing (runs in thread pool).                                                                           
                                                                                                                                
        Args:                                                                                                                    
            pdf_path: Path to PDF file                                                                                           
                                                                                                                                
        Returns:                                                                                                                 
            PDFContent with extracted data                                                                                       
        """                                                                                                                      
        start_time = time.time()                                                                                                 
                                                                                                                                
        try:                                                                                                                     
            converter = self._get_converter()                                                                                    
                                                                                                                                
            # Convert PDF                                                                                                        
            result = converter.convert(str(pdf_path))                                                                            
                                                                                                                                
            # Extract document                                                                                                   
            doc = result.document                                                                                                
                                                                                                                                
            # Extract raw text                                                                                                   
            raw_text = doc.export_to_text() if hasattr(doc, 'export_to_text') else ""                                            
                                                                                                                                
            # Extract sections                                                                                                   
            sections = []                                                                                                        
            if hasattr(doc, 'texts'):                                                                                            
                current_section = None                                                                                           
                                                                                                                                
                for item in doc.texts:                                                                                           
                    # Check if it's a heading                                                                                    
                    if hasattr(item, 'label') and 'heading' in str(item.label).lower():                                          
                        if current_section:                                                                                      
                            sections.append(current_section)                                                                     
                        current_section = Section(                                                                               
                            title=item.text.strip() if hasattr(item, 'text') else "",                                            
                            content="",                                                                                          
                            level=1                                                                                              
                        )                                                                                                        
                    elif current_section:                                                                                        
                        # Add to current section content                                                                         
                        if hasattr(item, 'text'):                                                                                
                            current_section.content += item.text + "\n"                                                          
                    else:                                                                                                        
                        # Create default section for content before first heading                                                
                        current_section = Section(                                                                               
                            title="Introduction",                                                                                
                            content=item.text + "\n" if hasattr(item, 'text') else "",                                           
                            level=1                                                                                              
                        )                                                                                                        
                                                                                                                                
                # Add last section                                                                                               
                if current_section:                                                                                              
                    sections.append(current_section)                                                                             
                                                                                                                                
            # Extract tables (as text representation)                                                                            
            tables = []                                                                                                          
            if hasattr(doc, 'tables'):                                                                                           
                for table in doc.tables:                                                                                         
                    if hasattr(table, 'export_to_text'):                                                                         
                        tables.append(table.export_to_text())                                                                    
                    elif hasattr(table, 'text'):                                                                                 
                        tables.append(table.text)                                                                                
                                                                                                                                
            # Extract figures (captions)                                                                                         
            figures = []                                                                                                         
            if hasattr(doc, 'pictures'):                                                                                         
                for pic in doc.pictures:                                                                                         
                    if hasattr(pic, 'caption') and pic.caption:                                                                  
                        figures.append(pic.caption)                                                                              
                                                                                                                                
            parse_time = time.time() - start_time                                                                                
                                                                                                                                
            return PDFContent(                                                                                                   
                raw_text=raw_text,                                                                                               
                sections=sections,                                                                                               
                tables=tables,                                                                                                   
                figures=figures,                                                                                                 
                parser_used="docling",                                                                                           
                parser_time_seconds=parse_time,                                                                                   
            )                                                                                                                    
                                                                                                                                
        except Exception as e:                                                                                                   
            logger.error(f"Docling parsing failed: {e}")                                                                         
            raise PDFParserError(f"Parsing failed: {e}")                                                                         
                                                                                                                                
    async def parse_pdf(self, pdf_path: Path) -> Optional[PDFContent]:                                                           
        """                                                                                                                      
        Parse PDF file asynchronously with timeout.                                                                              
                                                                                                                                
        Args:                                                                                                                    
            pdf_path: Path to PDF file                                                                                           
                                                                                                                                
        Returns:                                                                                                                 
            PDFContent or None on failure                                                                                        
        """                                                                                                                      
        logger.info(f"Parsing PDF: {pdf_path.name}")                                                                             
                                                                                                                                
        try:                                                                                                                     
            # Validate file first                                                                                                
            self._validate_file(pdf_path)                                                                                        
                                                                                                                                
            # Run parsing in thread pool with timeout                                                                            
            loop = asyncio.get_event_loop()                                                                                      
                                                                                                                                
            try:                                                                                                                 
                content = await asyncio.wait_for(                                                                                
                    loop.run_in_executor(self._executor, self._parse_sync, pdf_path),                                            
                    timeout=self.timeout                                                                                         
                )                                                                                                                
                                                                                                                                
                logger.info(                                                                                                     
                    f"Parsed {pdf_path.name}: "                                                                                  
                    f"{len(content.sections)} sections, "                                                                        
                    f"{len(content.raw_text)} chars, "                                                                           
                    f"{content.parser_time_seconds:.1f}s"                                                                         
                )                                                                                                                
                                                                                                                                
                return content                                                                                                   
                                                                                                                                
            except asyncio.TimeoutError:                                                                                         
                logger.warning(f"Parsing timeout ({self.timeout}s): {pdf_path.name}")                                            
                return None                                                                                                      
                                                                                                                                
        except PDFParserError as e:                                                                                              
            logger.warning(f"PDF validation failed: {e}")                                                                        
            return None                                                                                                          
                                                                                                                                
        except Exception as e:                                                                                                   
            logger.error(f"Unexpected parsing error: {e}")                                                                       
            return None                                                                                                          
                                                                                                                                
    async def parse_multiple(                                                                                                    
        self,                                                                                                                    
        pdf_paths: list[Path],                                                                                                   
        continue_on_error: bool = True                                                                                           
    ) -> dict[str, Optional[PDFContent]]:                                                                                        
        """                                                                                                                      
        Parse multiple PDFs.                                                                                                     
                                                                                                                                
        Args:                                                                                                                    
            pdf_paths: List of PDF paths                                                                                         
            continue_on_error: Continue if one fails                                                                             
                                                                                                                                
        Returns:                                                                                                                 
            Dict mapping filename to PDFContent (or None on failure)                                                             
        """                                                                                                                      
        results = {}                                                                                                             
                                                                                                                                
        for pdf_path in pdf_paths:                                                                                               
            try:                                                                                                                 
                content = await self.parse_pdf(pdf_path)                                                                         
                results[pdf_path.name] = content                                                                                 
            except Exception as e:                                                                                               
                logger.error(f"Failed to parse {pdf_path.name}: {e}")                                                            
                if continue_on_error:                                                                                            
                    results[pdf_path.name] = None                                                                                
                else:                                                                                                            
                    raise                                                                                                        
                                                                                                                                
        # Log summary                                                                                                            
        success_count = sum(1 for v in results.values() if v is not None)                                                        
        logger.info(f"Parsed {success_count}/{len(pdf_paths)} PDFs successfully")                                                
                                                                                                                                
        return results                                                                                                           
                                                                                                                                
    def close(self) -> None:                                                                                                     
        """Cleanup resources."""                                                                                                 
        self._executor.shutdown(wait=False)                                                                                      
        self._converter = None