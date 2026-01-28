"""Pydantic schemas for arXiv papers."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import re

class Section(BaseModel):
    """Represents a section extracted from PDF."""

    title: str = ""
    content: str = ""
    level: int = 1 # Heading level(1=H1, 2=H2, etc)

class PDFContent(BaseModel):
    """Parsed PDF content."""

    raw_text: str = ""
    sections: list[Section] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    figures: list[str] = Field(default_factory=list)
    parser_used: str = "docling"
    parser_time_seconds: float = 0.0

class ArxivPaper(BaseModel):
    """Paper data from arXiv API response"""

    arxiv_id : str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime | str
    updated_date: Optional[datetime | str] = None
    pdf_url: str


    @field_validator("arxiv_id")                                                                 
    @classmethod                                                                                 
    def validate_arxiv_id(cls, v: str) -> str:                                                   
        """Validate and sanitize arXiv ID."""                                                    
        # arXiv IDs: YYMM.NNNNN or archive/YYMMNNN                                               
        pattern = r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+/\d{7}(v\d+)?)$"                              
        if not re.match(pattern, v):                                                             
            raise ValueError(f"Invalid arXiv ID format: {v}")                                    
        return v                                                                                 
                                                                                                
    @field_validator("pdf_url")                                                                  
    @classmethod                                                                                 
    def validate_pdf_url(cls, v: str) -> str:                                                    
        """Validate PDF URL is from arXiv."""                                                    
        if not v.startswith("https://arxiv.org/") and not v.startswith("http://arxiv.org/"):     
            raise ValueError(f"PDF URL must be from arxiv.org: {v}")                             
        return v
    
class PaperCreate(BaseModel):
    """Schema for creating a new paper entry."""

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime | str
    updated_date: Optional[datetime | str] = None
    pdf_url: str

    # Optional PDF content (populated after parsing)
    pdf_content: Optional[str] = None
    sections: Optional[list[dict]] = None
    parsing_status: str = "pending"  # pending, success, failed
    parsing_error: Optional[str] = None

class PaperUpdate(BaseModel):
    """Schema for updating a paper."""
    
    title: Optional[str] = None
    abstract: Optional[str] = None
    pdf_content: Optional[str] = None
    sections: Optional[list[dict]] = None
    parsing_status: Optional[str] = None  # pending, success, failed
    parsing_error: Optional[str] = None

class PaperResponse(BaseModel):
    """Schema for paper response from database"""

    id: int
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    published_date: datetime | str
    updated_date: Optional[datetime | str] = None
    pdf_url: str
    pdf_content: Optional[PDFContent] = None
    sections: Optional[List[Section]] = None
    parsing_status: str
    parsing_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
