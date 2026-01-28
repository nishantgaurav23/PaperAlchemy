"""Paper repository for database operations."""                                                  
                                                                                                   
from datetime import datetime                                                                    
from typing import Optional                                                                      
                                                                                                
from sqlalchemy import select, and_, or_                                                         
from sqlalchemy.orm import Session                                                               
from sqlalchemy.dialects.postgresql import insert                                                
from sqlalchemy.exc import IntegrityError, SQLAlchemyError                                    
                                                                                                
from src.models.paper import Paper                                                               
from src.schemas.arxiv.paper import PaperCreate, PaperUpdate                                     
                                                                                                
                                                                                                
class PaperRepository:                                                                           
    """Repository for Paper CRUD operations."""                                                  
                                                                                                
    def __init__(self, session: Session):                                                        
        """                                                                                      
        Initialize repository with database session.                                             
                                                                                                
        Args:                                                                                    
            session: SQLAlchemy session                                                          
        """                                                                                      
        self.session = session                                                                   
                                                                                                
    def create(self, paper_data: PaperCreate) -> Paper:                                          
        """                                                                                      
        Create a new paper.                                                                      
                                                                                                
        Args:                                                                                    
            paper_data: Paper creation schema                                                    
                                                                                                
        Returns:                                                                                 
            Created Paper object                                                                 
                                                                                                
        Raises:                                                                                  
            IntegrityError: If arxiv_id already exists                                           
        """                                                                                      
        paper = Paper(                                                                           
            arxiv_id=paper_data.arxiv_id,                                                        
            title=paper_data.title,                                                              
            authors=paper_data.authors,                                                          
            abstract=paper_data.abstract,                                                        
            categories=paper_data.categories,                                                    
            published_date=paper_data.published_date,                                            
            updated_date=paper_data.updated_date,                                                
            pdf_url=paper_data.pdf_url,                                                          
            pdf_content=paper_data.pdf_content,                                                  
            sections=paper_data.sections,                                                        
            parsing_status=paper_data.parsing_status,                                            
            parsing_error=paper_data.parsing_error,                                              
        )                                                                                        
        self.session.add(paper)                                                                  
        self.session.flush()  # Get ID without committing                                        
        return paper                                                                             
                                                                                                
    def upsert(self, paper_data: PaperCreate) -> Paper:                                          
        """                                                                                      
        Insert or update a paper (upsert).                                                       
                                                                                                
        Args:                                                                                    
            paper_data: Paper creation schema                                                    
                                                                                                
        Returns:                                                                                 
            Created or updated Paper object                                                      
        """                                                                                      
        stmt = insert(Paper).values(                                                             
            arxiv_id=paper_data.arxiv_id,                                                        
            title=paper_data.title,                                                              
            authors=paper_data.authors,                                                          
            abstract=paper_data.abstract,                                                        
            categories=paper_data.categories,                                                    
            published_date=paper_data.published_date,                                            
            updated_date=paper_data.updated_date,                                                
            pdf_url=paper_data.pdf_url,                                                          
            pdf_content=paper_data.pdf_content,                                                  
            sections=paper_data.sections,                                                        
            parsing_status=paper_data.parsing_status,                                            
            parsing_error=paper_data.parsing_error,                                              
        )                                                                                        
                                                                                                
        # On conflict, update these fields                                                       
        stmt = stmt.on_conflict_do_update(                                                       
            index_elements=["arxiv_id"],                                                         
            set_={                                                                               
                "title": stmt.excluded.title,                                                    
                "authors": stmt.excluded.authors,                                                
                "abstract": stmt.excluded.abstract,                                              
                "categories": stmt.excluded.categories,                                          
                "updated_date": stmt.excluded.updated_date,                                      
                "pdf_content": stmt.excluded.pdf_content,                                        
                "sections": stmt.excluded.sections,                                              
                "parsing_status": stmt.excluded.parsing_status,                                  
                "parsing_error": stmt.excluded.parsing_error,                                    
                "updated_at": datetime.utcnow(),                                                 
            }                                                                                    
        ).returning(Paper)                                                                       
                                                                                                
        result = self.session.execute(stmt)                                                      
        self.session.flush()                                                                     
        return result.scalar_one()                                                               
                                                                                                
    def get_by_id(self, paper_id: int) -> Optional[Paper]:                                       
        """                                                                                      
        Get paper by database ID.                                                                
                                                                                                
        Args:                                                                                    
            paper_id: Paper database ID                                                          
                                                                                                
        Returns:                                                                                 
            Paper object or None                                                                 
        """                                                                                      
        stmt = select(Paper).where(Paper.id == paper_id)                                         
        return self.session.execute(stmt).scalar_one_or_none()                                   
                                                                                                
    def get_by_arxiv_id(self, arxiv_id: str) -> Optional[Paper]:                                 
        """                                                                                      
        Get paper by arXiv ID.                                                                   
                                                                                                
        Args:                                                                                    
            arxiv_id: arXiv identifier                                                           
                                                                                                
        Returns:                                                                                 
            Paper object or None                                                                 
        """                                                                                      
        stmt = select(Paper).where(Paper.arxiv_id == arxiv_id)                                   
        return self.session.execute(stmt).scalar_one_or_none()                                   
                                                                                                
    def exists(self, arxiv_id: str) -> bool:                                                     
        """                                                                                      
        Check if paper exists.                                                                   
                                                                                                
        Args:                                                                                    
            arxiv_id: arXiv identifier                                                           
                                                                                                
        Returns:                                                                                 
            True if exists, False otherwise                                                      
        """                                                                                      
        stmt = select(Paper.id).where(Paper.arxiv_id == arxiv_id)                                
        return self.session.execute(stmt).scalar_one_or_none() is not None                       
                                                                                                
    def update(self, arxiv_id: str, paper_data: PaperUpdate) -> Optional[Paper]:                 
        """                                                                                      
        Update a paper.                                                                          
                                                                                                
        Args:                                                                                    
            arxiv_id: arXiv identifier                                                           
            paper_data: Fields to update                                                         
                                                                                                
        Returns:                                                                                 
            Updated Paper object or None if not found                                            
        """                                                                                      
        paper = self.get_by_arxiv_id(arxiv_id)                                                   
        if not paper:                                                                            
            return None                                                                          
                                                                                                
        update_data = paper_data.model_dump(exclude_unset=True)                                  
        for field, value in update_data.items():                                                 
            setattr(paper, field, value)                                                         
                                                                                                
        self.session.flush()                                                                     
        return paper                                                                             
                                                                                                
    def update_parsing_status(                                                                   
        self,                                                                                    
        arxiv_id: str,                                                                           
        status: str,                                                                             
        pdf_content: Optional[str] = None,                                                       
        sections: Optional[list[dict]] = None,                                                   
        error: Optional[str] = None                                                              
    ) -> Optional[Paper]:                                                                        
        """                                                                                      
        Update paper's parsing status and content.                                               
                                                                                                
        Args:                                                                                    
            arxiv_id: arXiv identifier                                                           
            status: New parsing status (pending/success/failed)                                  
            pdf_content: Parsed text content                                                     
            sections: Parsed sections                                                            
            error: Error message if failed                                                       
                                                                                                
        Returns:                                                                                 
            Updated Paper object or None                                                         
        """                                                                                      
        paper = self.get_by_arxiv_id(arxiv_id)                                                   
        if not paper:                                                                            
            return None                                                                          
                                                                                                
        paper.parsing_status = status                                                            
        if pdf_content is not None:                                                              
            paper.pdf_content = pdf_content                                                      
        if sections is not None:                                                                 
            paper.sections = sections                                                            
        if error is not None:                                                                    
            paper.parsing_error = error                                                          
                                                                                                
        self.session.flush()                                                                     
        return paper                                                                             
                                                                                                
    def get_pending_parsing(self, limit: int = 100) -> list[Paper]:                              
        """                                                                                      
        Get papers pending PDF parsing.                                                          
                                                                                                
        Args:                                                                                    
            limit: Maximum papers to return                                                      
                                                                                                
        Returns:                                                                                 
            List of Paper objects                                                                
        """                                                                                      
        stmt = (                                                                                 
            select(Paper)                                                                        
            .where(Paper.parsing_status == "pending")                                            
            .order_by(Paper.created_at.asc())                                                    
            .limit(limit)                                                                        
        )                                                                                        
        return list(self.session.execute(stmt).scalars().all())                                  
                                                                                                
    def get_by_date_range(                                                                       
        self,                                                                                    
        from_date: datetime,                                                                     
        to_date: datetime,                                                                       
        limit: int = 100,                                                                        
        offset: int = 0                                                                          
    ) -> list[Paper]:                                                                            
        """                                                                                      
        Get papers within date range.                                                            
                                                                                                
        Args:                                                                                    
            from_date: Start date (inclusive)                                                    
            to_date: End date (inclusive)                                                        
            limit: Maximum papers to return                                                      
            offset: Number of papers to skip                                                     
                                                                                                
        Returns:                                                                                 
            List of Paper objects                                                                
        """                                                                                      
        stmt = (                                                                                 
            select(Paper)                                                                        
            .where(                                                                              
                and_(                                                                            
                    Paper.published_date >= from_date,                                           
                    Paper.published_date <= to_date                                              
                )                                                                                
            )                                                                                    
            .order_by(Paper.published_date.desc())                                               
            .offset(offset)                                                                      
            .limit(limit)                                                                        
        )                                                                                        
        return list(self.session.execute(stmt).scalars().all())                                  
                                                                                                
    def get_by_category(                                                                         
        self,                                                                                    
        category: str,                                                                           
        limit: int = 100,                                                                        
        offset: int = 0                                                                          
    ) -> list[Paper]:                                                                            
        """                                                                                      
        Get papers by category.                                                                  
                                                                                                
        Args:                                                                                    
            category: arXiv category (e.g., 'cs.AI')                                             
            limit: Maximum papers to return                                                      
            offset: Number of papers to skip                                                     
                                                                                                
        Returns:                                                                                 
            List of Paper objects                                                                
        """                                                                                      
        # PostgreSQL JSON array contains                                                         
        stmt = (                                                                                 
            select(Paper)                                                                        
            .where(Paper.categories.contains([category]))                                        
            .order_by(Paper.published_date.desc())                                               
            .offset(offset)                                                                      
            .limit(limit)                                                                        
        )                                                                                        
        return list(self.session.execute(stmt).scalars().all())                                  
                                                                                                
    def search(                                                                                  
        self,                                                                                    
        query: Optional[str] = None,                                                             
        category: Optional[str] = None,                                                          
        from_date: Optional[datetime] = None,                                                    
        to_date: Optional[datetime] = None,                                                      
        parsing_status: Optional[str] = None,                                                    
        limit: int = 100,                                                                        
        offset: int = 0                                                                          
    ) -> list[Paper]:                                                                            
        """                                                                                      
        Search papers with multiple filters.                                                     
                                                                                                
        Args:                                                                                    
            query: Text search in title/abstract                                                 
            category: Filter by category                                                         
            from_date: Filter by date (start)                                                    
            to_date: Filter by date (end)                                                        
            parsing_status: Filter by parsing status                                             
            limit: Maximum papers to return                                                      
            offset: Number of papers to skip                                                     
                                                                                                
        Returns:                                                                                 
            List of Paper objects                                                                
        """                                                                                      
        conditions = []                                                                          
                                                                                                
        if query:                                                                                
            # Simple text search (use OpenSearch for production)                                 
            search_pattern = f"%{query}%"                                                        
            conditions.append(                                                                   
                or_(                                                                             
                    Paper.title.ilike(search_pattern),                                           
                    Paper.abstract.ilike(search_pattern)                                         
                )                                                                                
            )                                                                                    
                                                                                                
        if category:                                                                             
            conditions.append(Paper.categories.contains([category]))                             
                                                                                                
        if from_date:                                                                            
            conditions.append(Paper.published_date >= from_date)                                 
                                                                                                
        if to_date:                                                                              
            conditions.append(Paper.published_date <= to_date)                                   
                                                                                                
        if parsing_status:                                                                       
            conditions.append(Paper.parsing_status == parsing_status)                            
                                                                                                
        stmt = (                                                                                 
            select(Paper)                                                                        
            .where(and_(*conditions) if conditions else True)                                    
            .order_by(Paper.published_date.desc())                                               
            .offset(offset)                                                                      
            .limit(limit)                                                                        
        )                                                                                        
        return list(self.session.execute(stmt).scalars().all())                                  
                                                                                                
    def count(self, parsing_status: Optional[str] = None) -> int:                                
        """                                                                                      
        Count papers, optionally filtered by status.                                             
                                                                                                
        Args:                                                                                    
            parsing_status: Filter by parsing status                                             
                                                                                                
        Returns:                                                                                 
            Number of papers                                                                     
        """                                                                                      
        from sqlalchemy import func                                                              
                                                                                                
        stmt = select(func.count(Paper.id))                                                      
        if parsing_status:                                                                       
            stmt = stmt.where(Paper.parsing_status == parsing_status)                            
                                                                                                
        return self.session.execute(stmt).scalar_one()                                           
                                                                                                
    def delete(self, arxiv_id: str) -> bool:                                                     
        """                                                                                      
        Delete a paper.                                                                          
                                                                                                
        Args:                                                                                    
            arxiv_id: arXiv identifier                                                           
                                                                                                
        Returns:                                                                                 
            True if deleted, False if not found                                                  
        """                                                                                      
        paper = self.get_by_arxiv_id(arxiv_id)                                                   
        if not paper:                                                                            
            return False                                                                         
                                                                                                
        self.session.delete(paper)                                                               
        self.session.flush()                                                                     
        return True                                                                              
                                                                                                
    def bulk_upsert(self, papers: list[PaperCreate]) -> int:                                     
        """                                                                                      
        Bulk upsert multiple papers.                                                             
                                                                                                
        Args:                                                                                    
            papers: List of paper creation schemas                                               
                                                                                                
        Returns:                                                                                 
            Number of papers processed                                                           
        """                                                                                      
        if not papers:                                                                           
            return 0                                                                             
                                                                                                
        values = [                                                                               
            {                                                                                    
                "arxiv_id": p.arxiv_id,                                                          
                "title": p.title,                                                                
                "authors": p.authors,                                                            
                "abstract": p.abstract,                                                          
                "categories": p.categories,                                                      
                "published_date": p.published_date,                                              
                "updated_date": p.updated_date,                                                  
                "pdf_url": p.pdf_url,                                                            
                "pdf_content": p.pdf_content,                                                    
                "sections": p.sections,                                                          
                "parsing_status": p.parsing_status,                                              
                "parsing_error": p.parsing_error,                                                
            }                                                                                    
            for p in papers                                                                      
        ]                                                                                        
                                                                                                
        stmt = insert(Paper).values(values)                                                      
        stmt = stmt.on_conflict_do_update(                                                       
            index_elements=["arxiv_id"],                                                         
            set_={                                                                               
                "title": stmt.excluded.title,                                                    
                "authors": stmt.excluded.authors,                                                
                "abstract": stmt.excluded.abstract,                                              
                "categories": stmt.excluded.categories,                                          
                "updated_date": stmt.excluded.updated_date,                                      
                "updated_at": datetime.utcnow(),                                                 
            }                                                                                    
        )                                                                                        
                                                                                                
        self.session.execute(stmt)                                                               
        self.session.flush()                                                                     
        return len(papers)