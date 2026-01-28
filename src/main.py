"""PaperAlchemy - FastAPI Application Entry Point."""                                              
                                                                                                     
from fastapi import FastAPI                                                                        
from fastapi.middleware.cors import CORSMiddleware                                                 
                                                                                                    
from src.config import settings                                                                    
                                                                                                    
app = FastAPI(                                                                                     
    title="PaperAlchemy",                                                                          
    description="Transform Academic Papers into Knowledge Gold - Production RAG System",           
    version="0.1.0",                                                                               
)                                                                                                  
                                                                                                    
# CORS middleware                                                                                  
app.add_middleware(                                                                                
    CORSMiddleware,                                                                                
    allow_origins=["*"],                                                                           
    allow_credentials=True,                                                                        
    allow_methods=["*"],                                                                           
    allow_headers=["*"],                                                                           
)                                                                                                  
                                                                                                    
                                                                                                    
@app.get("/")                                                                                      
async def root():                                                                                  
    """Root endpoint."""                                                                           
    return {                                                                                       
        "name": "PaperAlchemy",                                                                    
        "description": "Transform Academic Papers into Knowledge Gold",                            
        "version": "0.1.0",                                                                        
    }                                                                                              
                                                                                                    
                                                                                                    
@app.get("/health")                                                                                
async def health_check():                                                                          
    """Health check endpoint."""                                                                   
    return {                                                                                       
        "status": "healthy",                                                                       
        "debug": settings.app.debug,                                                               
    }