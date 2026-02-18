"""
What: Package marker for the agents module.
                                                                                                                            
Why: Makes src.services.agents an explicit Python package. Without it, Python relies on implicit namespace packages which   
can cause issues with some tools (pytest, mypy, IDE indexing). Every other service package in your project (ollama/, cache/,
langfuse/) has one.

How: Zero runtime cost. It simply signals to Python's import system that this directory is a package. All imports like from
src.services.agents.models import ... route through it.

"""                                                                                                                            

"""Agentic RAG package — LangGraph-powered multi-step research assistant."""