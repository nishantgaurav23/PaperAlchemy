"""Specialized agents for paper analysis (S6.8).

Provides four specialized agents — Summarizer, Fact-Checker, Trend Analyzer,
and Citation Tracker — plus an AgentRegistry for dispatch.
"""

from __future__ import annotations

from src.services.agents.specialized.base import SpecializedAgentBase, SpecializedAgentResult
from src.services.agents.specialized.citation_tracker import CitationTrackerAgent, CitationTrackResult
from src.services.agents.specialized.fact_checker import ClaimVerification, FactCheckerAgent, FactCheckResult
from src.services.agents.specialized.summarizer import SummarizerAgent, SummarizerResult
from src.services.agents.specialized.trend_analyzer import TrendAnalysisResult, TrendAnalyzerAgent, TrendItem


class AgentRegistry:
    """Registry for specialized agents. Maps names to agent instances."""

    def __init__(self) -> None:
        self._agents: dict[str, SpecializedAgentBase] = {
            "summarizer": SummarizerAgent(),
            "fact_checker": FactCheckerAgent(),
            "trend_analyzer": TrendAnalyzerAgent(),
            "citation_tracker": CitationTrackerAgent(),
        }

    @property
    def agent_names(self) -> list[str]:
        """List of registered agent names."""
        return list(self._agents.keys())

    def get(self, name: str) -> SpecializedAgentBase | None:
        """Get a specialized agent by name, or None if not found."""
        return self._agents.get(name)


__all__ = [
    "AgentRegistry",
    "CitationTrackResult",
    "CitationTrackerAgent",
    "ClaimVerification",
    "FactCheckResult",
    "FactCheckerAgent",
    "SpecializedAgentBase",
    "SpecializedAgentResult",
    "SummarizerAgent",
    "SummarizerResult",
    "TrendAnalysisResult",
    "TrendAnalyzerAgent",
    "TrendItem",
]
