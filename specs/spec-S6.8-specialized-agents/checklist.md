# Checklist -- Spec S6.8: Specialized Agents

## Phase 1: Setup & Dependencies
- [x] Verify S6.7 (Agent Orchestrator) is "done"
- [x] Create `src/services/agents/specialized/` directory
- [x] Create `__init__.py` with public exports

## Phase 2: Tests First (TDD)
- [x] Create `tests/unit/test_specialized_agents.py`
- [x] Write test fixtures (mock context, sample papers, mock LLM)
- [x] Write failing tests for SpecializedAgentBase protocol
- [x] Write failing tests for SummarizerAgent (FR-2)
- [x] Write failing tests for FactCheckerAgent (FR-3)
- [x] Write failing tests for TrendAnalyzerAgent (FR-4)
- [x] Write failing tests for CitationTrackerAgent (FR-5)
- [x] Write failing tests for AgentRegistry (FR-6)
- [x] Run tests -- expect failures (Red)

## Phase 3: Implementation
- [x] Implement `base.py` — SpecializedAgentBase protocol + SpecializedAgentResult
- [x] Implement `summarizer.py` — SummarizerAgent + SummarizerResult
- [x] Implement `fact_checker.py` — FactCheckerAgent + FactCheckResult + ClaimVerification
- [x] Implement `trend_analyzer.py` — TrendAnalyzerAgent + TrendAnalysisResult + TrendItem
- [x] Implement `citation_tracker.py` — CitationTrackerAgent + CitationTrackResult
- [x] Implement AgentRegistry in `__init__.py`
- [x] Run tests -- expect pass (Green) — 34/34 passed
- [x] Refactor if needed

## Phase 4: Integration
- [x] Export agents from `src/services/agents/specialized/__init__.py`
- [x] Wire registry into `src/services/agents/__init__.py`
- [x] Run lint (`ruff check`) — all passed
- [x] Run full test suite — 78/78 agent tests passed

## Phase 5: Verification
- [x] All tangible outcomes checked
- [x] All agents cite sources with arXiv links
- [x] All agents use retrieval pipeline (not LLM memory)
- [x] No hardcoded secrets
- [x] Create notebook: `notebooks/specs/S6.8_specialized.ipynb`
- [x] Update roadmap.md status to "done"
