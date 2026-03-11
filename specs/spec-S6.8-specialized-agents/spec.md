# Spec S6.8 -- Specialized Agents

## Overview
Multi-agent collaboration system for paper analysis. Four specialized agents — Summarizer, Fact-Checker, Trend Analyzer, and Citation Tracker — each with their own LangGraph sub-graph that can be invoked by the main agent orchestrator (S6.7) or directly via API. Each agent is a focused expert that leverages the retrieval pipeline and LLM to perform a specific analysis task on papers.

## Dependencies
- **S6.7** — Agent orchestrator (LangGraph StateGraph compilation): The main orchestrator routes requests to specialized agents and composes their outputs.

## Target Location
- `src/services/agents/specialized/` — Agent implementations
- `src/services/agents/specialized/__init__.py` — Public exports
- `src/services/agents/specialized/base.py` — Base specialized agent class
- `src/services/agents/specialized/summarizer.py` — Paper Summarizer agent
- `src/services/agents/specialized/fact_checker.py` — Fact-Checker agent
- `src/services/agents/specialized/trend_analyzer.py` — Trend Analyzer agent
- `src/services/agents/specialized/citation_tracker.py` — Citation Tracker agent
- `tests/unit/test_specialized_agents.py` — Unit tests

## Functional Requirements

### FR-1: Base Specialized Agent
- **What**: Abstract base class/protocol that all specialized agents implement. Defines common interface for invocation, state management, and output formatting.
- **Inputs**: `query: str`, `context: AgentContext`, optional `papers: list[SourceItem]` (pre-retrieved papers to analyze)
- **Outputs**: `SpecializedAgentResult` — structured output with `agent_name`, `analysis`, `sources`, `metadata`
- **Edge cases**: No papers provided (agent retrieves its own), LLM failure (graceful fallback), empty results

### FR-2: Summarizer Agent
- **What**: Generates structured summaries of academic papers. When given a paper or set of papers, produces a structured summary covering: objective, methodology, key findings, contributions, and limitations.
- **Inputs**: `query: str` (e.g., "Summarize the Attention Is All You Need paper"), `context: AgentContext`, optional `papers: list[SourceItem]`
- **Outputs**: `SummarizerResult` with fields: `objective`, `methodology`, `key_findings: list[str]`, `contributions: list[str]`, `limitations: list[str]`, `summary_text: str` (formatted markdown)
- **Edge cases**: Paper not found, paper too long (chunk and summarize per-section), multiple papers (summarize each)

### FR-3: Fact-Checker Agent
- **What**: Cross-references claims in a paper or query against other papers in the knowledge base. Verifies factual claims by finding supporting or contradicting evidence.
- **Inputs**: `query: str` (e.g., "Verify the claim that transformers outperform RNNs on all NLP tasks"), `context: AgentContext`, optional `papers: list[SourceItem]`
- **Outputs**: `FactCheckResult` with fields: `claims: list[ClaimVerification]` where each has `claim: str`, `verdict: Literal["supported", "contradicted", "insufficient_evidence"]`, `evidence: list[SourceItem]`, `explanation: str`
- **Edge cases**: Vague claims, no evidence found, conflicting evidence from multiple papers

### FR-4: Trend Analyzer Agent
- **What**: Analyzes research trends across papers in the knowledge base. Identifies emerging topics, methodological shifts, and citation patterns over time.
- **Inputs**: `query: str` (e.g., "What are the trends in NLP research over the last 3 years?"), `context: AgentContext`, optional `topic: str`
- **Outputs**: `TrendAnalysisResult` with fields: `trends: list[TrendItem]` where each has `topic: str`, `direction: Literal["rising", "stable", "declining"]`, `key_papers: list[SourceItem]`, `description: str`; plus `timeline: str` (markdown), `emerging_topics: list[str]`
- **Edge cases**: Too few papers for trend analysis, narrow topic with no trend data

### FR-5: Citation Tracker Agent
- **What**: Tracks citation relationships between papers. Finds papers that cite a given paper, papers cited by it, and maps citation networks.
- **Inputs**: `query: str` (e.g., "What papers cite Attention Is All You Need?"), `context: AgentContext`, optional `paper_id: str`
- **Outputs**: `CitationTrackResult` with fields: `target_paper: SourceItem`, `cited_by: list[SourceItem]` (papers in KB that reference target), `references: list[SourceItem]` (papers in KB that target cites), `citation_count: int`, `influence_summary: str`
- **Edge cases**: Paper not in knowledge base, no citation relationships found, circular citations

### FR-6: Agent Registry & Dispatch
- **What**: Registry that maps agent names to implementations. Provides a dispatch mechanism for the orchestrator to invoke the right specialized agent based on query intent.
- **Inputs**: Agent name or query intent classification
- **Outputs**: The appropriate specialized agent instance
- **Edge cases**: Unknown agent type (fallback to general RAG), ambiguous intent

## Tangible Outcomes
- [ ] `SpecializedAgentBase` protocol/ABC defines common interface
- [ ] `SummarizerAgent` produces structured paper summaries with all 5 sections
- [ ] `FactCheckerAgent` verifies claims with supporting/contradicting evidence
- [ ] `TrendAnalyzerAgent` identifies research trends with direction and key papers
- [ ] `CitationTrackerAgent` maps citation relationships within the knowledge base
- [ ] `AgentRegistry` maps names to agent classes and dispatches correctly
- [ ] All agents cite sources with title, authors, and arXiv links
- [ ] All agents handle missing papers gracefully
- [ ] All agents use the retrieval pipeline (not LLM memory alone)
- [ ] Unit tests cover all agents with mocked LLM and retrieval

## Test-Driven Requirements

### Tests to Write First
1. `test_specialized_agent_base_protocol`: Verify base interface contract
2. `test_summarizer_produces_structured_output`: Summarizer returns all 5 sections
3. `test_summarizer_handles_no_papers`: Graceful fallback when no papers found
4. `test_fact_checker_supported_claim`: Returns "supported" with evidence
5. `test_fact_checker_contradicted_claim`: Returns "contradicted" with evidence
6. `test_fact_checker_insufficient_evidence`: Returns "insufficient_evidence"
7. `test_trend_analyzer_identifies_trends`: Returns trends with direction
8. `test_trend_analyzer_emerging_topics`: Identifies emerging topics
9. `test_citation_tracker_finds_citations`: Maps citation relationships
10. `test_citation_tracker_paper_not_found`: Handles missing paper
11. `test_agent_registry_dispatch`: Dispatches to correct agent
12. `test_agent_registry_unknown_type`: Falls back for unknown agent
13. `test_all_agents_cite_sources`: Every agent includes arXiv links in output
14. `test_all_agents_use_retrieval`: Verify retrieval pipeline is called

### Mocking Strategy
- Mock `AgentContext.llm_provider` with `AsyncMock` returning structured outputs
- Mock `AgentContext.retrieval_pipeline` with `AsyncMock` returning `RetrievalResult`
- Use fixtures for `SourceItem` lists with realistic paper data
- Mock LLM `with_structured_output()` to return Pydantic models directly

### Coverage
- All public functions tested
- Edge cases covered (no papers, LLM failure, empty results)
- Error paths tested (graceful degradation)
- Source citation verified in all agent outputs
