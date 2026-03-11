# Spec S6.6 -- Answer Generation Node

## Overview
LangGraph agent node that generates citation-backed answers from relevant sources.
Takes graded/relevant documents from the agent state, constructs a citation-enforcing prompt,
invokes the LLM, and post-processes the output to ensure every answer has inline [N] references
mapped to real papers with title, authors, and arXiv links.

Uses the citation enforcement module from S5.5 to validate and format citations.

## Dependencies
- S6.1 (Agent state & context) — AgentState, AgentContext, SourceItem
- S5.5 (Citation enforcement) — parse_citations, validate_citations, format_source_list, enforce_citations

## Target Location
- `src/services/agents/nodes/generate_answer_node.py`

## Functional Requirements

### FR-1: Generation Prompt Construction
- **What**: Build a system+user prompt that includes the user query, relevant source chunks with numbered references, and instructions to cite sources inline using [N] notation.
- **Inputs**: AgentState (messages, relevant_sources, original_query/rewritten_query)
- **Outputs**: Formatted prompt string
- **Edge cases**: No relevant sources (generate "no papers found" response), empty query

### FR-2: LLM Answer Generation
- **What**: Invoke the LLM via AgentContext.llm_provider to generate a citation-backed answer. Uses the latest query (rewritten_query if available, else original_query).
- **Inputs**: AgentState, AgentContext
- **Outputs**: Raw LLM answer string
- **Edge cases**: LLM failure (return error message in state), timeout

### FR-3: Citation Post-Processing
- **What**: Convert SourceItems to SourceReferences, build a RAGResponse, run enforce_citations() from S5.5 to validate inline [N] refs and append formatted source list.
- **Inputs**: Raw LLM answer, relevant_sources from state
- **Outputs**: CitationResult with formatted_answer, validation, sources_markdown
- **Edge cases**: LLM produces no citations (validation.is_valid=False), invalid citation indices

### FR-4: State Update with AIMessage
- **What**: Add the final formatted answer as an AIMessage to state messages. Store citation metadata in state metadata. Return partial state dict.
- **Inputs**: CitationResult, current state
- **Outputs**: Partial AgentState dict with updated messages and metadata
- **Edge cases**: Empty answer after processing

### FR-5: No-Sources Fallback
- **What**: When relevant_sources is empty, generate a polite "I don't have papers on that topic" response instead of hallucinating. Skip LLM call entirely.
- **Inputs**: AgentState with empty relevant_sources
- **Outputs**: Partial state with fallback AIMessage
- **Edge cases**: N/A

## Tangible Outcomes
- [ ] `generate_answer_node.py` exists with `ainvoke_generate_answer_step` function
- [ ] Function signature: `async def ainvoke_generate_answer_step(state: AgentState, context: AgentContext) -> dict[str, Any]`
- [ ] Prompt includes numbered source chunks and citation instructions
- [ ] Uses S5.5 `enforce_citations()` for post-processing
- [ ] Returns AIMessage in messages with formatted answer + source list
- [ ] No-sources fallback returns honest "no papers found" message
- [ ] Citation validation metadata stored in state metadata
- [ ] All tests pass with mocked LLM
- [ ] Registered in `nodes/__init__.py`

## Test-Driven Requirements

### Tests to Write First
1. `test_generation_prompt_includes_sources`: Verify prompt construction includes numbered sources and citation instructions
2. `test_generate_answer_with_sources`: Full flow — mocked LLM returns cited answer, verify AIMessage and citation validation
3. `test_generate_answer_no_sources_fallback`: Empty relevant_sources triggers fallback message
4. `test_generate_answer_llm_failure`: LLM raises exception, node returns error message gracefully
5. `test_citation_post_processing`: Verify SourceItem → SourceReference conversion and enforce_citations integration
6. `test_uses_rewritten_query_when_available`: Prefer rewritten_query over original_query
7. `test_metadata_includes_citation_info`: Verify citation validation stored in metadata

### Mocking Strategy
- Mock `context.llm_provider.get_langchain_model()` to return a mock LLM
- Mock LLM `.ainvoke()` to return controlled answer text with [1], [2] citations
- Use real `enforce_citations()` from S5.5 (no need to mock — pure function)
- Provide fixture SourceItems with realistic arxiv_id, title, authors, url

### Coverage
- All public functions tested
- Edge cases: no sources, LLM failure, no citations in LLM output
- Error paths tested
