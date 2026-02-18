"""
What is needed: 
    A new file that creates a LangChain @tool-decorated function wrapping PaperAlchemy's exisiting OpenSearch + 
    JinaEmbedding services into a single callable that LangGraph's ToolNode can execute.
________________________________________________________________________________________________________________
Why it is needed:
    LangGraph's tool-calling flow works like this:
    1. The retrieve_node asks the LLM: "Should i search for paeprs?"
    2. The LLM returns an AIMessage with tool_calls=[{name: "retrieve_papers", args: {query: "..."}}]
    3. LangGraph's ToolNode sees the tool call and executes the matching @tool function
    4. The tool's return value becomes a ToolMessage added to state.

    Without this file, ToolNode has no tool to execute. You'd have to put the embed + search logic directly 
    in a node, bypassing LangGraph's tool-calling protocol entirely.

    The factory function pattern (create_retriever_tool() returns a @tool) is needed because:
    - The tool needs access to opensearch_client and embeddings_clients - live service instaces.
    - @tool functions can't take service clients as parameters (LangGraph passes only the args the LLM specified)
    - The closure captures the clients at creation time, so the @tool only exposes query: str to the LLM

How it helps:
    # At service startup (in AgenticRAGService):
    tool = create_retriever_tool(opensearch_client, embeddings_Client, top_k=3)

    # LangGraph wires it into ToolNode:
    tool_node = ToolNode(tools=[tool])

    # At runtime, LLM decides to call the tool:
    AIMessage(tool_calls) = [{"name": "retrive_papers", "args": {"query": "transformers"}}])
    # -> ToolNode executes retrieve_papers("transformers")
    # -> Returns list[Document] with paper chunks + metadata
    # -> LangGraph adds ToolMessage to state.messages
________________________________________________________________________________________________________________
What it does: Creates a LangChain @tool that wraps OpenSearch + Jina embeddings into a single retrieval
              function. LangGraph's ToolNode calls this tiil when the LLM decides it needs to search for
              papers.
┌────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │       Function         │                              Purpose                                │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ create_retriever_tool  │ Factory that returns a @tool-decorated async function.               │
  │                        │ Captures opensearch_client and embeddings_client via closure.        │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ retrieve_papers        │ The actual @tool. Takes a query string, embeds it via Jina,          │
  │ (inner function)       │ searches OpenSearch (hybrid or BM25), returns list[Document].        │
  └────────────────────────┴─────────────────────────────────────────────────────────────────────┘

Why a factory function:
    LangChain's @tool decorator expects the function signature to match exactly
    what the LLM will call - just (query: str). But the tool needs access to opensearch
    _client and embeddings_client to do its job. The factory pattern solves this: create_retriever_tool()
    captures cleinet via closure, and the returned @tool function only exposes (query: str) to the LLM

Why LangChain Document output:
    LangGraph's ToolNode serializes tool output into ToolMessage.content.
    LangChain's Document class has built-in serialization (page_content + metadata).
    This means downstream nodes can parse the tool results reliably.

How it fits in the graph:
    1. AgenticRAGService creates the tool at startup via create_retriever_tool()
    2. The tool is passed to ToolNode(tools=[retrieve_papers])
    3. retrieve_node creates an AIMessage with tool_calls asking for retrieval
    4. ToolNode executes retrieve_papers(query) and adds the result as ToolMessage
    5. grade_documents_node reads the ToolMessage to grade relevance
________________________________________________________________________________________________________________
"""

import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.services.embeddings.jina_client import JinaEmbeddingsClient
from src.services.opensearch.client import OpenSearchClient

logger = logging.getLogger(__name__)

def create_retriever_tool(
        opensearch_client: OpenSearchClient,
        embeddings_client: JinaEmbeddingsClient,
        top_k: int = 3,
        use_hybrid: bool = True,
):
    """Create a retriver tool that wraps OpenSearch and Jina embeddings.
    What it does:
        Returns a @tool-decorated async function that the LLM can call.
        to search for relevant arXiv papers. The returned function is 
        registered with LangGraph's ToolNode.

    Why it is needed:
        LangGraph's ToolNode requires LangChain-compatible @tool functions.
        The tool must only expose parameters the LLM controls (query).
        Service clients (OpenSearch, Jina) are captured time (from GraphConfig)
        the LLM sees them.

    How it helps:
        - Decouple retrieval logic from node logic
        - ToolNode handles serialization to ToolMessage automatically
        - top_k and use_hybrid are fixed at creation time (from GraphConfig)
        - The tool's docstring tells the LLM WHEN to use it

    Args:
        opensearch_client: Initialized OpenSearch client for searching
        embeddings_client: Initialized Jina client for query embedding
        top_k: Number fo chunks to retrieve per search
        use_hybrid: Whether to use hybrid search (BM25 + vector) or BM25 only

    Returns:
        A @tool-decorated async function ready for ToolNode
    """

    @tool
    async def retrieve_papers(query: str) -> list[Document]:
        """Search and return relevant arXiv research papers.
        
        Use this tool when user asks about:
        - Machine learning concepts or techiniques
        - Deep learning architetcures
        - Natural language processing
        - Computer vision methods
        - AI research topics
        - Specific algorithms or models

        Args:
            query: The search query describing what papers to find

        Returns:
            List of relevant paper excerpts with metadata
        """

        logger.info(f"Retrieving papers for query: {query[:100]}...")
        logger.debug(f"Search model: {'hybrid' if use_hybrid else 'bm25'}, top_k: {top_k}")

        # Step 1: Generate the queryembedding via Jina
        query_embedding = await embeddings_client.embed_query(query)
        logger.debug(f"Generated mebedding with {len(query_embedding)} dimensions")

        # Step 2: Search OpenSearch (hybrid or bm25 based on config)
        search_results = opensearch_client.search_unified(
            query=query,
            query_embedding=query_embedding,
            size=top_k,
            use_hybrid=use_hybrid,
        )

        # Step 3: Convert OpenSearch hits to LangChain Documents
        documents = []
        hits = search_results.get("hits", [])
        logger.info(f"Found {len(hits)} documents from OpenSearch")

        for hit in hits:
            doc = Document(
                page_content=hit["chunk_text"],
                metadata={
                    "arxiv_id": hit["arxiv_id"],
                    "title": hit.get("title", ""),
                    "authors": hit.get("authors", ""),
                    "score": hit.get("score", 0.0),
                    "source": f"https://arxiv.org/pdf/{hit['arxiv_id']}.pdf",
                    "section": hit.get("section_name", ""),
                    "search_mode": "hybrid" if use_hybrid else "bm25",
                    "top_k": top_k,
                },
            )
            documents.append(doc)

        logger.info(f"Retrived {len(documents)} papers successfully")
        return documents
    
    return retrieve_papers

