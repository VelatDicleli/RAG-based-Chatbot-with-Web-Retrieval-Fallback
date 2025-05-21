from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from core.state import RAGState
from nodes.document_processor import upload_file
from nodes.retriever import retrieve
from nodes.search import prepare_search_query, use_tools
from nodes.generator import generate
from nodes.decision import decide_next_node
from core.logger import setup_logger

logger = setup_logger(__name__)

def build_rag_graph():
    """Build the RAG workflow graph."""
    logger.info("Building RAG graph")
    
    # Create a state graph
    graph = StateGraph(RAGState)

    # Add nodes to the graph
    graph.add_node("upload_file", upload_file)
    graph.add_node("retrieve", retrieve)
    graph.add_node("prepare_search_query", prepare_search_query)
    graph.add_node("use_tools", use_tools)
    graph.add_node("generate", generate)

    # Define edges between nodes
    graph.set_entry_point("upload_file")
    graph.add_edge("upload_file", "retrieve")
    
    # Add conditional edge from retrieve to either prepare_search_query or generate
    graph.add_conditional_edges(
        "retrieve",
        decide_next_node,
        {
            "prepare_search_query": "prepare_search_query",
            "generate": "generate"
        }
    )
    
    # Complete the flow
    graph.add_edge("prepare_search_query", "use_tools")
    graph.add_edge("use_tools", "generate")
    graph.add_edge("generate", END)

    # Use in-memory checkpointer for state persistence
    mem = InMemorySaver()
    
    # Compile the graph
    rag_graph = graph.compile(checkpointer=mem)
    logger.info("RAG graph built successfully")
    
    return rag_graph