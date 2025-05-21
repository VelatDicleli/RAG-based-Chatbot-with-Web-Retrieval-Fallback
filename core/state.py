from typing import Annotated, List, Optional, TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    """State class for RAG workflow"""
    documents: Optional[List[Document]]
    answer: Optional[str]
    file_path: Optional[str]
    tool_result: Optional[str]
    collection: Optional[str]
    messages: Annotated[list, add_messages]
    tool_used: Optional[bool]
    search_query: Optional[str]