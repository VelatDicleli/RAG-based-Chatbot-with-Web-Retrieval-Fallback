from langchain_core.messages import HumanMessage
from services.vector_store import VectorStoreService
from core.state import RAGState
from core.logger import setup_logger

logger = setup_logger(__name__)

def retrieve(state: RAGState) -> RAGState:
    """Retrieve similar documents from vector database."""
    logger.info("Starting document retrieval")
   
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    
    if not last_message:
        logger.warning("No human message found in history")
        user_query = "No query provided"
    else:
        user_query = last_message.content
    
    logger.info(f"Retrieving documents for query: {user_query}")
    
    collection = state.get("collection")
    if not collection:
        logger.error("No collection specified for retrieval")
        state["documents"] = []
        return state
    
    try:
    
        vector_service = VectorStoreService()
        vector_docs = vector_service.similarity_search(collection, user_query, k=5)
        
        logger.info(f"Retrieved {len(vector_docs)} documents from vector store")
        
        
        all_docs = []
        doc_contents = set()
        
        for doc in vector_docs:
            if doc.page_content not in doc_contents:
                all_docs.append(doc)
                doc_contents.add(doc.page_content)
        
        state["documents"] = all_docs[:5]  
        logger.info(f"Added {len(state['documents'])} unique documents to state")
        
    except Exception as e:
        logger.error(f"Error during document retrieval: {str(e)}")
        state["documents"] = []
    
    return state