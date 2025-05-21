from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from models.llm import get_llm
from services.tools import SearchTools
from core.state import RAGState
from core.logger import setup_logger

logger = setup_logger(__name__)

def prepare_search_query(state: RAGState) -> RAGState:
    """Prepare search query for external tools."""
    logger.info("Preparing search query")
    
   
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    
    if not last_message:
        logger.warning("No human message found for search query preparation")
        state["search_query"] = "No query provided"
        return state
        
    user_query = last_message.content
    
    try:
        llm = get_llm()
        search_prompt = ChatPromptTemplate.from_messages([
            ("system", "Kullanıcının sorusunu dış kaynaklarda arama yapmak için kısa ve net bir sorguya dönüştür. Sadece sorgu metnini döndür."),
            ("human", "{input}")
        ])
        
        response = llm.invoke(search_prompt.format_messages(input=user_query))
        
        search_query = response.content.strip()
        logger.info(f"Generated search query: {search_query}")
        
        state["search_query"] = search_query
    except Exception as e:
        logger.error(f"Error preparing search query: {str(e)}")
        
        state["search_query"] = user_query
        
    return state


def use_tools(state: RAGState) -> RAGState:
    """Use external tools to search for information."""
    search_query = state.get("search_query")
    if not search_query:
        logger.warning("No search query provided for tools")
        state["tool_result"] = "Arama sorgusu bulunamadı."
        state["tool_used"] = True
        return state
        
    logger.info(f"Using tools with search query: {search_query}")
    
    try:
  
        search_tools = SearchTools()
        
        
        result = search_tools.search_all(search_query)
        
        state["tool_result"] = result
        state["tool_used"] = True
        logger.info("Tool search completed")
            
    except Exception as e:
        logger.error(f"Tool usage error: {str(e)}")
        state["tool_result"] = f"Araçlar kullanılırken bir hata oluştu: {str(e)}"
        state["tool_used"] = True
    
    return state