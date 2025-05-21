from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from models.llm import get_llm
from core.state import RAGState
from core.logger import setup_logger
from langgraph.prebuilt import create_react_agent

logger = setup_logger(__name__)

def decide_next_node(state: RAGState) -> str:
    """Decide next action based on available documents and query type."""
    logger.info("Deciding next node")
    
  
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    
    if not last_message:
        logger.warning("No human message found for decision making")
        return "generate"
        
    user_query = last_message.content
    documents = state.get("documents", [])
    
    try:
        
        doc_summary = "\n".join([f"Belge {i+1}: {doc.page_content[:100]}..." for i, doc in enumerate(documents)])
        
        decision_prompt = PromptTemplate.from_template("""
            Sen, bilgiye dayalı kararlar veren bir yapay zeka agentsin. Aşağıda kullanıcıdan gelen bir mesaj ve varsa ilgili belgeler verilmiştir. Görevin, bu girdilere göre bir sonraki uygun aksiyonu belirlemektir.

            Kullanıcı mesajı:
            {message}

            Belgeler:
            {documents}

            Karar Kuralları:
            - Eğer kullanıcı mesajı zamanla değişen, güncel bir bilgi içeriyorsa (örnek: hava durumu, döviz kuru, maç sonucu, güncel haberler, trafik durumu gibi) ve belgelerde bu bilgi yoksa ya da güncel değilse: "prepare_search_query" döndür.
            - Eğer belgeler, kullanıcının sorusuna doğrudan, açık ve yeterli cevap sağlayacak durumdaysa: "generate" döndür.
            - Eğer belgeler eksikse, yoksa ya da kullanıcı mesajı yanıtlanamayacak kadar belirsizse: "prepare_search_query" döndür.
            - Sadece aşağıdaki iki seçenekten birini döndür:
            - "generate"
            - "prepare_search_query"
            - Hiçbir açıklama yapma, sadece karar kelimesini tek satır olarak döndür.
            """)
        
        llm = get_llm()
        decision = llm.invoke(decision_prompt.format(
            message=user_query, 
            documents=doc_summary if documents else "Belge bulunamadı."
        ))
        
        result = decision.content.strip().lower()
        logger.info(f"Decision: {result}")
        
        if "prepare_search_query" in result:
            return "prepare_search_query"
        else:
            return "generate"
            
    except Exception as e:
        logger.error(f"Error in decision making: {str(e)}")
        return "generate"