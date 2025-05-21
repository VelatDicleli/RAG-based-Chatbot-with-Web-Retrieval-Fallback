from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from models.llm import get_llm
from core.state import RAGState
from core.logger import setup_logger

logger = setup_logger(__name__)

def generate(state: RAGState) -> RAGState:
    """Generate response based on retrieved documents and tool results."""
    logger.info("Generating response")
    
    docs = state.get("documents", [])
    tool_result = state.get("tool_result", "")
    tool_used = state.get("tool_used", False)
    
  
    if not state.get("messages"):
        logger.warning("No messages in state for response generation")
        state["messages"] = [
            SystemMessage(content="Ben bir RAG asistanıyım. Yüklediğiniz dokümanlar ve web kaynaklarını kullanarak sorularınızı yanıtlayabilirim.")
        ]
        state["answer"] = "Üzgünüm, bir sorunuz var mı?"
        state["messages"].append(AIMessage(content=state["answer"]))
        return state
    
    
    last_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_message = msg
            break
    
    if not last_message:
        logger.warning("No human message found for response generation")
        state["answer"] = "Üzgünüm, sorunuzu anlamadım. Lütfen tekrar sorar mısınız?"
        state["messages"].append(AIMessage(content=state["answer"]))
        return state
    
    
    doc_context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
    
    if doc_context and tool_result and tool_used:
        system_prompt = (
            "Aşağıdaki verileri kullanarak kullanıcının sorusunu cevaplayın:\n\n"
            f"Veritabanından Bilgiler:\n{doc_context}\n\n"
            f"Dış Kaynak Araştırması:\n{tool_result}\n\n"
            "Cevabınızda hem veritabanı bilgilerini hem de dış kaynak araştırmasını kullanın. "
            "Önce veritabanında bulunan bilgileri, sonra dış kaynak bilgilerini değerlendirerek birlikte cevap ver. "
            "Hangi kaynaklardan bilgi kullandığınızı belirtin. "
            "Tüm yanıtlarını Türkçe olarak ver."
        )
    elif doc_context:
        system_prompt = (
            "Aşağıdaki veritabanı bilgilerini kullanarak kullanıcının sorusunu cevaplayın:\n\n"
            f"Veritabanından Bilgiler:\n{doc_context}\n\n"
            "Cevabınızda veritabanındaki bilgilere dayalı olarak yanıt verin. "
            "Tüm yanıtlarını Türkçe olarak ver."
        )
    elif tool_result and tool_used:
        system_prompt = (
            "Aşağıdaki dış kaynak araştırmasını kullanarak kullanıcının sorusunu cevaplayın:\n\n"
            f"Dış Kaynak Araştırması:\n{tool_result}\n\n"
            "Veritabanında ilgili bilgi bulunamadığı için dış kaynaklardan elde edilen bilgileri kullanarak yanıt ver. "
            "Kullandığınız dış kaynakları belirt. "
            "Tüm yanıtlarını Türkçe olarak ver."
        )
    else:
        system_prompt = (
            "Kullanıcının sorusuna en iyi şekilde cevap verin. "
            "Ne veritabanında ne de dış kaynaklarda yeterli bilgi bulunamadı. "
            "Genel bilginizi kullanarak yanıtlayın ve bilgi eksikliğini belirtin. "
            "Tüm yanıtlarını Türkçe olarak ver."
        )

    try:
        llm = get_llm()
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        chat_history = [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
        
       
        prompt = qa_prompt.format_messages(
            chat_history=chat_history[:-1] if last_message in chat_history else chat_history,
            input=last_message.content
        )

        response = llm.invoke(prompt)
        answer_content = response.content if response.content else "Üzgünüm, şu anda bir cevap veremiyorum."
        logger.info("Response generated successfully")

        state["answer"] = answer_content
        state["messages"].append(AIMessage(content=answer_content))

    except Exception as e:
        logger.error(f"Error during response generation: {str(e)}")
        state["answer"] = f"Üzgünüm, cevap oluşturulurken bir hata oluştu: {str(e)}"
        state["messages"].append(AIMessage(content=state["answer"]))
    
    return state