from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from config.settings import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from core.logger import setup_logger

logger = setup_logger(__name__)

def get_llm():
    """
    Initialize and return the LLM for generation.
    """
    try:
        logger.info(f"Initializing LLM with model: {LLM_MODEL}")
        llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            api_key=GROQ_API_KEY
        )
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM: {str(e)}")
        raise

def get_embeddings_model():
    """
    Initialize and return the embeddings model.
    """
    try:
        logger.info("Initializing HuggingFace embeddings model")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        raise