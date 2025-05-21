import os
from dotenv import load_dotenv


load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL", "******")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "**********")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "*******")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "**********")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


DEFAULT_SYSTEM_MESSAGE = "Ben bir RAG asistanıyım. Yüklediğiniz dokümanlar ve web kaynaklarını kullanarak sorularınızı yanıtlayabilirim."

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY