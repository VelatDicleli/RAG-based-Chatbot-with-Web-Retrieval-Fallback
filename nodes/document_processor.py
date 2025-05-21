import os
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_SYSTEM_MESSAGE
from services.vector_store import VectorStoreService
from core.state import RAGState
from core.logger import setup_logger

logger = setup_logger(__name__)

def load_documents(file_path):
    """Load documents from the given file path using the appropriate loader."""
    logger.info(f"Loading documents from: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
            
        
        if file_path.lower().endswith(".pdf"):
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(".csv"):
            logger.info(f"Loading CSV: {file_path}")
            loader = CSVLoader(file_path)
        elif os.path.isdir(file_path):
            logger.info(f"Loading directory: {file_path}")
            loader = DirectoryLoader(file_path, glob="**/*.*")
        else:
            logger.info(f"Loading text file: {file_path}")
            loader = TextLoader(file_path, encoding="utf-8")
            
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

def split_documents(documents):
    """Split documents into chunks for embedding."""
    logger.info(f"Splitting {len(documents)} documents into chunks")
    
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(splits)} text chunks")
        return splits
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

def upload_file(state: RAGState) -> RAGState:
    """Upload file and save to vector database."""
    logger.info("Starting file upload process")
    
    file_path = state.get("file_path")
    if not file_path:
        logger.error("No file path provided in state")
        raise ValueError("No file path provided")
    
 
    vector_service = VectorStoreService()

    if not state.get("collection"):
        state["collection"] = vector_service.create_collection()
    else:
        logger.info(f"Using existing collection: {state['collection']}")
    
    try:
      
        documents = load_documents(file_path)
        splits = split_documents(documents)
        
       
        vector_service.add_documents(state["collection"], splits)
        
       
        if not state.get("messages"):
            state["messages"] = []
            
       
        if len(state.get("messages", [])) == 0:
            state["messages"] = [
                SystemMessage(content=DEFAULT_SYSTEM_MESSAGE)
            ]
            
        state["tool_used"] = False
        logger.info("File upload completed successfully")
    
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise
    
    return state