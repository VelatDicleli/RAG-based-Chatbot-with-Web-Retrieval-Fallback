import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from models.llm import get_embeddings_model
from config.settings import QDRANT_URL, QDRANT_API_KEY
from core.logger import setup_logger

logger = setup_logger(__name__)

class VectorStoreService:
    """Service for interacting with the vector database."""
    
    def __init__(self):
        """Initialize the vector store service."""
        try:
            logger.info(f"Connecting to Qdrant at {QDRANT_URL}")
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY
            )
            self.embeddings = get_embeddings_model()
        except Exception as e:
            logger.error(f"Error initializing Vector Store Service: {str(e)}")
            raise
    
    def create_collection(self):
        """Create a new collection with a unique ID."""
        try:
            collection = str(uuid.uuid4())
            logger.info(f"Creating new collection: {collection}")
            
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    def get_store(self, collection_name):
        """Get a QdrantVectorStore instance for the given collection."""
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
    
    def add_documents(self, collection_name, documents):
        """Add documents to the vector store."""
        try:
            store = self.get_store(collection_name)
            logger.info(f"Adding {len(documents)} documents to collection {collection_name}")
            store.add_documents(documents=documents)
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def similarity_search(self, collection_name, query, k=5):
        """Perform similarity search."""
        try:
            store = self.get_store(collection_name)
            logger.info(f"Searching for '{query}' in collection {collection_name}")
            return store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []