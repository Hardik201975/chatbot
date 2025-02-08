import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import os
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles all model-related operations optimized for Render deployment"""
    
    def __init__(self):
        try:
            logger.info("Starting ModelHandler initialization...")
            
            # Configure Google Generative AI
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.MODEL_NAME)
            logger.info("Google AI configured successfully")
            
            # Set HuggingFace cache directory
            os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
            os.environ['HF_HOME'] = '/tmp/huggingface'
            
            # Initialize embeddings directly with SentenceTransformer
            logger.info("Initializing embeddings model...")
            self.embeddings_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embeddings model initialized successfully")
            
            # Initialize vector store and document storage
            self.vector_store: Optional[faiss.IndexFlatL2] = None
            self.document_texts: List[str] = []
            
            logger.info("ModelHandler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ModelHandler: {str(e)}")
            raise RuntimeError(f"Failed to initialize ModelHandler: {str(e)}")

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings using SentenceTransformer"""
        return self.embeddings_model.encode(text, convert_to_numpy=True)

    def create_vector_store(self, texts: List[str]):
        """Create a FAISS vector store with memory optimization"""
        if not texts:
            raise ValueError("No texts provided for vector store creation")
            
        try:
            # Generate embeddings in batches to manage memory
            batch_size = 32
            vector_list = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = [self.embed_text(text) for text in batch]
                vector_list.extend(batch_vectors)
                logger.debug(f"Processed batch {i//batch_size + 1}")

            vectors = np.array(vector_list).astype('float32')
            
            # Create and configure FAISS index
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to index
            index.add(vectors)
            
            # Update instance variables
            self.vector_store = index
            self.document_texts = texts
            
            logger.info(f"Vector store created successfully with {len(texts)} documents")
            return self
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise RuntimeError(f"Failed to create vector store: {str(e)}")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant document chunks"""
        if not self.vector_store:
            logger.warning("No vector store available for context retrieval")
            return []
            
        try:
            # Generate query embedding
            query_vector = np.array([self.embed_text(query)]).astype('float32')
            
            # Search in vector store
            D, I = self.vector_store.search(query_vector, min(top_k, len(self.document_texts)))
            
            # Retrieve corresponding texts
            context = [self.document_texts[i] for i in I[0]]
            logger.debug(f"Retrieved {len(context)} context chunks")
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def generate_response(self, query: str) -> str:
        """Generate response using context-aware prompt"""
        try:
            context = self.retrieve_context(query)
            context_text = ' '.join(context) if context else "No relevant context found."
            
            prompt = f"""
            Context:
            {context_text}
            
            Question: {query}
            
            Please provide a precise and helpful answer based on the given context.
            If the context does not contain sufficient information,
            explain that you cannot find a specific answer in the provided document.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.MAX_TOKENS,
                    temperature=settings.TEMPERATURE
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."