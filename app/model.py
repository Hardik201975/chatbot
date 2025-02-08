# app/model.py
import google.generativeai as genai
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Optional
import logging
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """Handles all model-related operations optimized for Render deployment"""
    
    def __init__(self):
        try:
            logger.info("Starting ModelHandler initialization...")
            
            # Test import
            try:
                import sentence_transformers
                logger.info("sentence-transformers package imported successfully")
            except ImportError as e:
                logger.error(f"Failed to import sentence-transformers: {e}")
                raise
                
            # Configure Google Generative AI
            logger.info("Configuring Google Generative AI...")
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.MODEL_NAME)
            logger.info("Google Generative AI configured successfully")
            
            # Initialize embeddings
            logger.info("Initializing embeddings model...")
            self.embeddings = self.initialize_embeddings()
            logger.info("Embeddings model initialized successfully")
            
            # Initialize vector store and document storage
            self.vector_store: Optional[faiss.IndexFlatL2] = None
            self.document_texts: List[str] = []
            
            logger.info("ModelHandler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ModelHandler: {str(e)}")
            raise RuntimeError(f"Failed to initialize ModelHandler: {str(e)}")

    def initialize_embeddings(self):
        """Initialize embeddings using Sentence Transformers with error handling"""
        try:
            return HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                cache_folder="/tmp/hf_cache"  # Use temporary directory on Render
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")

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
                batch_vectors = [self.embeddings.embed_query(text) for text in batch]
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
        """Retrieve most relevant document chunks with error handling"""
        if not self.vector_store:
            logger.warning("No vector store available for context retrieval")
            return []
            
        try:
            # Generate query embedding
            query_vector = np.array([self.embeddings.embed_query(query)]).astype('float32')
            
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
        """Generate response using context-aware prompt with improved error handling"""
        try:
            # Retrieve relevant context
            context = self.retrieve_context(query)
            context_text = ' '.join(context) if context else "No relevant context found."
            
            # Construct prompt
            prompt = f"""
            Context:
            {context_text}
            
            Question: {query}
            
            Please provide a precise and helpful answer based on the given context.
            If the context does not contain sufficient information,
            explain that you cannot find a specific answer in the provided document.
            """
            
            # Generate response with timeout and retry logic
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=settings.MAX_TOKENS,
                        temperature=settings.TEMPERATURE,
                        # Add any additional generation parameters here
                    )
                )
                
                return response.text
                
            except Exception as model_error:
                logger.error(f"Error generating response: {str(model_error)}")
                return "I apologize, but I encountered an error generating a response. Please try again or rephrase your question."
                
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")