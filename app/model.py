# app/model.py
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
import os
import gc
from .config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHandler:
    """Memory-optimized model handler for Render deployment"""
    
    def __init__(self):
        try:
            logger.info("Starting ModelHandler initialization...")
            
            # Configure Google Generative AI
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.MODEL_NAME)
            logger.info("Google AI configured successfully")
            
            # Set smaller cache size for HuggingFace
            os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
            os.environ['HF_HOME'] = '/tmp/huggingface'
            
            # Initialize embeddings with smaller max_seq_length
            logger.info("Initializing embeddings model...")
            self.embeddings_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device='cpu',  # Force CPU usage
                cache_folder='/tmp/sentence-transformers'
            )
            self.embeddings_model.max_seq_length = 256  # Reduce max sequence length
            logger.info("Embeddings model initialized successfully")
            
            # Initialize vector store and document storage
            self.vector_store: Optional[faiss.IndexFlatL2] = None
            self.document_texts: List[str] = []
            
            logger.info("ModelHandler initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ModelHandler: {str(e)}")
            raise RuntimeError(f"Failed to initialize ModelHandler: {str(e)}")

    def embed_text(self, text: str) -> np.ndarray:
        """Memory-efficient embedding generation"""
        try:
            embedding = self.embeddings_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.astype('float32')  # Use float32 instead of float64
        finally:
            gc.collect()  # Force garbage collection after embedding

    def create_vector_store(self, texts: List[str]):
        """Memory-optimized vector store creation"""
        if not texts:
            raise ValueError("No texts provided for vector store creation")
            
        try:
            # Use smaller batch size
            batch_size = 16  # Reduced from 32
            vector_list = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_vectors = []
                
                # Process each text individually to minimize memory usage
                for text in batch:
                    vector = self.embed_text(text)
                    batch_vectors.append(vector)
                    
                vector_list.extend(batch_vectors)
                
                # Force garbage collection after each batch
                gc.collect()
                logger.debug(f"Processed batch {i//batch_size + 1}")

            vectors = np.array(vector_list, dtype='float32')
            
            # Create memory-efficient FAISS index
            dimension = vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors in smaller batches
            sub_batch_size = 100
            for i in range(0, len(vectors), sub_batch_size):
                index.add(vectors[i:i + sub_batch_size])
                gc.collect()
            
            # Update instance variables
            self.vector_store = index
            self.document_texts = texts
            
            logger.info(f"Vector store created successfully with {len(texts)} documents")
            return self
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise RuntimeError(f"Failed to create vector store: {str(e)}")
        finally:
            # Clean up temporary variables
            del vector_list
            gc.collect()