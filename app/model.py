

import google.generativeai as genai
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
import logging
import os
import gc
import psutil
from threading import Lock
from .config import settings
import re
import inflect

p = inflect.engine()

def normalize_text(text: str) -> str:
    """Convert numbers to words and apply basic text normalization."""
    text = text.lower()  # Convert to lowercase
        
        # Convert ordinal numbers (1st -> first, 2nd -> second)
    text = re.sub(r'(\d+)(st|nd|rd|th)', lambda x: p.number_to_words(x.group(1)), text)
        
        # Replace common abbreviations or variations
    text = text.replace("1st", "first").replace("2nd", "second").replace("3rd", "third")
        
    return text



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MemoryManager:
    @staticmethod
    def log_memory_usage():
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory Usage - RSS: {memory_info.rss / 1024 / 1024:.2f}MB, VMS: {memory_info.vms / 1024 / 1024:.2f}MB")

    @staticmethod
    def clear_memory():
        gc.collect()
        if hasattr(torch, 'cuda'):
            torch.cuda.empty_cache()

class ModelHandler:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelHandler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        try:
            logger.info("Initializing ModelHandler with memory optimization...")
            MemoryManager.log_memory_usage()
            
            # Configure environment
            os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
            os.environ['HF_HOME'] = '/tmp/huggingface'
            os.environ['TORCH_HOME'] = '/tmp/torch'
            os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
            
            self._setup_google_ai()
            self._initialize_base_components()
            
            self._initialized = True
            logger.info("ModelHandler initialization complete")
            MemoryManager.log_memory_usage()
            
        except Exception as e:
            logger.error(f"ModelHandler initialization failed: {str(e)}", exc_info=True)
            raise

    def _setup_google_ai(self):
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.MODEL_NAME)
        except Exception as e:
            logger.error(f"Google AI setup failed: {str(e)}")
            raise RuntimeError(f"Google AI configuration failed: {str(e)}")

    def _initialize_base_components(self):
        self.embeddings_model = None
        self.vector_store = None
        self.document_texts = []
        self.section_info = {}  # Store section context without changing interface
        self._embeddings_lock = Lock()

    def _load_embeddings_model(self):
        if self.embeddings_model is None:
            with self._embeddings_lock:
                if self.embeddings_model is None:
                    try:
                        logger.info("Loading embeddings model...")
                        MemoryManager.log_memory_usage()
                        
                        self.embeddings_model = SentenceTransformer(
                            settings.EMBEDDING_MODEL,
                            device='cpu',
                            cache_folder='/tmp/sentence-transformers'
                        )
                        self.embeddings_model.max_seq_length = settings.MAX_SEQ_LENGTH
                        
                        logger.info("Embeddings model loaded successfully")
                        MemoryManager.log_memory_usage()
                    except Exception as e:
                        logger.error(f"Failed to load embeddings model: {str(e)}")
                        raise

    def _process_text_chunk(self, text: str) -> str:
        """Internal helper to process text chunks while preserving context"""
        # Extract any section headers (all caps or ending with ":-")
        lines = text.split('\n')
        section_header = None
        for line in lines:
            if line.strip().isupper() or line.strip().endswith(':-'):
                section_header = line.strip()
                break
        
        # Keep track of important information
        if section_header:
            self.section_info[len(self.document_texts)] = {
                'section': section_header,
                'has_numbers': any(c.isdigit() for c in text),
                'has_currency': '₹' in text or 'Rs.' in text
            }
        
        return text
    



    def embed_text(self, text: str) -> np.ndarray:
        try:
            self._load_embeddings_model()

            text = normalize_text(text)
            
            # Truncate long texts
            max_chars = settings.MAX_SEQ_LENGTH * 4
            if len(text) > max_chars:
                text = text[:max_chars]
            
            embedding = self.embeddings_model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
        finally:
            MemoryManager.clear_memory()

    def create_vector_store(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts provided for vector store creation")
            
        try:
            logger.info(f"Creating vector store with {len(texts)} documents...")
            MemoryManager.log_memory_usage()
            
            vector_list = []
            processed_texts = []
            batch_size = settings.BATCH_SIZE
            
            # Process texts in batches while preserving context
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                for text in batch:
                    # Split long texts into smaller chunks (200-300 words) with overlap
                    words = text.split()
                    chunk_size = 150
                    overlap = 50
                    
                    for j in range(0, len(words), chunk_size - overlap):
                        chunk = ' '.join(words[j:j + chunk_size])
                        processed_chunk = self._process_text_chunk(chunk)
                        
                        try:
                            vector = self.embed_text(processed_chunk)
                            vector_list.append(vector)
                            processed_texts.append(processed_chunk)
                        except Exception as e:
                            logger.error(f"Failed to process text chunk: {str(e)}")
                            continue
                
                MemoryManager.clear_memory()
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            if not vector_list:
                raise RuntimeError("No vectors were successfully created")

            vectors = np.array(vector_list, dtype='float32')
            
            dimension = vectors.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            
            sub_batch_size = min(100, len(vectors))
            for i in range(0, len(vectors), sub_batch_size):
                self.vector_store.add(vectors[i:i + sub_batch_size])
                MemoryManager.clear_memory()
            
            self.document_texts = processed_texts
            
            logger.info("Vector store creation complete")
            MemoryManager.log_memory_usage()
            return self
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {str(e)}", exc_info=True)
            raise
        finally:
            MemoryManager.clear_memory()

    def retrieve_context(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve most relevant document chunks"""
        if not self.vector_store:
            logger.warning("No vector store available for context retrieval")
            return []
            
        try:
            query = normalize_text(query)
            query_vector = np.array([self.embed_text(query)]).astype('float32')
            
            # Search in vector store
            D, I = self.vector_store.search(query_vector, min(top_k, len(self.document_texts)))
            
            # Get relevant sections and their contexts
            contexts = []
            for idx in I[0]:
                text = self.document_texts[idx]
                # Add section header if available
                if idx in self.section_info:
                    section = self.section_info[idx]['section']
                    text = f"{section}\n{text}" if section else text
                contexts.append(text)
            
            logger.debug(f"Retrieved {len(contexts)} context chunks")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def generate_response(self, query: str) -> str:
        """Generate response using context-aware prompt"""
        try:
            context = self.retrieve_context(query)
            context_text = '\n---\n'.join(context) if context else "No relevant context found."
            
            # Enhanced prompt structure while maintaining the same interface
            prompt = f"""
            Context Information:
            {context_text}
            
            Question: {query}
            
            Important Instructions:
            - Answer specifically using the provided context
            - Include any relevant numbers, amounts, or specific details
            - If the context mentions multiple related pieces of information, combine them
            - If some information is missing from the context, clearly state what's not available
            
            Please provide a precise and helpful answer based on the above context.
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.MAX_TOKENS,
                    temperature=0.3  # Lower temperature for more precise responses
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."