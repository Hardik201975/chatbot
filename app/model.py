# # app/model.py
# import google.generativeai as genai
# import faiss
# import torch
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List, Optional, Dict
# import logging
# import os
# import gc
# import psutil
# from threading import Lock
# from .config import settings

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# class MemoryManager:
#     @staticmethod
#     def log_memory_usage():
#         process = psutil.Process(os.getpid())
#         memory_info = process.memory_info()
#         logger.info(f"Memory Usage - RSS: {memory_info.rss / 1024 / 1024:.2f}MB, VMS: {memory_info.vms / 1024 / 1024:.2f}MB")

#     @staticmethod
#     def clear_memory():
#         gc.collect()
#         if hasattr(torch, 'cuda'):
#             torch.cuda.empty_cache()

# class ModelHandler:
#     _instance = None
#     _lock = Lock()
    
#     def __new__(cls):
#         with cls._lock:
#             if cls._instance is None:
#                 cls._instance = super(ModelHandler, cls).__new__(cls)
#                 cls._instance._initialized = False
#             return cls._instance

#     def __init__(self):
#         if self._initialized:
#             return
            
#         try:
#             logger.info("Initializing ModelHandler with memory optimization...")
#             MemoryManager.log_memory_usage()
            
#             # Configure environment
#             os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface'
#             os.environ['HF_HOME'] = '/tmp/huggingface'
#             os.environ['TORCH_HOME'] = '/tmp/torch'
#             os.environ['XDG_CACHE_HOME'] = '/tmp/cache'
            
#             # Initialize with lazy loading
#             self._setup_google_ai()
#             self._initialize_base_components()
            
#             self._initialized = True
#             logger.info("ModelHandler initialization complete")
#             MemoryManager.log_memory_usage()
            
#         except Exception as e:
#             logger.error(f"ModelHandler initialization failed: {str(e)}", exc_info=True)
#             raise

#     def _setup_google_ai(self):
#         try:
#             genai.configure(api_key=settings.GOOGLE_API_KEY)
#             self.model = genai.GenerativeModel(settings.MODEL_NAME)
#         except Exception as e:
#             logger.error(f"Google AI setup failed: {str(e)}")
#             raise RuntimeError(f"Google AI configuration failed: {str(e)}")

#     def _initialize_base_components(self):
#         self.embeddings_model = None  # Lazy load
#         self.vector_store = None
#         self.document_texts = []
#         self._embeddings_lock = Lock()

#     def _load_embeddings_model(self):
#         if self.embeddings_model is None:
#             with self._embeddings_lock:
#                 if self.embeddings_model is None:  # Double-check
#                     try:
#                         logger.info("Loading embeddings model...")
#                         MemoryManager.log_memory_usage()
                        
#                         self.embeddings_model = SentenceTransformer(
#                             settings.EMBEDDING_MODEL,
#                             device='cpu',
#                             cache_folder='/tmp/sentence-transformers'
#                         )
#                         self.embeddings_model.max_seq_length = settings.MAX_SEQ_LENGTH
                        
#                         logger.info("Embeddings model loaded successfully")
#                         MemoryManager.log_memory_usage()
#                     except Exception as e:
#                         logger.error(f"Failed to load embeddings model: {str(e)}")
#                         raise

#     def embed_text(self, text: str) -> np.ndarray:
#         try:
#             self._load_embeddings_model()  # Lazy load
            
#             # Truncate long texts
#             max_chars = settings.MAX_SEQ_LENGTH * 4  # Approximate char limit
#             if len(text) > max_chars:
#                 text = text[:max_chars]
            
#             embedding = self.embeddings_model.encode(
#                 text,
#                 convert_to_numpy=True,
#                 show_progress_bar=False,
#                 normalize_embeddings=True  # Adds stability
#             )
#             return embedding.astype('float32')
#         except Exception as e:
#             logger.error(f"Embedding generation failed: {str(e)}")
#             raise
#         finally:
#             MemoryManager.clear_memory()

#     def create_vector_store(self, texts: List[str]):
#         if not texts:
#             raise ValueError("No texts provided for vector store creation")
            
#         try:
#             logger.info(f"Creating vector store with {len(texts)} documents...")
#             MemoryManager.log_memory_usage()
            
#             vector_list = []
#             batch_size = settings.BATCH_SIZE
            
#             for i in range(0, len(texts), batch_size):
#                 batch = texts[i:i + batch_size]
                
#                 # Process each text individually
#                 for text in batch:
#                     try:
#                         vector = self.embed_text(text)
#                         vector_list.append(vector)
#                     except Exception as e:
#                         logger.error(f"Failed to process text chunk {i}: {str(e)}")
#                         continue
                
#                 MemoryManager.clear_memory()
#                 logger.debug(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

#             if not vector_list:
#                 raise RuntimeError("No vectors were successfully created")

#             vectors = np.array(vector_list, dtype='float32')
            
#             # Create and populate index
#             dimension = vectors.shape[1]
#             self.vector_store = faiss.IndexFlatL2(dimension)
            
#             # Add vectors in smaller batches
#             sub_batch_size = min(100, len(vectors))
#             for i in range(0, len(vectors), sub_batch_size):
#                 self.vector_store.add(vectors[i:i + sub_batch_size])
#                 MemoryManager.clear_memory()
            
#             self.document_texts = texts
            
#             logger.info("Vector store creation complete")
#             MemoryManager.log_memory_usage()
#             return self
            
#         except Exception as e:
#             logger.error(f"Vector store creation failed: {str(e)}", exc_info=True)
#             raise
#         finally:
#             MemoryManager.clear_memory()

#     def retrieve_context(self, query: str, top_k: int = 3) -> List[str]:
#         """Retrieve most relevant document chunks"""
#         if not self.vector_store:
#             logger.warning("No vector store available for context retrieval")
#             return []
            
#         try:
#             # Generate query embedding
#             query_vector = np.array([self.embed_text(query)]).astype('float32')
            
#             # Search in vector store
#             D, I = self.vector_store.search(query_vector, min(top_k, len(self.document_texts)))
            
#             # Retrieve corresponding texts
#             context = [self.document_texts[i] for i in I[0]]
#             logger.debug(f"Retrieved {len(context)} context chunks")
#             return context
            
#         except Exception as e:
#             logger.error(f"Error retrieving context: {str(e)}")
#             return []


#     def generate_response(self, query: str) -> str:
#         """Generate response using context-aware prompt"""
#         try:
#             context = self.retrieve_context(query)
#             context_text = ' '.join(context) if context else "No relevant context found."
            
#             prompt = f"""
#             Context:
#             {context_text}
            
#             Question: {query}
            
#             Please provide a precise and helpful answer based on the given context.
#             If the context does not contain sufficient information,
#             explain that you cannot find a specific answer in the provided document.
#             """
            
#             response = self.model.generate_content(
#                 prompt,
#                 generation_config=genai.types.GenerationConfig(
#                     max_output_tokens=settings.MAX_TOKENS,
#                     temperature=settings.TEMPERATURE
#                 )
#             )
            
#             return response.text
            
#         except Exception as e:
#             logger.error(f"Error generating response: {str(e)}")
#             return "I apologize, but I encountered an error generating a response. Please try again."


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
        self.document_metadata = []  # New: Store metadata for each chunk
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

    def _chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks for better context preservation"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _extract_section_info(self, text: str) -> Dict[str, str]:
        """Extract section headers and other metadata from text"""
        metadata = {
            'section': '',
            'subsection': '',
            'keywords': []
        }
        
        # Simple header detection
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.isupper() or line.endswith(':-'):
                metadata['section'] = line
                break
            
        # Extract potential keywords
        important_words = [word for word in text.split() 
                         if word.isupper() or word.startswith('₹') 
                         or any(char.isdigit() for char in word)]
        metadata['keywords'] = important_words[:5]  # Store up to 5 keywords
        
        return metadata

    def create_vector_store(self, texts: List[str]):
        if not texts:
            raise ValueError("No texts provided for vector store creation")
            
        try:
            logger.info(f"Creating vector store with {len(texts)} documents...")
            MemoryManager.log_memory_usage()
            
            processed_chunks = []
            vector_list = []
            metadata_list = []
            
            for text in texts:
                # Create overlapping chunks
                chunks = self._chunk_text(text)
                
                for chunk in chunks:
                    try:
                        # Extract metadata for the chunk
                        metadata = self._extract_section_info(chunk)
                        vector = self.embed_text(chunk)
                        
                        processed_chunks.append(chunk)
                        vector_list.append(vector)
                        metadata_list.append(metadata)
                        
                    except Exception as e:
                        logger.error(f"Failed to process chunk: {str(e)}")
                        continue
                    
                MemoryManager.clear_memory()

            if not vector_list:
                raise RuntimeError("No vectors were successfully created")

            vectors = np.array(vector_list, dtype='float32')
            
            # Create and populate index
            dimension = vectors.shape[1]
            self.vector_store = faiss.IndexFlatL2(dimension)
            self.vector_store.add(vectors)
            
            self.document_texts = processed_chunks
            self.document_metadata = metadata_list
            
            logger.info("Vector store creation complete")
            MemoryManager.log_memory_usage()
            return self
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {str(e)}", exc_info=True)
            raise
        finally:
            MemoryManager.clear_memory()

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """Retrieve most relevant document chunks with their metadata"""
        if not self.vector_store:
            logger.warning("No vector store available for context retrieval")
            return []
            
        try:
            query_vector = np.array([self.embed_text(query)]).astype('float32')
            D, I = self.vector_store.search(query_vector, min(top_k, len(self.document_texts)))
            
            # Return chunks with their metadata
            context = []
            for idx in I[0]:
                context.append({
                    'text': self.document_texts[idx],
                    'metadata': self.document_metadata[idx],
                    'relevance_score': float(D[0][idx])
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []

    def generate_response(self, query: str) -> str:
        """Generate response using enhanced context-aware prompt"""
        try:
            context_items = self.retrieve_context(query)
            
            # Sort contexts by relevance and section importance
            context_items.sort(key=lambda x: (
                x['relevance_score'],
                1 if any(kw in query.upper() for kw in x['metadata']['keywords']) else 0
            ))
            
            # Prepare context with metadata
            formatted_context = []
            for item in context_items:
                section_info = f"[{item['metadata']['section']}]" if item['metadata']['section'] else ""
                formatted_context.append(f"{section_info}\n{item['text']}")
            
            context_text = '\n\n'.join(formatted_context)
            
            prompt = f"""
            Context Information:
            {context_text}
            
            Question: {query}
            
            Instructions:
            1. Answer the question specifically using information from the provided context
            2. If the answer involves numbers, prizes, or specific details, make sure to include them
            3. If multiple sections are relevant, combine the information coherently
            4. If the context doesn't contain enough information to answer the question fully, 
               clearly state what information is missing
            
            Please provide a detailed answer:
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=settings.MAX_TOKENS,
                    temperature=0.3  # Lower temperature for more focused responses
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error generating a response. Please try again."