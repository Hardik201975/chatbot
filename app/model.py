import google.generativeai as genai
import faiss
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict

class ModelHandler:
    """Handles all model-related operations"""
    
    def __init__(self):
        # Configure Google Generative AI
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        self.model = genai.GenerativeModel(settings.MODEL_NAME)
        self.vector_store = None
        self.document_texts = []
    
    def initialize_embeddings(self):
        """Initialize embeddings using Sentence Transformers"""
        return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    
    def create_vector_store(self, texts: List[str]):
        """Create a FAISS vector store for document retrieval"""
        # Generate embeddings
        embeddings = self.initialize_embeddings()
        
        # Convert texts to embeddings
        vector_list = [embeddings.embed_query(text) for text in texts]
        vectors = np.array(vector_list).astype('float32')
        
        # Create FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        self.vector_store = index
        self.document_texts = texts
        return self
    
    def retrieve_context(self, query: str, top_k: int = 3):
        """Retrieve most relevant document chunks"""
        embeddings = self.initialize_embeddings()
        query_vector = np.array([embeddings.embed_query(query)]).astype('float32')
        
        if self.vector_store is None:
            return []
        
        # Search in vector store
        D, I = self.vector_store.search(query_vector, top_k)
        
        # Retrieve corresponding texts
        return [self.document_texts[i] for i in I[0]]
    
    def generate_response(self, query: str) -> str:
        """Generate response using context-aware prompt"""
        # Retrieve relevant context
        context = self.retrieve_context(query)
        
        # Construct prompt with context
        prompt = f"""
        Context: 
        {' '.join(context)}
        
        Question: {query}
        
        Please provide a precise and helpful answer based on the given context. 
        If the context does not contain sufficient information, 
        explain that you cannot find a specific answer in the provided document.
        """
        
        # Generate response
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE
            )
        )
        
        return response.text