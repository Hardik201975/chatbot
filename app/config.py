# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings with smaller model configuration"""
   
    # Google Generative AI settings
    GOOGLE_API_KEY: str
    MODEL_NAME: str = "gemini-pro"
    
    # Changed to a smaller, more efficient model
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"
   
    # Optimized document processing settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
   
    # Memory-conscious model parameters
    MAX_TOKENS: int = 256
    TEMPERATURE: float = 0.7
    
    # Memory management settings
    BATCH_SIZE: int = 16
    MAX_SEQ_LENGTH: int = 128  # Reduced from 256
   
    class Config:
        extra = 'ignore'

settings = Settings()