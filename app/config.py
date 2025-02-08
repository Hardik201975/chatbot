# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Memory-optimized application settings"""
   
    # Google Generative AI settings
    GOOGLE_API_KEY: str
    MODEL_NAME: str = "gemini-pro"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
   
    # Optimized document processing settings
    CHUNK_SIZE: int = 500  # Reduced from 1000
    CHUNK_OVERLAP: int = 50  # Reduced from 200
   
    # Memory-conscious model parameters
    MAX_TOKENS: int = 256  # Reduced from 512
    TEMPERATURE: float = 0.7
    
    # Memory management settings
    BATCH_SIZE: int = 16
    MAX_SEQ_LENGTH: int = 256
   
    class Config:
        extra = 'ignore'

settings = Settings()