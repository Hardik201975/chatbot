from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Google Generative AI settings
    GOOGLE_API_KEY: str
    MODEL_NAME: str = "gemini-pro"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Model parameters
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    
    class Config:
        env_file = ".env"

settings = Settings()