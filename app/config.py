from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # Hugging Face model settings
    MODEL_NAME: str = "meta-llama/Llama-2-7b-chat-hf"
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Model parameters
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    REPETITION_PENALTY: float = 1.15
    
    # Document processing settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # API settings
    HUGGINGFACE_TOKEN: str
    
    class Config:
        env_file = ".env"

settings = Settings()