import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pydantic_settings import BaseSettings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class ModelHandler:
    """Handles all model-related operations"""
    
    def __init__(self):
        self.vector_store = None
        self.conversation_chain = None
    
    def initialize_model(self):
        """Initialize the Llama 2 model and embeddings"""
        try:
            # Initialize tokenizer with Hugging Face token
            tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                token=settings.HUGGINGFACE_TOKEN
            )
            
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(
                settings.MODEL_NAME,
                token=settings.HUGGINGFACE_TOKEN,
                device_map="auto"
            )
            
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=settings.MAX_NEW_TOKENS,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                repetition_penalty=settings.REPETITION_PENALTY
            )
            
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            )
            
            logger.info("Model and embeddings initialized successfully")
            return pipe, embeddings
        
        except Exception as e:
            logger.error(f"Model initialization error: {e}", exc_info=True)
            raise
    
    def setup_conversation_chain(self, vector_store):
        """Set up the conversation chain with memory"""
        try:
            llm, _ = self.initialize_model()
            
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=memory,
                verbose=True
            )
            
            logger.info("Conversation chain setup complete")
        except Exception as e:
            logger.error(f"Conversation chain setup error: {e}", exc_info=True)
            raise