from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from .config import settings

class ModelHandler:
    """Handles all model-related operations"""
    
    def __init__(self):
        self.vector_store = None
        self.conversation_chain = None
        
    def initialize_model(self):
        """Initialize the Llama 2 model and embeddings"""
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
        
        return pipe, embeddings
    
    def setup_conversation_chain(self, vector_store):
        """Set up the conversation chain with memory"""
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
