import os
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from datetime import datetime

# Import your local modules
from .schemas import Query, Response, UploadResponse, DocumentMetadata
from .model import ModelHandler
from .config import settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()
model_handler = ModelHandler()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload and processing"""
    file_path = None
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("./temp", exist_ok=True)
        
        # Create a unique filename
        file_path = f"./temp/temp_{os.urandom(6).hex()}_{file.filename}"
        
        logger.info(f"Starting file upload: {file.filename}")
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Process PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        logger.info(f"Loaded {len(documents)} documents from PDF")
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"Split documents into {len(texts)} chunks")
        
        # Initialize model and create embeddings
        _, embeddings = model_handler.initialize_model()
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        logger.info("Vector store created successfully")
        
        # Set up conversation chain
        model_handler.setup_conversation_chain(vector_store)
        
        # Create document metadata
        document_info = DocumentMetadata(
            filename=file.filename,
            page_count=len(documents),
            upload_timestamp=datetime.now(),
            chunk_count=len(texts)
        )
        
        return UploadResponse(
            message="PDF processed successfully", 
            document_info=document_info,
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        
        # Ensure file is cleaned up
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    finally:
        # Clean up file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat", response_model=Response)
async def chat(query: Query):
    """Handle chat interactions"""
    if not model_handler.conversation_chain:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first"
        )
    
    try:
        response = model_handler.conversation_chain({
            "question": query.question
        })
        return Response(response=response["answer"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))