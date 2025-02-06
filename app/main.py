from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from .schemas import Query, Response, UploadResponse
from .model import ModelHandler
from .config import settings

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
        # Create a unique filename to avoid conflicts
        file_path = f"temp_{os.urandom(6).hex()}_{file.filename}"
        
        # Save uploaded file temporarily
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        
        # Initialize model and create embeddings
        _, embeddings = model_handler.initialize_model()
        
        # Add error handling for vector store creation
        if not texts:
            raise ValueError("No text extracted from PDF")
            
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Set up conversation chain
        model_handler.setup_conversation_chain(vector_store)
        
        return UploadResponse(message="PDF processed successfully")
    
    except Exception as e:
        # Ensure we always clean up the file
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    finally:
        # Clean up in success case too
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