from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import time
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .schemas import UploadResponse, DocumentMetadata
from .model import ModelHandler
from .config import settings
from .schemas import Query, Response

app = FastAPI()
model_handler = ModelHandler()

# CORS setup
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
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = f"temp_{int(time.time())}_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Read PDF and extract text
        pdf = PdfReader(file_path)
        text_content = ""
        for page in pdf.pages:
            text_content += page.extract_text() + "\n"
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        texts = text_splitter.split_text(text_content)
        
        # Create vector store
        model_handler.create_vector_store(texts)
        
        # Create metadata
        document_info = DocumentMetadata(
            filename=file.filename,
            page_count=len(pdf.pages),
            chunk_count=len(texts)
        )
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return UploadResponse(
            message="PDF processed successfully",
            document_info=document_info,
            success=True
        )
    
    except Exception as e:
        # Clean up temporary file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(query: Query):
    """Handle chat interactions"""
    if not model_handler.vector_store:
        raise HTTPException(
            status_code=400, 
            detail="Please upload a PDF first"
        )
    
    try:
        start_time = time.time()
        response_text = model_handler.generate_response(query.question)
        processing_time = time.time() - start_time
        
        return Response(
            response=response_text,
            confidence=0.8,  # This is a placeholder. You might want to calculate this
            context_used=model_handler.retrieve_context(query.question),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))