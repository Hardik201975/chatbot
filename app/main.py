from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    file_path = f"temp_{file.filename}"
    try:
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
        
        # Extract text content
        text_contents = [doc.page_content for doc in texts]
        
        # Create vector store
        model_handler.create_vector_store(text_contents)
        
        # Clean up
        os.remove(file_path)
        
        return UploadResponse(message="PDF processed successfully")
    
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=Response)
async def chat(query: Query):
    """Handle chat interactions"""
    try:
        response = model_handler.generate_response(query.question)
        return Response(response=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))