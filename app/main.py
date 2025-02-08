# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import time
import logging
from typing import Optional
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .schemas import (
    UploadResponse, 
    DocumentMetadata, 
    Query, 
    Response, 
    ErrorResponse
)
from .model import ModelHandler
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Chat API",
    description="API for processing PDFs and answering questions about their content",
    version="1.0.0"
)

# Initialize model handler
model_handler = ModelHandler()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            detail=str(exc)
        ).dict()
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """
    Handle PDF upload and processing
    """
    start_time = time.time()
    
    try:
        # Read file content
        content = await file.read()
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are allowed"
            )

        # Process PDF in memory
        pdf_stream = io.BytesIO(content)
        
        try:
            pdf = PdfReader(pdf_stream)
            if len(pdf.pages) == 0:
                raise ValueError("PDF file appears to be empty")
            
            text_content = ""
            for page_num, page in enumerate(pdf.pages, 1):
                logger.debug(f"Processing page {page_num}")
                page_text = page.extract_text()
                if not page_text:
                    logger.warning(f"Page {page_num} appears to be empty or unreadable")
                text_content += page_text + "\n"

            if not text_content.strip():
                raise ValueError("No readable text content found in PDF")

        except Exception as pdf_error:
            logger.error(f"PDF processing error: {str(pdf_error)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error processing PDF: {str(pdf_error)}"
            )

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_text(text_content)

        if not texts:
            raise ValueError("Text splitting produced no chunks")

        # Create vector store
        try:
            model_handler.create_vector_store(texts)
        except Exception as vector_error:
            logger.error(f"Vector store creation error: {str(vector_error)}")
            raise HTTPException(
                status_code=500,
                detail="Error creating vector store for document"
            )

        # Create metadata
        document_info = DocumentMetadata(
            filename=file.filename,
            page_count=len(pdf.pages),
            chunk_count=len(texts)
        )

        processing_time = time.time() - start_time
        logger.info(f"PDF processed successfully in {processing_time:.2f} seconds")

        return UploadResponse(
            message="PDF processed successfully",
            document_info=document_info,
            success=True
        )

    except HTTPException as http_error:
        raise http_error

    except Exception as e:
        logger.error(f"Unexpected error processing upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clear the BytesIO buffer
        if 'pdf_stream' in locals():
            pdf_stream.close()

@app.post("/chat", response_model=Response)
async def chat(query: Query) -> Response:
    """
    Handle chat interactions with the uploaded document
    """
    start_time = time.time()
    
    try:
        if not model_handler.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Please upload a PDF first"
            )

        # Generate response
        response_text = model_handler.generate_response(query.question)
        
        # Get context used
        context = model_handler.retrieve_context(query.question)
        
        # Calculate processing time
        processing_time = time.time() - start_time

        return Response(
            response=response_text,
            confidence=0.8,  # This could be improved with actual confidence calculation
            context_used=context,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(settings.PORT if hasattr(settings, 'PORT') else 8000),
        reload=False  # Disable reload on production
    )