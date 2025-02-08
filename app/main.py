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
import psutil
import os

from .schemas import (
    UploadResponse, 
    DocumentMetadata, 
    Query, 
    Response, 
    ErrorResponse
)
from .model import ModelHandler, MemoryManager
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF Chat API",
    description="Memory-optimized API for PDF processing and QA",
    version="1.0.0"
)

# Initialize model handler
model_handler = None  # Lazy initialization

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_model_handler():
    global model_handler
    if model_handler is None:
        try:
            model_handler = ModelHandler()
        except Exception as e:
            logger.error(f"Failed to initialize ModelHandler: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize model handler"
            )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error handler caught: {str(exc)}", exc_info=True)
    MemoryManager.log_memory_usage()
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
    """Enhanced health check endpoint"""
    memory_info = psutil.Process(os.getpid()).memory_info()
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "memory_usage_mb": memory_info.rss / 1024 / 1024
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Memory-optimized PDF upload handler"""
    start_time = time.time()
    pdf_stream = None
    
    try:
        MemoryManager.log_memory_usage()
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        # Read file in chunks
        content = await file.read()
        pdf_stream = io.BytesIO(content)
        
        # Process PDF
        try:
            pdf = PdfReader(pdf_stream)
            if len(pdf.pages) == 0:
                raise ValueError("PDF file appears to be empty")
            
            # Extract text with memory optimization
            text_content = []
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                    MemoryManager.clear_memory()
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num}: {str(e)}")
            
            if not text_content:
                raise ValueError("No readable text content found in PDF")
            
            # Join text content
            full_text = "\n".join(text_content)
            del text_content  # Free memory
            MemoryManager.clear_memory()

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
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(full_text)
        del full_text  # Free memory
        MemoryManager.clear_memory()

        # Initialize model handler if needed
        init_model_handler()

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
        MemoryManager.log_memory_usage()

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
        if pdf_stream:
            pdf_stream.close()
        MemoryManager.clear_memory()

@app.post("/chat", response_model=Response)
async def chat(query: Query) -> Response:
    """Memory-optimized chat endpoint"""
    start_time = time.time()
    
    try:
        init_model_handler()
        
        if not model_handler.vector_store:
            raise HTTPException(
                status_code=400,
                detail="Please upload a PDF first"
            )

        MemoryManager.log_memory_usage()
        response_text = model_handler.generate_response(query.question)
        context = model_handler.retrieve_context(query.question)
        
        processing_time = time.time() - start_time
        
        return Response(
            response=response_text,
            confidence=0.8,
            context_used=context,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )
    finally:
        MemoryManager.clear_memory()