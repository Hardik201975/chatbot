from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class Query(BaseModel):
    """
    Schema for incoming chat queries.
    
    Attributes:
        question (str): The user's question about the PDF content
        context (Optional[str]): Optional additional context for the question
    """
    question: str = Field(
        ...,  # ... means required
        min_length=1,
        max_length=1000,
        description="The question to ask about the PDF content"
    )
    context: Optional[str] = Field(
        None,
        max_length=2000,
        description="Additional context for the question"
    )

    @validator('question')
    def question_not_empty(cls, v):
        """Validate that question is not just whitespace"""
        if v.strip() == "":
            raise ValueError("Question cannot be empty or just whitespace")
        return v.strip()

class Response(BaseModel):
    """
    Schema for chat responses.
    
    Attributes:
        response (str): The model's answer to the question
        confidence (float): Confidence score of the answer
        context_used (List[str]): Relevant PDF sections used for the answer
        processing_time (float): Time taken to generate the response
    """
    response: str = Field(
        ...,
        description="The model's answer to the question"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    context_used: List[str] = Field(
        default_factory=list,
        description="Relevant sections from the PDF used to generate the answer"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Time taken to generate response in seconds"
    )

class DocumentMetadata(BaseModel):
    """
    Schema for PDF document metadata.
    
    Attributes:
        filename (str): Name of the uploaded PDF file
        page_count (int): Number of pages in the PDF
        upload_timestamp (datetime): When the document was uploaded
        chunk_count (int): Number of text chunks created
    """
    filename: str = Field(
        ...,
        description="Original filename of the PDF"
    )
    page_count: int = Field(
        ...,
        gt=0,
        description="Number of pages in the PDF"
    )
    upload_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the document was uploaded"
    )
    chunk_count: int = Field(
        ...,
        gt=0,
        description="Number of text chunks created from the PDF"
    )

class UploadResponse(BaseModel):
    """
    Schema for PDF upload responses.
    
    Attributes:
        message (str): Success/error message
        document_info (DocumentMetadata): Metadata about the processed document
        success (bool): Whether the upload was successful
    """
    message: str = Field(
        ...,
        description="Status message about the upload"
    )
    document_info: Optional[DocumentMetadata] = Field(
        None,
        description="Metadata about the processed document"
    )
    success: bool = Field(
        ...,
        description="Whether the upload was successful"
    )

class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    
    Attributes:
        error (str): Error message
        error_code (str): Specific error code
        timestamp (datetime): When the error occurred
    """
    error: str = Field(
        ...,
        description="Error message"
    )
    error_code: str = Field(
        ...,
        description="Specific error code for this type of error"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the error occurred"
    )

class ChatHistory(BaseModel):
    """
    Schema for chat history entries.
    
    Attributes:
        question (str): User's question
        response (str): Model's response
        timestamp (datetime): When the exchange occurred
    """
    question: str = Field(
        ...,
        description="User's question"
    )
    response: str = Field(
        ...,
        description="Model's response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this chat exchange occurred"
    )

class ChatSession(BaseModel):
    """
    Schema for complete chat sessions.
    
    Attributes:
        session_id (str): Unique identifier for the chat session
        document_info (DocumentMetadata): Information about the PDF being discussed
        chat_history (List[ChatHistory]): List of all Q&A exchanges
        created_at (datetime): When the session was created
    """
    session_id: str = Field(
        ...,
        description="Unique identifier for this chat session"
    )
    document_info: DocumentMetadata
    chat_history: List[ChatHistory] = Field(
        default_factory=list,
        description="List of all question and answer exchanges"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this chat session was created"
    )