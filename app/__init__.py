# app/__init__.py
from .config import settings
from .model import ModelHandler
from .schemas import (
    Query, 
    Response, 
    DocumentMetadata, 
    UploadResponse, 
    ErrorResponse, 
    ChatHistory, 
    ChatSession
)

__all__ = [
    'settings',
    'ModelHandler',
    'Query',
    'Response',
    'DocumentMetadata',
    'UploadResponse',
    'ErrorResponse',
    'ChatHistory',
    'ChatSession'
]