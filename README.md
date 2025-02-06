# Llama2 PDF Chatbot

A cloud-based chatbot that uses Llama 2 to answer questions about uploaded PDF documents. This project uses Hugging Face's hosted models and can be deployed directly to Render without requiring local GPU resources.

## Features
- PDF document processing
- Question answering using Llama 2
- Cloud-based deployment
- Easy integration with frontend applications

## Setup
1. Clone this repository
2. Set up environment variables
3. Deploy to Render
4. Connect your frontend

## Environment Variables
- HUGGINGFACE_TOKEN: Your Hugging Face API token
- PORT: Server port (default: 8000)

## API Endpoints
- POST /upload: Upload PDF document
- POST /chat: Ask questions about the uploaded document