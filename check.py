from fastapi import FastAPI
import os
import requests

app = FastAPI()

@app.get("/check-google-api")
def check_google_api():
    API_KEY = 
    if not API_KEY:
        return {"error": "GOOGLE_API_KEY is not set"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateText?key={API_KEY}"
    
    data = {"prompt": {"text": "Hello, AI!"}, "temperature": 0.7}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=data, headers=headers)
        return response.json()  # Check if it returns a valid response
    except Exception as e:
        return {"error": str(e)}
