"""
Smart Form Fill API - Core form analysis endpoints
Note: Vector database management has been moved to main.py
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
from pydantic import BaseModel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Pydantic models
class FormAnalysisRequest(BaseModel):
    url: str

class FormAnalysisResponse(BaseModel):
    status: str
    message: str
    url: str
    analysis: Dict[str, Any] = None

class HealthResponse(BaseModel):
    status: str
    message: str = "API is running"

# Initialize FastAPI
app = FastAPI(
    title="Smart Form Fill Core API",
    description="Core form analysis functionality",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "message": "Smart Form Fill Core API is running"
    }

@app.post("/api/analyze-form", response_model=FormAnalysisResponse)
async def analyze_form(request: FormAnalysisRequest) -> Dict[str, Any]:
    """
    Basic form analysis endpoint
    Note: For vector database operations, use the main API at main.py
    """
    url = str(request.url)
    
    try:
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL format. URL must start with http:// or https://"
            )
        
        # Placeholder for form analysis
        # In a real implementation, this would analyze the form structure
        analysis_result = {
            "status": "success",
            "message": "Form analysis completed",
            "url": url,
            "analysis": {
                "form_detected": True,
                "field_count": 0,
                "form_type": "unknown",
                "note": "This is a placeholder. For full functionality, use the vector database API in main.py"
            }
        }
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Form analysis failed: {str(e)}"
        )

@app.get("/api/info")
async def get_api_info():
    """
    Get information about available APIs
    """
    return {
        "message": "Smart Form Fill API System",
        "apis": {
            "core_api": {
                "description": "Basic form analysis (this API)",
                "endpoints": ["/api/health", "/api/analyze-form", "/api/info"]
            },
            "vector_database_api": {
                "description": "Vector database management (main.py)",
                "base_url": "http://localhost:8000",
                "endpoints": [
                    "/api/v1/resume/reembed",
                    "/api/v1/personal-info/reembed",
                    "/api/v1/resume/search",
                    "/api/v1/personal-info/search",
                    "/api/v1/reembed-all"
                ]
            }
        },
        "note": "For vector database operations (re-embedding, searching), use the main API server"
    } 