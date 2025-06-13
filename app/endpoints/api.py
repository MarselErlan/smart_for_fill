"""
Smart Form Fill API - Simplified endpoints for basic form operations
Note: Complex form analysis has been moved to vector database approach in main.py
"""

import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List
from loguru import logger
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize router
router = APIRouter()

# Simplified Request/Response Models
class FormAnalysisRequest(BaseModel):
    url: HttpUrl
    force_refresh: bool = False

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class DocumentStatusResponse(BaseModel):
    status: str
    documents: Dict[str, Any]
    message: str

# Get environment variables with defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for the form analysis API
    """
    return {
        "status": "healthy",
        "message": "Smart Form Fill API endpoints are running",
        "timestamp": datetime.now().isoformat(),
        "note": "For vector database operations, use the main API at main.py"
    }

@router.get("/info")
async def get_api_info() -> Dict[str, Any]:
    """
    Get information about the API system
    """
    return {
        "message": "Smart Form Fill API System",
        "version": "2.0.0",
        "apis": {
            "endpoints_api": {
                "description": "Basic form endpoints (this API)",
                "base_path": "/api",
                "endpoints": ["/health", "/info", "/analyze-form"]
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
        "note": "For full functionality including vector database operations, use the main API server"
    }

@router.post("/analyze-form")
async def analyze_form(request: FormAnalysisRequest) -> Dict[str, Any]:
    """
    Basic form analysis endpoint (simplified version)
    For full form analysis with vector database integration, use main.py API
    """
    try:
        url = str(request.url)
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=400,
                detail="Invalid URL format. URL must start with http:// or https://"
            )
        
        # Simplified analysis response
        return {
            "status": "success",
            "message": "Basic form analysis completed",
            "url": url,
            "analysis": {
                "url_valid": True,
                "timestamp": datetime.now().isoformat(),
                "note": "This is a simplified analysis. For full functionality with vector database integration, use the main API server at main.py"
            },
            "recommendation": "Use the vector database API for complete form filling with resume and personal info integration"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Form analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Form analysis failed: {str(e)}"
        )

@router.get("/documents/status")
async def get_document_status() -> Dict[str, Any]:
    """
    Check the status of user document files
    """
    try:
        from pathlib import Path
        
        # Check for document files
        docs_resume = Path("docs/resume")
        docs_info = Path("docs/info")
        
        resume_files = list(docs_resume.glob("*.docx")) if docs_resume.exists() else []
        info_files = list(docs_info.glob("*.txt")) if docs_info.exists() else []
        
        status = {
            "resume_directory": {
                "exists": docs_resume.exists(),
                "path": str(docs_resume),
                "files": [str(f.name) for f in resume_files]
            },
            "info_directory": {
                "exists": docs_info.exists(),
                "path": str(docs_info),
                "files": [str(f.name) for f in info_files]
            },
            "vector_databases": {
                "resume_vectordb": Path("resume/vectordb").exists(),
                "info_vectordb": Path("info/vectordb").exists()
            }
        }
        
        return {
            "status": "success",
            "documents": status,
            "message": "Document status retrieved successfully",
            "note": "Use the vector database API for re-embedding and searching documents"
        }
        
    except Exception as e:
        logger.error(f"Error checking document status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/urls")
async def get_urls() -> Dict[str, Any]:
    """
    Get information about available URLs and endpoints
    """
    return {
        "status": "success",
        "message": "Available API endpoints",
        "endpoints": {
            "health": "/api/health",
            "info": "/api/info", 
            "analyze_form": "/api/analyze-form",
            "document_status": "/api/documents/status"
        },
        "vector_database_api": {
            "base_url": "http://localhost:8000",
            "main_endpoints": [
                "/api/v1/resume/reembed",
                "/api/v1/personal-info/reembed",
                "/api/v1/reembed-all"
            ]
        }
    } 
    