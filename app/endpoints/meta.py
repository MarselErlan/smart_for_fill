"""
Smart Form Fill API - Meta endpoints for health, status, and API info.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: str

class DocumentStatusResponse(BaseModel):
    status: str
    documents: Dict[str, Any]
    message: str

@router.get("/health", tags=["Meta"])
async def health_check() -> Dict[str, Any]:
    """
    Health check for the entire API.
    """
    return {
        "status": "healthy",
        "message": "Smart Form Fill API is running",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/info", tags=["Meta"])
async def get_api_info() -> Dict[str, Any]:
    """
    Get information about the API system and its components.
    """
    return {
        "message": "Smart Form Fill API System",
        "version": "2.0.0",
        "apis": {
            "meta_api": {
                "description": "API health and information",
                "base_path": "/",
                "endpoints": ["/health", "/info", "/documents/status"]
            },
            "form_operations_api": {
                "description": "Endpoints for form analysis and filling",
                "base_path": "/api",
                "endpoints": ["/run-pipeline", "/analyze-form", "/fields", "/autofill-preview", "/fill-form-in-browser"]
            },
            "vector_database_api": {
                "description": "Vector database management",
                "base_path": "/api/v1",
                "endpoints": [
                    "/resume/status", "/personal-info/status",
                    "/resume/reembed", "/personal-info/reembed",
                    "/resume/search", "/personal-info/search",
                    "/reembed-all"
                ]
            }
        }
    }

@router.get("/documents/status", response_model=DocumentStatusResponse, tags=["Meta"])
async def get_document_status() -> Dict[str, Any]:
    """
    Check the status and existence of user document files and vector databases.
    """
    try:
        docs_resume = Path("docs/resume")
        docs_info = Path("docs/info")
        
        resume_files = [f.name for f in docs_resume.glob("*.docx") if f.is_file()] if docs_resume.exists() else []
        info_files = [f.name for f in docs_info.glob("*.txt") if f.is_file()] if docs_info.exists() else []
        
        status = {
            "resume_directory": {
                "exists": docs_resume.exists(),
                "path": str(docs_resume),
                "files": resume_files
            },
            "info_directory": {
                "exists": docs_info.exists(),
                "path": str(docs_info),
                "files": info_files
            },
            "vector_databases": {
                "resume_vectordb_exists": Path("resume/vectordb").exists(),
                "info_vectordb_exists": Path("info/vectordb").exists()
            }
        }
        
        return {
            "status": "success",
            "documents": status,
            "message": "Document status retrieved successfully."
        }
    except Exception as e:
        return {
            "status": "error",
            "documents": {},
            "message": f"Error checking document status: {e}"
        }

@router.get("/", tags=["Meta"])
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "Welcome to the Smart Form Fill API",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "info_url": "/info"
    } 
    