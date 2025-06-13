#!/usr/bin/env python3
"""
Smart Form Fill - Main FastAPI Application
Enhanced with vector database re-embedding endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from datetime import datetime

# Import our extractors
from resume_extractor import ResumeExtractor
from personal_info_extractor import PersonalInfoExtractor

app = FastAPI(
    title="Smart Form Fill API",
    description="AI-powered form filling with vector database management",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class ReembedResponse(BaseModel):
    status: str
    message: str
    timestamp: str
    processing_time: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class VectorDBStatus(BaseModel):
    database_type: str
    status: str
    last_updated: Optional[str] = None
    total_entries: Optional[int] = None
    latest_entry: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Form Fill API with Vector Database Management",
        "version": "2.0.0",
        "endpoints": {
            "resume": {
                "reembed": "/api/v1/resume/reembed",
                "status": "/api/v1/resume/status",
                "search": "/api/v1/resume/search"
            },
            "personal_info": {
                "reembed": "/api/v1/personal-info/reembed", 
                "status": "/api/v1/personal-info/status",
                "search": "/api/v1/personal-info/search"
            }
        }
    }

@app.post("/api/v1/resume/reembed", response_model=ReembedResponse)
async def reembed_resume():
    """
    Re-embed the resume document and update the vector database
    Processes docs/resume/ERIC_ABRAM33.docx and saves to resume/vectordb
    """
    try:
        start_time = datetime.now()
        
        # Initialize resume extractor
        extractor = ResumeExtractor()
        
        # Process the resume
        result = extractor.process_resume()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result["status"] == "success":
            return ReembedResponse(
                status="success",
                message="Resume vector database updated successfully",
                timestamp=result["timestamp"],
                processing_time=processing_time,
                details={
                    "documents_loaded": result["documents_loaded"],
                    "chunks_created": result["chunks_created"],
                    "embedding_dimension": result["embedding_data"]["dimension"],
                    "model": result["embedding_data"]["model"],
                    "vectorstore_created": result["vectorstore_created"],
                    "files_created": result["files_created"]
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Resume processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to re-embed resume: {str(e)}"
        )

@app.post("/api/v1/personal-info/reembed", response_model=ReembedResponse)
async def reembed_personal_info():
    """
    Re-embed the personal information document and update the vector database
    Processes docs/info/personal_information.txt and saves to info/vectordb
    """
    try:
        start_time = datetime.now()
        
        # Initialize personal info extractor
        extractor = PersonalInfoExtractor()
        
        # Process the personal info
        result = extractor.process_personal_info()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if result["status"] == "success":
            return ReembedResponse(
                status="success",
                message="Personal info vector database updated successfully",
                timestamp=result["timestamp"],
                processing_time=processing_time,
                details={
                    "documents_loaded": result["documents_loaded"],
                    "chunks_created": result["chunks_created"],
                    "embedding_dimension": result["embedding_data"]["dimension"],
                    "model": result["embedding_data"]["model"],
                    "vectorstore_created": result["vectorstore_created"],
                    "files_created": result["files_created"]
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Personal info processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to re-embed personal info: {str(e)}"
        )

@app.get("/api/v1/resume/status", response_model=VectorDBStatus)
async def get_resume_status():
    """Get the current status of the resume vector database"""
    try:
        from pathlib import Path
        import json
        
        index_file = Path("resume/vectordb/index.json")
        
        if not index_file.exists():
            return VectorDBStatus(
                database_type="resume",
                status="not_initialized",
                message="Resume vector database not found. Run /api/v1/resume/reembed first."
            )
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        latest_entry = index_data["entries"][-1] if index_data["entries"] else None
        
        return VectorDBStatus(
            database_type="resume",
            status="ready",
            last_updated=index_data.get("last_updated"),
            total_entries=index_data.get("total_entries", 0),
            latest_entry=latest_entry
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resume status: {str(e)}"
        )

@app.get("/api/v1/personal-info/status", response_model=VectorDBStatus)
async def get_personal_info_status():
    """Get the current status of the personal info vector database"""
    try:
        from pathlib import Path
        import json
        
        index_file = Path("info/vectordb/index.json")
        
        if not index_file.exists():
            return VectorDBStatus(
                database_type="personal_info",
                status="not_initialized",
                message="Personal info vector database not found. Run /api/v1/personal-info/reembed first."
            )
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        latest_entry = index_data["entries"][-1] if index_data["entries"] else None
        
        return VectorDBStatus(
            database_type="personal_info",
            status="ready",
            last_updated=index_data.get("last_updated"),
            total_entries=index_data.get("total_entries", 0),
            latest_entry=latest_entry
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get personal info status: {str(e)}"
        )

@app.post("/api/v1/resume/search")
async def search_resume(query: str, k: int = 5):
    """Search the resume vector database"""
    try:
        extractor = ResumeExtractor()
        results = extractor.search_resume(query, k=k)
        
        if "error" in results:
            raise HTTPException(
                status_code=404,
                detail=results["error"]
            )
        
        return {
            "status": "success",
            "query": query,
            "results": results["results"],
            "total_results": results["total_results"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Resume search failed: {str(e)}"
        )

@app.post("/api/v1/personal-info/search")
async def search_personal_info(query: str, k: int = 5):
    """Search the personal info vector database"""
    try:
        extractor = PersonalInfoExtractor()
        results = extractor.search_personal_info(query, k=k)
        
        if "error" in results:
            raise HTTPException(
                status_code=404,
                detail=results["error"]
            )
        
        return {
            "status": "success",
            "query": query,
            "results": results["results"],
            "total_results": results["total_results"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Personal info search failed: {str(e)}"
        )

@app.post("/api/v1/reembed-all")
async def reembed_all():
    """
    Re-embed both resume and personal info vector databases
    Convenient endpoint to update everything at once
    """
    try:
        start_time = datetime.now()
        results = {}
        
        # Re-embed resume
        try:
            resume_extractor = ResumeExtractor()
            resume_result = resume_extractor.process_resume()
            results["resume"] = {
                "status": resume_result["status"],
                "timestamp": resume_result.get("timestamp"),
                "details": resume_result.get("embedding_data", {})
            }
        except Exception as e:
            results["resume"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Re-embed personal info
        try:
            info_extractor = PersonalInfoExtractor()
            info_result = info_extractor.process_personal_info()
            results["personal_info"] = {
                "status": info_result["status"],
                "timestamp": info_result.get("timestamp"),
                "details": info_result.get("embedding_data", {})
            }
        except Exception as e:
            results["personal_info"] = {
                "status": "error",
                "error": str(e)
            }
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Determine overall status
        overall_status = "success"
        if results["resume"]["status"] == "error" or results["personal_info"]["status"] == "error":
            overall_status = "partial_success" if (results["resume"]["status"] == "success" or results["personal_info"]["status"] == "success") else "error"
        
        return {
            "status": overall_status,
            "message": f"Batch re-embedding completed in {processing_time:.2f} seconds",
            "processing_time": processing_time,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch re-embedding failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 