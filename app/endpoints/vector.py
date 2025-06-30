from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
import json

from app.services.resume_extractor import ResumeExtractor
from app.services.personal_info_extractor import PersonalInfoExtractor

router = APIRouter(
    prefix="/api/v1",
    tags=["Vector Database"]
)

# Initialize services
resume_extractor = ResumeExtractor()
personal_info_extractor = PersonalInfoExtractor()

# Pydantic models
class ReembedResponse(BaseModel):
    status: str
    message: str
    processing_time: float
    timestamp: str
    details: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    k: int = 3

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[Dict[str, Any]]
    search_time: float
    total_results: int

class StatusResponse(BaseModel):
    status: str
    database_info: Dict[str, Any]
    last_updated: Optional[str]

class BatchReembedResponse(BaseModel):
    status: str
    message: str
    results: Dict[str, Any]
    processing_time: float
    total_time: float

@router.get("/resume/status", response_model=StatusResponse)
async def get_resume_status():
    """Get the current status of the resume vector database"""
    try:
        # Check if vector database exists and get info
        vectordb_path = "resume/vectordb"
        index_file = f"{vectordb_path}/index.json"
        
        if not os.path.exists(index_file):
            return StatusResponse(
                status="not_found",
                database_info={"message": "Resume vector database not found. Run re-embed first."},
                last_updated=None
            )
        
        # Read index file for database info
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        return StatusResponse(
            status="ready",
            database_info={
                "total_documents": len(index_data.get("files", [])),
                "latest_file": index_data.get("latest_file", ""),
                "vector_dimensions": 1536,
                "database_path": vectordb_path
            },
            last_updated=index_data.get("timestamp", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resume status: {str(e)}")

@router.get("/personal-info/status", response_model=StatusResponse)
async def get_personal_info_status():
    """Get the current status of the personal info vector database"""
    try:
        vectordb_path = "info/vectordb"
        index_file = f"{vectordb_path}/index.json"
        
        if not os.path.exists(index_file):
            return StatusResponse(
                status="not_found",
                database_info={"message": "Personal info vector database not found. Run re-embed first."},
                last_updated=None
            )
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        return StatusResponse(
            status="ready",
            database_info={
                "total_documents": len(index_data.get("files", [])),
                "latest_file": index_data.get("latest_file", ""),
                "vector_dimensions": 1536,
                "database_path": vectordb_path
            },
            last_updated=index_data.get("timestamp", "")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get personal info status: {str(e)}")

@router.post("/resume/reembed", response_model=ReembedResponse)
async def reembed_resume():
    """Re-process and re-embed the resume document"""
    try:
        start_time = datetime.now()
        
        # Process resume with LangChain
        result = resume_extractor.process_resume()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return ReembedResponse(
            status="success",
            message="Resume re-embedded successfully",
            processing_time=processing_time,
            timestamp=result.get("timestamp", ""),
            details=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-embed resume: {str(e)}")

@router.post("/personal-info/reembed", response_model=ReembedResponse)
async def reembed_personal_info():
    """Re-process and re-embed the personal information document"""
    try:
        start_time = datetime.now()
        
        # Process personal info with LangChain
        result = personal_info_extractor.process_personal_info()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return ReembedResponse(
            status="success",
            message="Personal info re-embedded successfully",
            processing_time=processing_time,
            timestamp=result.get("timestamp", ""),
            details=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-embed personal info: {str(e)}")

@router.post("/resume/search", response_model=SearchResponse)
async def search_resume(
    query: str = Query(..., description="Search query or field question"),
    k: int = Query(3, description="Number of results")
):
    """Search the resume vector database"""
    try:
        start_time = datetime.now()
        
        # Perform search using the existing service
        search_result = resume_extractor.search(query, k)
        
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        
        return SearchResponse(
            status="success",
            query=query,
            results=search_result.get("results", []),
            search_time=search_time,
            total_results=search_result.get("total_results", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search resume: {str(e)}")

@router.post("/personal-info/search", response_model=SearchResponse)
async def search_personal_info(
    query: str = Query(..., description="Search query or field question"),
    k: int = Query(3, description="Number of results")
):
    """Search the personal info vector database"""
    try:
        start_time = datetime.now()
        
        # Perform search
        search_result = personal_info_extractor.search(query, k)
        
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        
        return SearchResponse(
            status="success",
            query=query,
            results=search_result.get("results", []),
            search_time=search_time,
            total_results=search_result.get("total_results", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search personal info: {str(e)}")

@router.post("/reembed-all", response_model=BatchReembedResponse)
async def reembed_all():
    """Re-embed both resume and personal info, and return combined results"""
    try:
        start_time = datetime.now()
        
        # Re-embed resume
        resume_result = resume_extractor.process_resume()
        
        # Re-embed personal info
        personal_info_result = personal_info_extractor.process_personal_info()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return BatchReembedResponse(
            status="success",
            message="Re-embedding completed for all sources",
            results={
                "resume": resume_result,
                "personal_info": personal_info_result
            },
            processing_time=total_time,
            total_time=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-embed all: {str(e)}") 