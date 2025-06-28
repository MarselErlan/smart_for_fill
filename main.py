#!/usr/bin/env python3
"""
Smart Form Fill API - Vector Database Management + Form Auto-Fill Pipeline
Comprehensive API for managing resume/personal info vector databases and form filling
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from app.services.resume_extractor import ResumeExtractor
from app.services.personal_info_extractor import PersonalInfoExtractor



# Import form filling services
from app.services.form_pipeline import FormPipeline
from app.services.cache_service import CacheService

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smart_form_filler")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize services
resume_extractor = ResumeExtractor()
personal_info_extractor = PersonalInfoExtractor()
cache_service = CacheService()

# Clear Redis cache on startup for fresh analysis
def clear_redis_cache_on_startup():
    """Clear Redis cache on server startup to ensure fresh form analysis"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()  # Test connection
        
        keys = r.keys("*")
        if keys:
            r.flushall()
            logger.info(f"🧹 Cleared {len(keys)} cached entries on startup")
        else:
            logger.info("📭 No cache entries found on startup")
            
    except redis.ConnectionError:
        logger.warning("⚠️  Redis not available - cache clearing skipped")
    except Exception as e:
        logger.warning(f"⚠️  Cache clearing failed: {e}")

# Clear cache on startup
clear_redis_cache_on_startup()

# Initialize form pipeline for auto-filling
try:
    form_pipeline = FormPipeline(
        db_url=DATABASE_URL,
        openai_api_key=OPENAI_API_KEY,
        cache_service=cache_service
    )
    logger.info("✅ Form pipeline initialized successfully")
except Exception as e:
    logger.error(f"❌ Form pipeline initialization failed: {e}")
    form_pipeline = None

# Pydantic models
class ReembedResponse(BaseModel):
    status: str
    message: str
    processing_time: float
    database_info: Dict[str, Any]

class SearchRequest(BaseModel):
    query: str
    k: int = 3

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[Dict[str, Any]]
    search_time: float

class StatusResponse(BaseModel):
    status: str
    database_info: Dict[str, Any]
    last_updated: Optional[str]

class BatchReembedResponse(BaseModel):
    status: str
    message: str
    results: Dict[str, Any]
    total_time: float

# Form pipeline models
class PipelineRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}
    force_refresh: bool = False
    submit: bool = False
    manual_submit: bool = True
    headless: bool = False
    use_documents: bool = True

class PipelineResponse(BaseModel):
    status: str
    url: str
    pipeline_status: str
    steps: Dict[str, Any]
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="Smart Form Fill API",
    description="Vector Database Management + Form Auto-Fill Pipeline",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# VECTOR DATABASE ENDPOINTS
# ============================================================================

@app.get("/api/v1/resume/status", response_model=StatusResponse)
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
        import json
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

@app.get("/api/v1/personal-info/status", response_model=StatusResponse)
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
        
        import json
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

@app.post("/api/v1/resume/reembed", response_model=ReembedResponse)
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
            database_info=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-embed resume: {str(e)}")

@app.post("/api/v1/personal-info/reembed", response_model=ReembedResponse)
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
            database_info=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to re-embed personal info: {str(e)}")

@app.post("/api/v1/resume/search", response_model=SearchResponse)
async def search_resume(
    query: str = Query(..., description="Search query or field question"),
    k: int = Query(3, description="Number of results")
):
    """Search the resume vector database (embedding-based only)"""
    try:
        start_time = datetime.now()
        from app.services.resume_extractor import ResumeExtractor
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        resume_extractor = ResumeExtractor()
        embeddings = resume_extractor.embeddings
        vectordb_path = resume_extractor.vectordb_path
        import json, os
        index_file = vectordb_path / "index.json"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="No resume vector database found.")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        if not index_data["entries"]:
            raise HTTPException(status_code=404, detail="No entries in resume vector database.")
        latest_entry = index_data["entries"][-1]
        faiss_path = latest_entry.get("faiss_store")
        if not faiss_path or not os.path.exists(faiss_path):
            raise HTTPException(status_code=404, detail="No FAISS store available for resume.")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        query_embedding = embeddings.embed_query(query)
        results = vectorstore.similarity_search_by_vector(query_embedding, k=k)
        formatted_results = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in results
        ]
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        return SearchResponse(
            status="success",
            query=query,
            results=formatted_results,
            search_time=search_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume search failed: {str(e)}")

@app.post("/api/v1/personal-info/search", response_model=SearchResponse)
async def search_personal_info(
    query: str = Query(..., description="Search query or field question"),
    k: int = Query(3, description="Number of results")
):
    """Search the personal info vector database (embedding-based only)"""
    try:
        start_time = datetime.now()
        from app.services.personal_info_extractor import PersonalInfoExtractor
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        personal_info_extractor = PersonalInfoExtractor()
        embeddings = personal_info_extractor.embeddings
        vectordb_path = personal_info_extractor.vectordb_path
        import json, os
        index_file = vectordb_path / "index.json"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="No personal info vector database found.")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        if not index_data["entries"]:
            raise HTTPException(status_code=404, detail="No entries in personal info vector database.")
        latest_entry = index_data["entries"][-1]
        faiss_path = latest_entry.get("faiss_store")
        if not faiss_path or not os.path.exists(faiss_path):
            raise HTTPException(status_code=404, detail="No FAISS store available for personal info.")
        vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
        query_embedding = embeddings.embed_query(query)
        results = vectorstore.similarity_search_by_vector(query_embedding, k=k)
        formatted_results = [
            {"content": doc.page_content, "metadata": doc.metadata} for doc in results
        ]
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        return SearchResponse(
            status="success",
            query=query,
            results=formatted_results,
            search_time=search_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Personal info search failed: {str(e)}")

@app.post("/api/v1/reembed-all", response_model=BatchReembedResponse)
async def reembed_all():
    """Re-embed both resume and personal info databases"""
    try:
        start_time = datetime.now()
        results = {}
        
        # Re-embed resume
        try:
            resume_result = resume_extractor.process_resume()
            results["resume"] = {"status": "success", "info": resume_result}
        except Exception as e:
            results["resume"] = {"status": "error", "error": str(e)}
        
        # Re-embed personal info
        try:
            personal_info_result = personal_info_extractor.process_personal_info()
            results["personal_info"] = {"status": "success", "info": personal_info_result}
        except Exception as e:
            results["personal_info"] = {"status": "error", "error": str(e)}
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Determine overall status
        overall_status = "success" if all(r["status"] == "success" for r in results.values()) else "partial"
        
        return BatchReembedResponse(
            status=overall_status,
            message=f"Batch re-embedding completed in {total_time:.2f}s",
            results=results,
            total_time=total_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch re-embedding failed: {str(e)}")

# ============================================================================
# FORM AUTO-FILL PIPELINE ENDPOINTS
# ============================================================================

@app.post("/api/run-pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    """
    Run the complete form pipeline: analyze labels, extract questions, (optionally fill/submit)
    """
    if form_pipeline is None:
        raise HTTPException(status_code=503, detail="Form pipeline not initialized.")
    result = await form_pipeline.run_complete_pipeline(
        url=str(request.url),
        user_data=request.user_data,
        force_refresh=request.force_refresh,
        submit_form=request.submit,
        preview_only=not request.submit
    )
    # Return the new structure with label_html and question for each label
    return {
        "status": "success" if result.get("pipeline_status") == "completed" else "error",
        "url": result.get("url"),
        "pipeline_status": result.get("pipeline_status"),
        "steps": result.get("steps"),
        "message": result.get("results", {}).get("status", "No result"),
    }

@app.post("/api/analyze-form")
async def analyze_form(request: PipelineRequest) -> Dict[str, Any]:
    """
    Analyze a form without filling it (fast mode only)
    """
    if not form_pipeline:
        raise HTTPException(status_code=503, detail="Form pipeline service not available")
    try:
        url = str(request.url)
        result = await form_pipeline.form_analyzer.analyze_form(url, request.force_refresh)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form analysis failed: {str(e)}")

# ============================================================================
# HEALTH AND INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Form Fill API - Vector Database + Form Auto-Fill",
        "version": "2.0.0",
        "features": [
            "Vector database management for resume and personal info",
            "Intelligent form analysis and auto-filling",
            "LangChain-powered document processing",
            "FAISS vector search capabilities"
        ],
        "endpoints": {
            "vector_db": "/api/v1/",
            "form_pipeline": "/api/run-pipeline",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "vector_databases": "ready",
            "form_pipeline": "ready" if form_pipeline else "unavailable"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

 