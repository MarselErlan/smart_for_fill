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

from resume_extractor import ResumeExtractor
from personal_info_extractor import PersonalInfoExtractor

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
        openai_api_key=OPENAI_API_KEY,
        db_url=DATABASE_URL,
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
async def search_resume(query: str = Query(..., description="Search query"), k: int = Query(3, description="Number of results")):
    """Search the resume vector database"""
    try:
        start_time = datetime.now()
        
        # Perform search
        results = resume_extractor.search_resume(query, k)
        
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        
        return SearchResponse(
            status="success",
            query=query,
            results=results,
            search_time=search_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume search failed: {str(e)}")

@app.post("/api/v1/personal-info/search", response_model=SearchResponse)
async def search_personal_info(query: str = Query(..., description="Search query"), k: int = Query(3, description="Number of results")):
    """Search the personal info vector database"""
    try:
        start_time = datetime.now()
        
        # Perform search
        results = personal_info_extractor.search_personal_info(query, k)
        
        end_time = datetime.now()
        search_time = (end_time - start_time).total_seconds()
        
        return SearchResponse(
            status="success",
            query=query,
            results=results,
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
    🚀 MAIN ENDPOINT: Run complete form filling pipeline (analyze → fill → submit)
    
    Features:
    - Automatic form analysis and filling
    - Vector database integration for user data
    - Manual submission mode (keeps browser open)
    - Document-based user data loading
    """
    if not form_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="Form pipeline service not available. Check environment variables."
        )
    
    try:
        url = str(request.url)
        user_data = request.user_data.copy()
        
        # Load user data from vector databases if requested
        if request.use_documents:
            logger.info("📄 Loading user data from vector databases")
            
            try:
                # Search resume for relevant info
                resume_search_result = resume_extractor.search_resume("professional experience skills education", k=5)
                resume_text = ""
                if resume_search_result and "results" in resume_search_result:
                    resume_text = " ".join([r.get("content", "") for r in resume_search_result["results"]])
                
                # Search personal info for contact details
                personal_search_result = personal_info_extractor.search_personal_info("contact information work authorization salary", k=3)
                personal_text = ""
                if personal_search_result and "results" in personal_search_result:
                    personal_text = " ".join([r.get("content", "") for r in personal_search_result["results"]])
                
                # Combine vector database data
                vector_data = {
                    "resume_content": resume_text,
                    "personal_info": personal_text,
                    "data_source": "vector_databases"
                }
                
                # Merge with provided user_data
                user_data.update(vector_data)
                logger.info("✅ Vector database data loaded successfully")
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load vector database data: {e}")
        
        # Run the form pipeline
        if request.manual_submit:
            logger.info("🖱️ Manual submission mode - browser will stay open")
            
            # For manual submission, we'll use the form filler directly
            form_filler = form_pipeline.form_filler
            form_filler.headless = request.headless
            
            # First analyze the form
            analysis_result = await form_pipeline.form_analyzer.analyze_form(url, request.force_refresh)
            
            if analysis_result["status"] != "success":
                return PipelineResponse(
                    status="error",
                    url=url,
                    pipeline_status="failed",
                    steps={"analysis": {"status": "failed", "error": analysis_result.get("error")}},
                    message=f"Form analysis failed: {analysis_result.get('error', 'Unknown error')}"
                )
            
            # Fill the form with manual submission
            fill_result = await form_filler.auto_fill_form(
                url=url,
                user_data=user_data,
                submit=False,  # Never auto-submit in manual mode
                manual_submit=True
            )
            
            return PipelineResponse(
                status="success" if fill_result["status"] == "success" else "error",
                url=url,
                pipeline_status="completed_manual" if fill_result["status"] == "success" else "failed",
                steps={
                    "analysis": {"status": "success", "timestamp": analysis_result.get("timestamp")},
                    "filling": {
                        "status": fill_result["status"],
                        "filled_fields": fill_result.get("filled_fields", 0),
                        "screenshot": fill_result.get("screenshot")
                    }
                },
                message="Form filled successfully. Browser kept open for manual submission." if fill_result["status"] == "success" else f"Form filling failed: {fill_result.get('error', 'Unknown error')}"
            )
        
        else:
            # Regular pipeline execution
            form_pipeline.form_filler.headless = request.headless
            
            result = await form_pipeline.run_complete_pipeline(
                url=url,
                user_data=user_data,
                force_refresh=request.force_refresh,
                submit_form=request.submit,
                preview_only=False
            )
            
            return PipelineResponse(
                status="success" if result["pipeline_status"] == "completed" else "error",
                url=url,
                pipeline_status=result["pipeline_status"],
                steps=result.get("steps", {}),
                message=result.get("error", "Pipeline completed successfully") if result["pipeline_status"] != "completed" else "Pipeline completed successfully"
            )
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.post("/api/analyze-form")
async def analyze_form(request: PipelineRequest) -> Dict[str, Any]:
    """
    Analyze a form without filling it
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

 