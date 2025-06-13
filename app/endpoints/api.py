"""
Smart Form Fill API - FastAPI routes for form analysis, filling, and pipeline orchestration
"""

import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any, List
from loguru import logger
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime

from app.services.form_analyzer import FormAnalyzer
from app.services.form_filler import FormFiller
from app.services.form_pipeline import FormPipeline
from app.services.cache_service import CacheService

# Load environment variables
load_dotenv()

# Initialize router
router = APIRouter()

# Request/Response Models
class FormAnalysisRequest(BaseModel):
    url: HttpUrl
    force_refresh: bool = False

class PipelineRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}  # Optional: LLM can generate professional data if minimal
    force_refresh: bool = False
    submit: bool = False
    manual_submit: bool = True  # New: Keep browser open for manual submission
    headless: bool = False  # New: Control browser visibility

class PipelinePreviewRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}  # Optional: LLM can generate professional data if minimal
    force_refresh: bool = False

class BatchPipelineRequest(BaseModel):
    forms: List[Dict[str, Any]]  # List of {url, user_data, submit}

class AutoFillRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}  # Optional: LLM can generate professional data if minimal
    submit: bool = False
    manual_submit: bool = False  # New: Keep browser open for manual submission
    headless: bool = False  # New: Control browser visibility

class FormFillRequest(BaseModel):
    url: HttpUrl
    field_map: str  # JSON string from form analyzer
    user_data: Dict[str, Any] = {}  # Optional: LLM can generate professional data if minimal
    submit: bool = False
    manual_submit: bool = False  # New: Keep browser open for manual submission
    headless: bool = False  # New: Control browser visibility

class AutoFillPreviewRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}  # Optional: LLM can generate professional data if minimal

class FormFillPreviewRequest(BaseModel):
    url: HttpUrl
    field_map: str
    user_data: Dict[str, Any] = {}  # Optional: LLM can generate professional data if minimal

class UrlStatus(BaseModel):
    url: str
    status: str

# Initialize services
cache_service = CacheService()
form_analyzer = FormAnalyzer(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    db_url=os.getenv("DATABASE_URL"),
    cache_service=cache_service
)

# Note: FormFiller will be created dynamically based on request parameters

# Initialize pipeline service
form_pipeline = FormPipeline(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    db_url=os.getenv("DATABASE_URL"),
    cache_service=cache_service
)

# ============================================================================
# PIPELINE ENDPOINTS (Main workflow orchestration)
# ============================================================================

@router.post("/run-pipeline")
async def run_pipeline(request: PipelineRequest) -> Dict[str, Any]:
    """
    🚀 MAIN ENDPOINT: Run complete pipeline (analyze → fill → submit)
    This orchestrates the entire workflow automatically
    
    New features:
    - headless: false = Show browser in action
    - manual_submit: true = Keep browser open for manual submission
    """
    try:
        # If manual_submit is requested, handle it directly without using the pipeline
        if request.manual_submit:
            logger.info("🖱️  Manual submission mode - running direct form filling with browser kept open")
            
            # Create FormFiller with requested settings
            form_filler = FormFiller(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                cache_service=cache_service,
                headless=request.headless
            )
            
            # First ensure form is analyzed
            analysis_result = await form_analyzer.analyze_form(str(request.url), request.force_refresh)
            
            if analysis_result["status"] != "success":
                return {
                    "pipeline_status": "failed",
                    "error": f"Analysis failed: {analysis_result.get('error', 'Unknown error')}",
                    "url": str(request.url)
                }
            
            # Now run form filler with manual submission mode
            result = await form_filler.auto_fill_form(
                url=str(request.url),
                user_data=request.user_data,
                submit=False,  # Never auto-submit in manual mode
                manual_submit=True
            )
            
            # Update database status if form was successfully filled
            if result["status"] == "success" and result.get("filled_fields", 0) > 0:
                form_analyzer._create_or_update_form_record(str(request.url), "applied")
            
            # Return result with pipeline-like structure
            return {
                "url": str(request.url),
                "pipeline_status": "completed" if result["status"] == "success" else "failed",
                "steps": {
                    "analysis": {
                        "status": "success",
                        "timestamp": analysis_result.get("timestamp"),
                        "cached": "cached" in str(analysis_result)
                    },
                    "filling": {
                        "status": result["status"],
                        "timestamp": result.get("timestamp"),
                        "filled_fields": result.get("filled_fields", 0),
                        "screenshot": result.get("screenshot"),
                        "submit_result": result.get("submit_result")
                    }
                },
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "user_data_provided": bool(request.user_data),
                "results": result,
                "browser_status": "open_for_manual_submission"
            }
        
        # Regular pipeline execution (non-manual mode)
        else:
            # Create FormFiller with requested settings
            form_filler = FormFiller(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                cache_service=cache_service,
                headless=request.headless
            )
            
            # Update pipeline's form_filler
            form_pipeline.form_filler = form_filler
            
            result = await form_pipeline.run_complete_pipeline(
                url=str(request.url),
                user_data=request.user_data,
                force_refresh=request.force_refresh,
                submit_form=request.submit,
                preview_only=False
            )
            
            return result
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@router.post("/preview-pipeline")
async def preview_pipeline(request: PipelinePreviewRequest) -> Dict[str, Any]:
    """
    👀 Preview complete pipeline (analyze → preview fill)
    Shows what would happen without actually filling the form
    """
    try:
        result = await form_pipeline.run_complete_pipeline(
            url=str(request.url),
            user_data=request.user_data,
            force_refresh=request.force_refresh,
            submit_form=False,
            preview_only=True
        )
        return result
    except Exception as e:
        logger.error(f"Pipeline preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline preview failed: {str(e)}")

@router.post("/batch-pipeline")
async def batch_pipeline(request: BatchPipelineRequest) -> Dict[str, Any]:
    """
    📦 Process multiple forms in batch using the pipeline
    """
    try:
        result = await form_pipeline.batch_process_forms(request.forms)
        return result
    except Exception as e:
        logger.error(f"Batch pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch pipeline failed: {str(e)}")

@router.get("/pipeline-status/{url:path}")
async def get_pipeline_status(url: str) -> Dict[str, Any]:
    """
    📊 Get pipeline status for a specific URL
    """
    try:
        result = await form_pipeline.get_pipeline_status(url)
        return result
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

# ============================================================================
# INDIVIDUAL SERVICE ENDPOINTS (For granular control)
# ============================================================================

@router.post("/analyze-form")
async def analyze_form(request: FormAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze a web form at the provided URL
    """
    try:
        result = await form_analyzer.analyze_form(str(request.url), request.force_refresh)
        return result
    except Exception as e:
        logger.error(f"Form analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Form analysis failed: {str(e)}")

@router.post("/auto-fill-form")
async def auto_fill_form(request: AutoFillRequest) -> Dict[str, Any]:
    """
    Automatically fill a web form using cached analysis data from Redis
    
    New features:
    - headless: false = Show browser in action
    - manual_submit: true = Keep browser open for manual submission
    """
    try:
        # Create FormFiller with requested settings
        form_filler = FormFiller(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cache_service=cache_service,
            headless=request.headless
        )
        
        result = await form_filler.auto_fill_form(
            url=str(request.url),
            user_data=request.user_data,
            submit=request.submit,
            manual_submit=request.manual_submit
        )
        
        # Update database status if form was successfully filled
        if result["status"] == "success" and result.get("filled_fields", 0) > 0:
            form_analyzer._create_or_update_form_record(str(request.url), "applied")
        
        return result
    except Exception as e:
        logger.error(f"Auto form filling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto form filling failed: {str(e)}")

@router.post("/preview-auto-fill")
async def preview_auto_fill(request: AutoFillPreviewRequest) -> Dict[str, Any]:
    """
    Preview what values would be filled automatically using cached analysis data
    """
    try:
        # Use default headless form filler for preview
        form_filler = FormFiller(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cache_service=cache_service,
            headless=True
        )
        
        result = await form_filler.preview_auto_fill(
            url=str(request.url),
            user_data=request.user_data
        )
        return result
    except Exception as e:
        logger.error(f"Auto fill preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto fill preview failed: {str(e)}")

@router.post("/fill-form")
async def fill_form(request: FormFillRequest) -> Dict[str, Any]:
    """
    Fill a web form using AI-powered field mapping and user data
    (Legacy endpoint - requires manual field_map parameter)
    
    New features:
    - headless: false = Show browser in action
    - manual_submit: true = Keep browser open for manual submission
    """
    try:
        # Create FormFiller with requested settings
        form_filler = FormFiller(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cache_service=cache_service,
            headless=request.headless
        )
        
        result = await form_filler.fill_form(
            url=str(request.url),
            field_map=request.field_map,
            user_data=request.user_data,
            submit=request.submit,
            manual_submit=request.manual_submit
        )
        
        # Update database status if form was successfully filled
        if result["status"] == "success" and result.get("filled_fields", 0) > 0:
            form_analyzer._create_or_update_form_record(str(request.url), "applied")
        
        return result
    except Exception as e:
        logger.error(f"Form filling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Form filling failed: {str(e)}")

@router.post("/preview-form-fill")
async def preview_form_fill(request: FormFillPreviewRequest) -> Dict[str, Any]:
    """
    Preview what values would be filled in a form without actually filling it
    (Legacy endpoint - requires manual field_map parameter)
    """
    try:
        # Use default headless form filler for preview
        form_filler = FormFiller(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cache_service=cache_service,
            headless=True
        )
        
        result = await form_filler.preview_form_filling(
            url=str(request.url),
            field_map=request.field_map,
            user_data=request.user_data
        )
        return result
    except Exception as e:
        logger.error(f"Form fill preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Form fill preview failed: {str(e)}")

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/urls", response_model=List[UrlStatus])
async def get_urls():
    """
    Get all analyzed URLs and their current status from the database
    """
    try:
        with form_analyzer._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT url, status 
                    FROM form_submissions 
                    ORDER BY created_at DESC
                """)
                results = cur.fetchall()
                return [{"url": row[0], "status": row[1]} for row in results]
    except Exception as e:
        logger.error(f"Error fetching URLs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    """
    return {"status": "healthy"} 
    