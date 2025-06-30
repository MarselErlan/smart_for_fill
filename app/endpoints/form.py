from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, HttpUrl
from typing import Dict, Any
import os

from app.services.form_pipeline import FormPipeline
from app.services.cache_service import CacheService
from app.services.form_analyzer import FormAnalyzer
from app.services.form_filler import FormFiller
from loguru import logger

router = APIRouter(
    prefix="/api",
    tags=["Form Operations"]
)

# Pydantic Models
class FillFormRequest(BaseModel):
    url: HttpUrl
    user_data: Dict[str, Any] = {}
    headless: bool = True

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

# Pydantic Models from legacy api.py
class FormAnalysisRequest(BaseModel):
    url: HttpUrl
    force_refresh: bool = False

# In-memory instances for now, consider dependency injection for long-term
cache_service = CacheService()
form_pipeline = FormPipeline(
    db_url=os.getenv("DATABASE_URL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    cache_service=cache_service,
    headless=False
)

@router.post("/run-pipeline", response_model=PipelineResponse)
async def run_pipeline_endpoint(request: PipelineRequest) -> Dict[str, Any]:
    """Run the complete form filling pipeline"""
    if not form_pipeline:
        raise HTTPException(status_code=500, detail="Form pipeline not initialized")
    
    try:
        result = await form_pipeline.run_complete_pipeline(
            url=str(request.url),
            user_data=request.user_data,
            force_refresh=request.force_refresh,
            submit_form=request.submit,
            preview_only=not request.manual_submit
        )
        return result
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-form")
async def analyze_form(request: PipelineRequest) -> Dict[str, Any]:
    """Analyze a form to extract fields (enhanced version)"""
    if not form_pipeline:
        raise HTTPException(status_code=500, detail="Form pipeline not initialized")
    
    try:
        # Using the pipeline's analysis service for consistency
        result = await form_pipeline.analyzer.analyze_form(
            url=str(request.url),
            force_refresh=request.force_refresh
        )
        return {
            "status": "success",
            "url": str(request.url),
            "analysis": result
        }
    except Exception as e:
        logger.error(f"Form analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/fields")
async def get_form_fields(payload: dict = Body(...)):
    """Get form fields using the fast label analyzer"""
    url = payload.get("url")
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body.")
    
    analyzer = FormAnalyzer(cache_service=cache_service)
    result = await analyzer.analyze_labels_fast(url)
    return {"status": "success", "fields": result}

@router.post("/autofill-preview")
async def autofill_preview(payload: dict = Body(...)):
    """Generate a preview of the auto-filled form data"""
    url = payload.get("url")
    user_data = payload.get("user_data", {})
    if not url:
        raise HTTPException(status_code=400, detail="Missing 'url' in request body.")
    
    analyzer = FormAnalyzer(cache_service=cache_service)
    filler = FormFiller(openai_api_key=os.getenv("OPENAI_API_KEY"), cache_service=cache_service, headless=True)
    
    label_result = await analyzer.analyze_labels_fast(url)
    
    # Build dummy field mappings for preview
    field_mappings = [{"label": label_html} for label_html in label_result.get("labels", [])]
    
    preview = await filler.auto_fill_form(url, field_mappings, user_data)
    return {"status": "success", "autofill_preview": preview}

@router.post("/fill-form-in-browser", response_model=Dict)
async def fill_form_in_browser(request: FillFormRequest):
    """
    Run the full pipeline and open the form in a browser for manual review.
    """
    try:
        # The pipeline now returns the complete result dictionary
        result = await form_pipeline.run_complete_pipeline(
            url=str(request.url), 
            user_data=request.user_data,
            submit_form=False, 
            preview_only=False
        )
        
        if result.get("status") == "failed":
            raise HTTPException(status_code=500, detail=result)
            
        return {"status": "success", "details": "Form filling process completed.", "result": result}
    except Exception as e:
        logger.error(f"Error in /fill-form-in-browser endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 