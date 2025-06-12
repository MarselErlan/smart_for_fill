"""
Smart Form Fill API - FastAPI routes for form analysis
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

from app.services.form_analyzer import FormAnalyzer
from app.services.cache_service import CacheService

# Load environment variables
load_dotenv()

# Initialize router
router = APIRouter()

# Request/Response Models
class FormAnalysisRequest(BaseModel):
    url: HttpUrl
    force_refresh: bool = False

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
    