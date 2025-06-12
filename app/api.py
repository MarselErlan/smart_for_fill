"""
Smart Form Fill API - FastAPI routes for form analysis with Supabase integration
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
import time

from app.services.form_analyzer import FormAnalyzer
from app.services.supabase_service import SupabaseService
from db.schemas import (
    FormAnalysisRequest,
    FormResponse,
    FormAnalysisResponse,
    HealthResponse,
    DetailedFormAnalysisResponse
)
from app.services.resume_extractor import ResumeExtractor
from app.services.personal_info_vector import PersonalInfoVectorDB

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Check environment variables
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL or SUPABASE_KEY environment variables not set")

# Initialize services
form_analyzer = FormAnalyzer(openai_api_key=OPENAI_API_KEY)
supabase_service = SupabaseService()
resume_extractor = ResumeExtractor()
personal_info_vector_db = PersonalInfoVectorDB()

# Initialize FastAPI
app = FastAPI(
    title="Smart Form Fill API",
    description="AI-powered API for analyzing web forms with Supabase integration",
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

@app.post("/api/analyze-form", response_model=FormAnalysisResponse)
async def analyze_form(request: FormAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze a web form at the provided URL and save to Supabase
    """
    url = str(request.url)
    
    try:
        # Save URL to Supabase first
        supabase_result = supabase_service.add_form_url(url)
        
        if supabase_result["status"] == "error":
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to save URL to database: {supabase_result['message']}"
            )
        
        # Analyze the form
        analysis_result = form_analyzer.analyze_form(url)
        
        if analysis_result["status"] == "success":
            # Update the form status to analyzed
            supabase_service.update_form_analysis_status(url, True)
            
        # Return combined result
        return {
            **analysis_result,
            "database_id": supabase_result.get("id"),
            "url": url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form analysis failed: {str(e)}")

@app.post("/api/analyze-form-detailed", response_model=DetailedFormAnalysisResponse)
async def analyze_form_detailed(request: FormAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze a web form with detailed field structure information
    """
    url = str(request.url)
    
    try:
        # Get basic analysis first
        basic_result = await analyze_form(request)
        
        # TODO: In a real implementation, you would parse the field_map to create structured fields
        # This is a placeholder implementation
        fields = []
        
        # Return enhanced result
        return {
            **basic_result,
            "fields": fields
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detailed form analysis failed: {str(e)}")

@app.get("/api/forms", response_model=List[FormResponse])
async def get_forms() -> List[FormResponse]:
    """
    Get all forms from the database
    """
    try:
        forms = supabase_service.get_all_forms()
        return forms
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve forms: {str(e)}")

@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.post("/api/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    """
    Extract structured information from a resume file (PDF, DOCX, etc.)
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    # Extract info
    info = resume_extractor.extract(temp_path)
    return {"status": "success", "data": info}

@app.post("/api/vectorize-personal-info")
async def vectorize_personal_info(personal_info: dict):
    """
    Vectorize and store personal info in the vector DB
    """
    vector = personal_info_vector_db.vectorize(personal_info)
    # Example: use email as user_id, or generate one
    user_id = personal_info.get("email", "unknown")
    personal_info_vector_db.store(user_id, vector, personal_info)
    return {"status": "success", "vector": vector}

@app.post("/api/search-personal-info")
async def search_personal_info(query: dict):
    """
    Search for similar personal info vectors
    """
    query_vector = personal_info_vector_db.vectorize(query)
    results = personal_info_vector_db.search(query_vector)
    return {"status": "success", "results": results} 