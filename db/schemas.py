"""
API schemas for the Smart Form Fill API
These define the shape of requests and responses for API endpoints.
"""

from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime


class FormAnalysisRequest(BaseModel):
    """Request model for form analysis"""
    url: HttpUrl = Field(..., description="URL of the form to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/job-application"
            }
        }


class FormResponse(BaseModel):
    """Response model for form entries"""
    id: str
    url: str
    created_at: str
    analyzed: bool
    status: str  # "applied" or "not_applied"
    applied_counter: int = 0
    applied_date: Optional[str] = None  # ISO date string
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com/job-application",
                "created_at": "2023-06-01T12:00:00Z",
                "analyzed": True,
                "status": "not_applied",
                "applied_counter": 0,
                "applied_date": None
            }
        }


class FormFieldSchema(BaseModel):
    """Schema for a form field extracted during analysis"""
    field_type: str
    purpose: str
    selector: str
    validation: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "field_type": "text",
                "purpose": "full_name",
                "selector": "#fullName",
                "validation": "required"
            }
        }


class FormAnalysisResponse(BaseModel):
    """Response model for form analysis results"""
    status: str
    field_map: str
    timestamp: str
    database_id: Optional[str] = None
    url: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "field_map": "Field 1: Name, Field 2: Email...",
                "timestamp": "2023-06-01T12:00:00Z",
                "database_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com/job-application"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy"
            }
        }


class DetailedFormAnalysisResponse(FormAnalysisResponse):
    """Extended response model with structured field data"""
    fields: List[FormFieldSchema] = []
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "field_map": "Field 1: Name, Field 2: Email...",
                "timestamp": "2023-06-01T12:00:00Z",
                "database_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com/job-application",
                "fields": [
                    {
                        "field_type": "text",
                        "purpose": "full_name",
                        "selector": "#fullName",
                        "validation": "required"
                    }
                ]
            }
        } 