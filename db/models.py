"""
Database models for the Smart Form Fill API
These represent the data structures as they are stored in Supabase.
"""

from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, List, Any, Optional
from datetime import datetime, date


class FormDb(BaseModel):
    """Database model for form entries"""
    id: str
    url: str
    status: str  # "applied" or "not_applied"
    applied_counter: int = 0
    applied_date: Optional[str] = None  # ISO date string
    created_at: str
    
    class Config:
        from_attributes = True


class FormField(BaseModel):
    """Model for a form field extracted during analysis"""
    field_type: str
    purpose: str
    selector: str
    validation: Optional[str] = None


class SupabaseResult(BaseModel):
    """Result model for Supabase operations"""
    status: str
    message: str
    id: Optional[str] = None 

def mark_form_applied(self, url: str):
    today = date.today().isoformat()
    form = self.supabase.table("forms").select("*").eq("url", url).execute()
    if form.data and len(form.data) > 0:
        entry = form.data[0]
        if entry.get("applied_date") == today:
            new_counter = entry.get("applied_counter", 0) + 1
        else:
            new_counter = 1
        self.supabase.table("forms").update({
            "status": "applied",
            "applied_counter": new_counter,
            "applied_date": today
        }).eq("url", url).execute()
        return True
    return False 