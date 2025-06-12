"""
Supabase Service - Handles interactions with Supabase for form URL tracking
"""

import os
from supabase import create_client
from dotenv import load_dotenv
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from loguru import logger

from db.models import FormDb, SupabaseResult
from db.schemas import FormResponse

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.error("SUPABASE_URL or SUPABASE_KEY environment variables not set")


class SupabaseService:
    def __init__(self):
        """Initialize the Supabase client"""
        self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
    def add_form_url(self, url: str) -> Dict[str, Any]:
        """
        Add a form URL to the Supabase 'forms' table
        
        Args:
            url: The form URL to save
            
        Returns:
            Dict: Result of the operation
        """
        try:
            # Check if URL already exists
            existing = self.supabase.table("forms").select("*").eq("url", url).execute()
            
            if existing.data and len(existing.data) > 0:
                # URL already exists
                result = SupabaseResult(
                    status="success",
                    message="URL already exists in database",
                    id=existing.data[0]["id"]
                )
                return result.model_dump()
            
            # Insert new URL with default status
            result = self.supabase.table("forms").insert({
                "url": url,
                "created_at": datetime.now().isoformat(),
                "status": "not_applied",
                "applied_counter": 0,
                "applied_date": None
            }).execute()
            
            if result.data and len(result.data) > 0:
                return SupabaseResult(
                    status="success",
                    message="URL added successfully",
                    id=result.data[0]["id"]
                ).model_dump()
            else:
                return SupabaseResult(
                    status="error", 
                    message="Failed to add URL"
                ).model_dump()
                
        except Exception as e:
            logger.error(f"Error adding form URL: {e}")
            return SupabaseResult(
                status="error", 
                message=str(e)
            ).model_dump()
    
    def get_all_forms(self) -> List[FormResponse]:
        """
        Get all form URLs from the database
        
        Returns:
            List[FormResponse]: List of form entries
        """
        try:
            result = self.supabase.table("forms").select("*").order("created_at", desc=True).execute()
            forms = []
            
            for form_data in result.data:
                form = FormResponse(
                    id=form_data["id"],
                    url=form_data["url"],
                    created_at=form_data["created_at"],
                    analyzed=form_data["analyzed"],
                    status=form_data.get("status", "not_applied"),
                    applied_counter=form_data.get("applied_counter", 0),
                    applied_date=form_data.get("applied_date")
                )
                forms.append(form)
                
            return forms
        except Exception as e:
            logger.error(f"Error getting form URLs: {e}")
            return []
    
    def update_form_analysis_status(self, url: str, analyzed: bool = True) -> Dict[str, Any]:
        """
        Update the analyzed status of a form
        
        Args:
            url: The form URL to update
            analyzed: The analysis status
            
        Returns:
            Dict: Result of the operation
        """
        try:
            result = self.supabase.table("forms").update({"analyzed": analyzed}).eq("url", url).execute()
            
            if result.data and len(result.data) > 0:
                return SupabaseResult(
                    status="success", 
                    message="Form status updated"
                ).model_dump()
            else:
                return SupabaseResult(
                    status="error", 
                    message="Failed to update form status"
                ).model_dump()
                
        except Exception as e:
            logger.error(f"Error updating form status: {e}")
            return SupabaseResult(
                status="error", 
                message=str(e)
            ).model_dump()

    def mark_form_applied(self, url: str) -> Dict[str, Any]:
        """
        Mark a form as applied and update its application counter.
        If it's a new day, reset the counter to 1, otherwise increment it.
        
        Args:
            url: The form URL to update
            
        Returns:
            Dict: Result of the operation
        """
        try:
            today = date.today().isoformat()
            
            # Get current form data
            form = self.supabase.table("forms").select("*").eq("url", url).execute()
            
            if not form.data or len(form.data) == 0:
                return SupabaseResult(
                    status="error",
                    message="Form URL not found"
                ).model_dump()
                
            current_form = form.data[0]
            
            # Determine new counter value
            if current_form.get("applied_date") == today:
                new_counter = current_form.get("applied_counter", 0) + 1
            else:
                new_counter = 1
                
            # Update form status and counter
            result = self.supabase.table("forms").update({
                "status": "applied",
                "applied_counter": new_counter,
                "applied_date": today
            }).eq("url", url).execute()
            
            if result.data and len(result.data) > 0:
                return SupabaseResult(
                    status="success",
                    message=f"Form marked as applied (attempt #{new_counter} today)"
                ).model_dump()
            else:
                return SupabaseResult(
                    status="error",
                    message="Failed to update form application status"
                ).model_dump()
                
        except Exception as e:
            logger.error(f"Error marking form as applied: {e}")
            return SupabaseResult(
                status="error",
                message=str(e)
            ).model_dump() 