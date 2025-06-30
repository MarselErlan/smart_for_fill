# smart_form_fill/app/services/form_pipeline.py

"""
Form Pipeline - Orchestrates the complete workflow from analysis to filling
Automatically triggers form_filler.py after form_analyzer.py completes successfully
"""

import os
from typing import Dict, Any, Optional
from app.services.personal_info_extractor import PersonalInfoExtractor
from app.services.resume_extractor import ResumeExtractor
from loguru import logger
from datetime import datetime
from app.services.form_analyzer import FormAnalyzer
from app.services.form_filler import FormFiller
from app.services.cache_service import CacheService
from app.services.llm_services import extract_question_from_label_html
from app.services.embed_questions_service import embed_question
from dotenv import load_dotenv
import time
import traceback
load_dotenv()

class FormPipeline:
    def __init__(self, openai_api_key: str, db_url: str, cache_service: CacheService = None, headless: bool = True):
        self.cache = cache_service or CacheService()
        
        # Initialize both services
        self.form_analyzer = FormAnalyzer(
            cache_service=self.cache
        )
        
        self.form_filler = FormFiller(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            cache_service=self.cache,
            headless=headless
        )
        
        logger.info("Form Pipeline initialized with analyzer and filler services")
    
    async def run_complete_pipeline(
        self, 
        url: str, 
        user_data: Dict[str, Any], 
        force_refresh: bool = False,
        submit_form: bool = False,
        preview_only: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: analyze → fill → submit (optional)
        
        Args:
            url: Form URL to process
            user_data: User's professional data
            force_refresh: Force re-analysis even if cached
            submit_form: Whether to submit the form after filling
            preview_only: If True, only preview what would be filled
        """
        pipeline_start = datetime.now()
        logger.info(f"Starting complete pipeline for: {url}")
        
        pipeline_result = {
            "url": url,
            "status": "running",
            "steps": {},
            "start_time": pipeline_start.isoformat(),
            "user_data_provided": bool(user_data)
        }
        
        try:
            # Step 1: Analyze Form to get fields with actions
            logger.info("Pipeline Step 1: Full Form Analysis")
            analysis_result = await self.form_analyzer.analyze_form(url, force_refresh)
            
            if analysis_result.get("status") != "success":
                pipeline_result["status"] = "failed"
                pipeline_result["error"] = f"Form analysis failed: {analysis_result.get('error', 'Unknown error')}"
                return pipeline_result

            # Retrieve the full field data from the cache
            cache_key = f"form_analysis:{url}"
            cached_data = self.cache.get(cache_key)
            field_data = cached_data.get("fields", []) if cached_data else []
            
            if not field_data:
                 logger.warning(f"No field data found in cache for {url}. The form may be empty or analysis failed to extract fields.")

            pipeline_result["steps"]["form_analysis"] = {
                "status": "success",
                "field_count": len(field_data),
                "questions": [item.get("question") for item in field_data]
            }
            logger.info(f"Form analysis result: {pipeline_result['steps']['form_analysis']}")
            
            # Step 2: Fill the form in browser (launch Playwright)
            logger.info("Pipeline Step 2: Filling form in browser (Playwright)")
            
            # Call fill_form with the detailed field data from the analysis
            fill_result = await self.form_filler.fill_form(
                url=url, 
                field_map=field_data,
                user_data=user_data,
                submit=False,
                manual_submit=not self.form_filler.headless  # Keep browser open if not headless
            )
            
            if fill_result.get("status") != "success":
                raise Exception(f"Form filling failed: {fill_result.get('error', 'Unknown error')}")

            pipeline_result["steps"]["form_filling"] = fill_result

            pipeline_end = datetime.now()
            pipeline_result.update({
                "status": "success",
                "end_time": pipeline_end.isoformat(),
                "duration_seconds": (pipeline_end - pipeline_start).total_seconds(),
                "results": pipeline_result["steps"]["form_filling"]
            })
            logger.info(f"🎉 Pipeline finished successfully in {pipeline_result['duration_seconds']:.2f}s")
            return pipeline_result
            
        except Exception as e:
            pipeline_end = datetime.now()
            # logger.error(f"❌ Pipeline failed after {end_time - start_time:.2f}s: {e}")
            logger.error(traceback.format_exc())
            # Return a success status to allow processing of partial results
            pipeline_result.update({
                "status": "failed",
                "end_time": pipeline_end.isoformat(),
                "duration_seconds": (pipeline_end - pipeline_start).total_seconds(),
                "error": str(e)
            })
            return pipeline_result
    
    async def analyze_and_preview(self, url: str, user_data: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
        """
        Convenience method: analyze form and preview filling
        """
        return await self.run_complete_pipeline(
            url=url,
            user_data=user_data,
            force_refresh=force_refresh,
            preview_only=True
        )
    
    async def analyze_and_fill(self, url: str, user_data: Dict[str, Any], force_refresh: bool = False, submit: bool = False) -> Dict[str, Any]:
        """
        Convenience method: analyze form and fill it
        """
        return await self.run_complete_pipeline(
            url=url,
            user_data=user_data,
            force_refresh=force_refresh,
            submit_form=submit,
            preview_only=False
        )
    
    async def get_pipeline_status(self, url: str) -> Dict[str, Any]:
        """
        Get the current status of a form in the pipeline
        """
        try:
            # Check cache for analysis
            cache_key = f"form:{url}"
            cached_data = self.cache.get(cache_key)
            
            # Check database for status
            with self.form_analyzer._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT status, created_at, updated_at, applied_counter
                        FROM form_submissions 
                        WHERE url = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """, (url,))
                    db_record = cur.fetchone()
            
            status_info = {
                "url": url,
                "cached_analysis": bool(cached_data),
                "database_status": db_record[0] if db_record else "not_found",
                "last_updated": db_record[2].isoformat() if db_record and db_record[2] else None,
                "applied_count": db_record[3] if db_record else 0,
                "ready_for_filling": bool(cached_data and db_record and db_record[0] == "analyzed")
            }
            
            return {"status": "success", "pipeline_info": status_info}
            
        except Exception as e:
            logger.error(f"Failed to get pipeline status for {url}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def batch_process_forms(self, form_configs: list) -> Dict[str, Any]:
        """
        Process multiple forms in batch
        
        Args:
            form_configs: List of dicts with 'url', 'user_data', and optional 'submit'
        """
        logger.info(f"Starting batch processing of {len(form_configs)} forms")
        
        batch_results = {
            "total_forms": len(form_configs),
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for i, config in enumerate(form_configs):
            logger.info(f"Processing form {i+1}/{len(form_configs)}: {config['url']}")
            
            result = await self.run_complete_pipeline(
                url=config["url"],
                user_data=config["user_data"],
                submit_form=config.get("submit", False)
            )
            
            batch_results["results"].append(result)
            batch_results["processed"] += 1
            
            if result["status"] == "success":
                batch_results["successful"] += 1
            else:
                batch_results["failed"] += 1
        
        logger.info(f"Batch processing completed: {batch_results['successful']}/{batch_results['total_forms']} successful")
        return batch_results 

    def get_user_data_from_sources(self):
        """
        Gathers user data from available sources like vector databases.
        This is a placeholder for a more robust data aggregation logic.
        """
        user_data = {}
        
        # Example: Load from resume vector store
        try:
            resume_extractor = ResumeExtractor()
            if resume_extractor.get_latest_faiss_store_path():
                # In a real scenario, you might extract key info here
                user_data['resume_summary'] = "Resume data is available."
        except Exception as e:
            logger.warning(f"Could not load resume data: {e}")
            
        # Example: Load from personal info vector store
        try:
            personal_info_extractor = PersonalInfoExtractor()
            if personal_info_extractor.get_latest_faiss_store_path():
                user_data['personal_info_summary'] = "Personal info data is available."
        except Exception as e:
            logger.warning(f"Could not load personal info data: {e}")
            
        return user_data 