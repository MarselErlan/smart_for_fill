# smart_form_fill/app/services/form_pipeline.py

"""
Form Pipeline - Orchestrates the complete workflow from analysis to filling
Automatically triggers form_filler.py after form_analyzer.py completes successfully
"""

import os
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime

from app.services.form_analyzer import FormAnalyzer
from app.services.form_filler import FormFiller
from app.services.cache_service import CacheService

class FormPipeline:
    def __init__(self, openai_api_key: str, db_url: str, cache_service: CacheService = None):
        self.cache = cache_service or CacheService()
        
        # Initialize both services
        self.form_analyzer = FormAnalyzer(
            openai_api_key=openai_api_key,
            db_url=db_url,
            cache_service=self.cache
        )
        
        self.form_filler = FormFiller(
            openai_api_key=openai_api_key,
            cache_service=self.cache,
            headless=True
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
            "pipeline_status": "running",
            "steps": {},
            "start_time": pipeline_start.isoformat(),
            "user_data_provided": bool(user_data)
        }
        
        try:
            # Step 1: Form Analysis
            logger.info("Pipeline Step 1: Form Analysis")
            analysis_result = await self.form_analyzer.analyze_form(url, force_refresh)
            
            pipeline_result["steps"]["analysis"] = {
                "status": analysis_result["status"],
                "timestamp": analysis_result.get("timestamp"),
                "cached": not force_refresh and "cached" in str(analysis_result)
            }
            
            if analysis_result["status"] != "success":
                pipeline_result["pipeline_status"] = "failed"
                pipeline_result["error"] = f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"
                return pipeline_result
            
            logger.info("✅ Pipeline Step 1 completed: Form Analysis successful")
            
            # Step 2: Form Filling (Preview or Actual)
            if preview_only:
                logger.info("Pipeline Step 2: Form Fill Preview")
                fill_result = await self.form_filler.preview_auto_fill(url, user_data)
                step_name = "preview"
            else:
                logger.info("Pipeline Step 2: Form Filling")
                fill_result = await self.form_filler.auto_fill_form(url, user_data, submit_form)
                step_name = "filling"
            
            pipeline_result["steps"][step_name] = {
                "status": fill_result["status"],
                "timestamp": datetime.now().isoformat()
            }
            
            if fill_result["status"] != "success":
                pipeline_result["pipeline_status"] = "failed"
                pipeline_result["error"] = f"Form {step_name} failed: {fill_result.get('error', 'Unknown error')}"
                return pipeline_result
            
            # Add specific results based on step type
            if preview_only:
                pipeline_result["steps"]["preview"].update({
                    "total_fields": fill_result.get("total_fields", 0),
                    "fillable_fields": fill_result.get("fillable_fields", 0),
                    "summary": fill_result.get("summary", "")
                })
                logger.info(f"✅ Pipeline Step 2 completed: Preview generated ({fill_result.get('fillable_fields', 0)} fillable fields)")
            else:
                pipeline_result["steps"]["filling"].update({
                    "filled_fields": fill_result.get("filled_fields", 0),
                    "screenshot": fill_result.get("screenshot"),
                    "submit_result": fill_result.get("submit_result")
                })
                logger.info(f"✅ Pipeline Step 2 completed: Form filled ({fill_result.get('filled_fields', 0)} fields)")
                
                # Step 3: Submission (if requested and not already done)
                if submit_form and fill_result.get("submit_result"):
                    submit_status = fill_result["submit_result"]["status"]
                    pipeline_result["steps"]["submission"] = {
                        "status": submit_status,
                        "timestamp": datetime.now().isoformat(),
                        "final_url": fill_result["submit_result"].get("final_url"),
                        "method": fill_result["submit_result"].get("method")
                    }
                    
                    if submit_status == "submitted":
                        logger.info("✅ Pipeline Step 3 completed: Form submitted successfully")
                    else:
                        logger.warning(f"⚠️ Pipeline Step 3: Submission issue - {submit_status}")
            
            # Pipeline completed successfully
            pipeline_end = datetime.now()
            pipeline_result.update({
                "pipeline_status": "completed",
                "end_time": pipeline_end.isoformat(),
                "duration_seconds": (pipeline_end - pipeline_start).total_seconds(),
                "results": fill_result
            })
            
            logger.info(f"🎉 Complete pipeline finished successfully in {pipeline_result['duration_seconds']:.2f}s")
            return pipeline_result
            
        except Exception as e:
            pipeline_result.update({
                "pipeline_status": "error",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            })
            logger.error(f"Pipeline failed with exception: {e}")
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
            
            if result["pipeline_status"] == "completed":
                batch_results["successful"] += 1
            else:
                batch_results["failed"] += 1
        
        logger.info(f"Batch processing completed: {batch_results['successful']}/{batch_results['total_forms']} successful")
        return batch_results 