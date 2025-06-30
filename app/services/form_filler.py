# smart_form_fill/app/services/form_filler.py

"""
Form Filler - AI-powered intelligent form filling service
Uses LLM to understand form context and fill fields professionally
Integrates with Redis cache to retrieve form analysis data automatically
"""

import os
import json
from playwright.async_api import async_playwright
from langchain_openai import ChatOpenAI
from loguru import logger
from typing import Dict, Optional, Any, List
from datetime import datetime

from app.services.cache_service import CacheService
from app.services.personal_info_extractor import PersonalInfoExtractor
from app.services.embed_questions_service import embed_question
from langchain_community.vectorstores import FAISS
from app.services.resume_extractor import ResumeExtractor
from app.services.llm_services import (
    create_llm_prompt,
    parse_llm_response,
    generate_professional_value
)

class FormFiller:
    def __init__(self, openai_api_key: str = None, cache_service: CacheService = None, headless: bool = True, keep_browser_open: bool = False):
        self.client = None
        openai_api_base = os.getenv("OPENAI_API_BASE")
        
        if openai_api_base:
            # Use local LLM
            model_name = os.getenv("LOCAL_LLM_MODEL_NAME", "llama3")
            logger.info(f"Using local LLM at {openai_api_base} with model {model_name}")
            self.client = ChatOpenAI(
                api_key=openai_api_key or "local", # Local LLMs may not need a key
                base_url=openai_api_base,
                model=model_name,
                temperature=0.3
            )
        elif openai_api_key:
            # Use OpenAI
            logger.info("Using OpenAI API with model gpt-3.5-turbo")
            self.client = ChatOpenAI(
                api_key=openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.3
            )
        else:
            logger.warning("No API key or local LLM base URL found. LLM functionality will be disabled.")
            
        self.cache = cache_service or CacheService()
        self.headless = headless
        self.keep_browser_open = keep_browser_open
    
    async def fill_form(self, url: str, field_map: List[Dict], user_data: Dict, submit: bool = False, manual_submit: bool = False) -> Dict[str, Any]:
        """
        Intelligently fill form fields using AI analysis and user data
        
        Args:
            url: Form URL to fill
            field_map: A list of dictionaries containing field information from the analyzer
            user_data: User's professional data (resume info, personal details)
            submit: Whether to submit the form automatically
            manual_submit: If True, keeps browser open for manual submission
        """
        logger.info(f"Starting intelligent form fill for: {url}")
        
        # The field_map is now passed as a list of dictionaries
        if "fields" in field_map:
            fields = field_map["fields"]
        else:
            fields = field_map
        
        # Generate intelligent field values using LLM
        field_values = await self._generate_field_values(fields, user_data)
        logger.info(f"Generated field values: {field_values}")
        
        if field_values["status"] != "success":
            return field_values
        
        # The `fill_form` method was updated to pass the correct structure
        # to `_fill_form_with_playwright` so we just pass field_values["values"]
        # which now contains the complete field definitions with the values.
        playwright_result = await self._fill_form_with_playwright(url, field_values["values"], submit, manual_submit)
        logger.info(f"Playwright result: {playwright_result}")
        return playwright_result
    
    async def _generate_field_values(self, fields: List[Dict], user_data: Dict) -> Dict[str, Any]:
        """
        🧠 INTELLIGENT 3-TIER DATA RETRIEVAL SYSTEM
        
        Refactored to centralize embedding and vector search for efficiency.
        """
        logger.info(f"🔍 Starting 3-tier data retrieval for {len(fields)} fields.")
        
        # Step 1: Get unique field purposes and embed them once
        field_purposes = list(set([f.get("field_purpose", f.get("name", "unknown")) for f in fields]))
        
        logger.info(f"📋 Embedding {len(field_purposes)} unique field purposes.")
        purpose_embeddings = {purpose: embed_question(purpose) for purpose in field_purposes}
        logger.info("✅ Embeddings created successfully.")

        # Step 2 & 3: Search vector databases
        resume_context = self._search_vectorstore(
            "resume", purpose_embeddings, ResumeExtractor()
        )
        personal_context = self._search_vectorstore(
            "personal_info", purpose_embeddings, PersonalInfoExtractor()
        )

        # Step 4: Combine all available data
        combined_data = {
            "provided_user_data": user_data,
            "resume_vectordb_data": resume_context,
            "personal_info_vectordb_data": personal_context
        }
        
        # Step 5: Create a preliminary set of values from vector stores
        filled_values = {}
        for purpose in field_purposes:
            if purpose in resume_context:
                filled_values[purpose] = resume_context[purpose]
            elif purpose in personal_context:
                filled_values[purpose] = personal_context[purpose]
            elif user_data.get(purpose):
                filled_values[purpose] = user_data[purpose]
            else:
                filled_values[purpose] = "" # Default to empty

        # Step 6: Use LLM to fill remaining gaps
        if self.client:
            fields_to_generate = [
                field for field in fields 
                if not filled_values.get(field.get("field_purpose", field.get("name", "unknown")))
            ]
            if fields_to_generate:
                logger.info(f"🧠 Using LLM to generate values for {len(fields_to_generate)} fields.")
                try:
                    llm_prompt = create_llm_prompt(fields_to_generate, combined_data)
                    llm_response = self.client.invoke(llm_prompt)
                    llm_values = parse_llm_response(llm_response.content)
                    filled_values.update(llm_values)
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}")
                    for field in fields_to_generate:
                        purpose = field.get("field_purpose", field.get("name", "unknown"))
                        filled_values[purpose] = generate_professional_value(purpose)

        # Step 7: Finalize field values for Playwright
        final_values = []
        for field in fields:
            purpose = field.get("field_purpose", field.get("name", "unknown"))
            field_copy = field.copy()
            field_copy["value"] = filled_values.get(purpose, "")
            
            # Final check for resume upload: if the value is not a valid file, clear it
            if field_copy.get("action") == "upload" and not os.path.exists(str(field_copy["value"])):
                logger.warning(f"Invalid file path for resume upload: '{field_copy['value']}'. Clearing value.")
                field_copy["value"] = ""

            final_values.append(field_copy)
            
        return {"status": "success", "values": final_values}

    def _search_vectorstore(self, store_name: str, purpose_embeddings: Dict[str, List[float]], extractor) -> Dict[str, str]:
        """Helper to search a vector store."""
        context = {}
        try:
            vectorstore = self._load_vectorstore(extractor)
            if vectorstore:
                for purpose, embedding in purpose_embeddings.items():
                    results = vectorstore.similarity_search_by_vector(embedding, k=1)
                    if results:
                        context[purpose] = results[0].page_content
            else:
                logger.warning(f"Vector store '{store_name}' not found or empty.")
        except Exception as e:
            logger.error(f"Error searching '{store_name}' vector store: {e}")
        return context

    def _load_vectorstore(self, extractor):
        """Loads the latest FAISS vector store."""
        vectordb_path = extractor.vectordb_path
        index_file = vectordb_path / "index.json"
        if not index_file.exists():
            return None
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        if not index_data.get("entries"):
            return None
        latest_entry = index_data["entries"][-1]
        faiss_path = latest_entry.get("faiss_store")
        if faiss_path and os.path.exists(faiss_path):
            return FAISS.load_local(faiss_path, extractor.embeddings, allow_dangerous_deserialization=True)
        return None

    async def _fill_form_with_playwright(self, url: str, field_mappings: List[Dict], submit: bool = False, manual_submit: bool = False) -> Dict[str, Any]:
        """
        Use Playwright to actually fill the form with the LLM-generated values
        """
        if manual_submit:
            # For manual submission, don't use context manager to prevent auto-close
            p = await async_playwright().start()
            browser = await p.chromium.launch(
                headless=self.headless,
                slow_mo=1000 if not self.headless else 0
            )
            page = await browser.new_page()
            
            try:
                # Load the page
                logger.info(f"🌐 Loading page: {url}")
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(3000)
                
                results = {}
                filled_count = 0
                
                logger.info(f"🤖 Starting AI-powered form filling...")
                
                # Extract performance metrics for final summary
                total_fields = len(field_mappings)
                resume_data_count = sum(1 for m in field_mappings if m.get("data_source") == "resume_vectordb")
                personal_data_count = sum(1 for m in field_mappings if m.get("data_source") == "personal_info_vectordb")
                generated_count = sum(1 for m in field_mappings if m.get("data_source") == "generated")
                user_provided_count = sum(1 for m in field_mappings if m.get("data_source") == "user_provided")
                
                real_data_count = resume_data_count + personal_data_count + user_provided_count
                real_data_percentage = (real_data_count / total_fields * 100) if total_fields > 0 else 0
                generation_percentage = (generated_count / total_fields * 100) if total_fields > 0 else 0
                
                # Process each field mapping
                for i, mapping in enumerate(field_mappings):
                    selector = mapping["selector"]
                    action = mapping["action"]
                    value = mapping["value"]
                    field_purpose = mapping["field_purpose"]
                    
                    logger.info(f"📝 Filling field {i+1}/{len(field_mappings)}: {field_purpose}")
                    
                    try:
                        # 🍪 AUTO-SKIP COOKIE AND PRIVACY FIELDS (user will handle manually)
                        cookie_keywords = [
                            'cookie', 'privacy', 'consent', 'vendor', 'tracking', 'analytics',
                            'advertisement', 'marketing', 'preference', 'gdpr', 'ccpa',
                            'allow_all', 'reject_all', 'accept_all', 'confirm_cookie',
                            'select_all_vendors', 'apply_filters', 'cancel_filters',
                            'ot-group', 'optanon', 'onetrust'
                        ]
                        
                        is_cookie_field = any(keyword in field_purpose.lower() for keyword in cookie_keywords)
                        is_cookie_selector = any(keyword in selector.lower() for keyword in cookie_keywords)
                        
                        if is_cookie_field or is_cookie_selector:
                            logger.info(f"    ⏩ Skipping cookie/privacy field: {field_purpose}")
                            results[field_purpose] = "skipped_cookie_field"
                            continue
                        
                        element = page.locator(selector).first

                        if action == "upload":
                            if not value or not os.path.exists(str(value)):
                                logger.warning(f"   ⚠️ File not found or value is empty for upload: {value}. Skipping field {field_purpose}.")
                                results[field_purpose] = f"error: file not found at {value}"
                                continue
                            
                            # Force upload to bypass visibility checks for hidden inputs
                            await element.set_input_files(str(value), timeout=10000, force=True)
                            logger.info(f"   📎 Attached file for {field_purpose}: {os.path.basename(str(value))}")
                            results[field_purpose] = f"file_attached: {os.path.basename(str(value))}"
                            filled_count += 1
                        
                        elif action in ["fill", "select", "check", "click"]:
                            await element.scroll_into_view_if_needed()
                            if not self.headless:
                                await element.highlight()

                            if action == "fill":
                                await element.fill(str(value))
                                logger.info(f"   ✅ Filled: {field_purpose} = {value}")
                                results[field_purpose] = f"filled: {value}"
                            elif action == "select":
                                await element.select_option(label=str(value))
                                logger.info(f"   ✅ Selected: {field_purpose} = {value}")
                                results[field_purpose] = f"selected: {value}"
                            elif action == "check":
                                await element.check()
                                logger.info(f"   ✅ Checked: {field_purpose}")
                                results[field_purpose] = "checked"
                            elif action == "click":
                                await element.click()
                                logger.info(f"   ✅ Clicked: {field_purpose}")
                                results[field_purpose] = "clicked"

                            filled_count += 1
                        else:
                            logger.warning(f"   ❓ Unknown or unhandled action '{action}' for field {field_purpose}")
                            results[field_purpose] = f"unknown_action: {action}"
                        
                    except Exception as e:
                        logger.warning(f"   ❌ Failed to process field {field_purpose}: {e}")
                        results[field_purpose] = f"error: {e}"
                
                # Take a final screenshot
                screenshot_path = ""
                
                # Manual submission mode - keep browser open (no screenshot needed)
                logger.info(f"🖱️  Browser kept open for manual submission")
                logger.info(f"   You can now manually review and submit the form")
                logger.info(f"   Close the browser when done")
                logger.info(f"   ⚠️  IMPORTANT: Browser will stay open - DO NOT close this terminal!")
                logger.info(f"   🌐 Browser is running and visible")
                logger.info(f"   🔗 Current URL: {page.url}")
                
                # 🎉 BEAUTIFUL PERFORMANCE METRICS DISPLAY
                logger.info(f"")
                logger.info(f"{'='*60}")
                logger.info(f"📊 OUTSTANDING PERFORMANCE METRICS:")
                logger.info(f"{'='*60}")
                
                if real_data_percentage >= 80:
                    logger.info(f"🏆 {real_data_percentage:.1f}% Real Data Usage - EXCEEDS EXPECTATIONS!")
                elif real_data_percentage >= 60:
                    logger.info(f"✅ {real_data_percentage:.1f}% Real Data Usage - EXCELLENT!")
                elif real_data_percentage >= 40:
                    logger.info(f"👍 {real_data_percentage:.1f}% Real Data Usage - GOOD!")
                else:
                    logger.info(f"⚠️  {real_data_percentage:.1f}% Real Data Usage - NEEDS IMPROVEMENT")
                
                logger.info(f"🤖 Only {generation_percentage:.1f}% Generation - {'MINIMAL AI generation' if generation_percentage <= 20 else 'MODERATE AI generation' if generation_percentage <= 40 else 'HIGH AI generation'}")
                logger.info(f"")
                logger.info(f"📈 DATA SOURCE BREAKDOWN:")
                resume_pct = (resume_data_count/total_fields*100) if total_fields > 0 else 0
                personal_pct = (personal_data_count/total_fields*100) if total_fields > 0 else 0
                user_pct = (user_provided_count/total_fields*100) if total_fields > 0 else 0
                logger.info(f"   📄 Resume Vector DB: {resume_data_count} fields ({resume_pct:.1f}%)")
                logger.info(f"   👤 Personal Info DB: {personal_data_count} fields ({personal_pct:.1f}%)")
                if user_provided_count > 0:
                    logger.info(f"   📝 User Provided: {user_provided_count} fields ({user_pct:.1f}%)")
                logger.info(f"   🤖 AI Generated: {generated_count} fields ({generation_percentage:.1f}%)")
                logger.info(f"")
                logger.info(f"🎯 TOTAL FIELDS PROCESSED: {total_fields}")
                
                # Performance rating
                if real_data_percentage >= 80:
                    logger.info(f"🏆 PERFORMANCE RATING: OUTSTANDING - Maximized authentic data usage!")
                elif real_data_percentage >= 60:
                    logger.info(f"⭐ PERFORMANCE RATING: EXCELLENT - High authentic data usage!")
                elif real_data_percentage >= 40:
                    logger.info(f"✅ PERFORMANCE RATING: GOOD - Moderate authentic data usage")
                else:
                    logger.info(f"📈 PERFORMANCE RATING: IMPROVING - Consider adding more data to vector databases")
                
                logger.info(f"{'='*60}")
                logger.info(f"🎉 FORM FILLING COMPLETED WITH INTELLIGENCE!")
                
                submit_result = {"status": "manual_submission_mode", "message": "Browser kept open for manual submission"}
                
                # DO NOT close browser or playwright - let user handle it manually
                return {
                    "status": "success",
                    "filled_fields": filled_count,
                    "results": results,
                    "submit_result": submit_result,
                    "timestamp": datetime.now().isoformat(),
                    "browser_status": "open_for_manual_submission",
                    "important_note": "Browser is kept open - close manually when done"
                }
                
            except Exception as e:
                logger.error(f"Form filling failed: {e}")
                await browser.close()
                await p.stop()
                return {"status": "error", "error": str(e)}
        
        else:
            # Regular mode with context manager (auto-close)
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.headless,
                    slow_mo=1000 if not self.headless else 0
                )
                page = await browser.new_page()
                
                try:
                    # Load the page
                    logger.info(f"🌐 Loading page: {url}")
                    await page.goto(url, timeout=60000)
                    await page.wait_for_timeout(3000)
                    
                    results = {}
                    filled_count = 0
                    
                    logger.info(f"🤖 Starting AI-powered form filling...")
                    
                    # Extract performance metrics for final summary
                    total_fields = len(field_mappings)
                    resume_data_count = sum(1 for m in field_mappings if m.get("data_source") == "resume_vectordb")
                    personal_data_count = sum(1 for m in field_mappings if m.get("data_source") == "personal_info_vectordb")
                    generated_count = sum(1 for m in field_mappings if m.get("data_source") == "generated")
                    user_provided_count = sum(1 for m in field_mappings if m.get("data_source") == "user_provided")
                    
                    real_data_count = resume_data_count + personal_data_count + user_provided_count
                    real_data_percentage = (real_data_count / total_fields * 100) if total_fields > 0 else 0
                    generation_percentage = (generated_count / total_fields * 100) if total_fields > 0 else 0
                    
                    # Process each field mapping
                    for i, mapping in enumerate(field_mappings):
                        selector = mapping["selector"]
                        action = mapping["action"]
                        value = mapping["value"]
                        field_purpose = mapping["field_purpose"]
                        
                        logger.info(f"📝 Filling field {i+1}/{len(field_mappings)}: {field_purpose}")
                        
                        try:
                            if action == "skip" or not value:
                                results[field_purpose] = "skipped"
                                logger.info(f"   ⏭️  Skipped: {field_purpose}")
                                continue
                            
                            await page.wait_for_selector(selector, timeout=5000)
                            
                            if not self.headless:
                                try:
                                    element = page.locator(selector).first
                                    await element.highlight()
                                    await page.wait_for_timeout(500)
                                except Exception as highlight_error:
                                    logger.debug(f"   ⚠️  Could not highlight field: {highlight_error}")
                            
                            if action == "upload":
                                # Use set_input_files for file uploads, which works on hidden inputs
                                if not value or not os.path.exists(value):
                                    logger.warning(f"   ⚠️ File not found or value is empty for upload: {value}. Skipping field {field_purpose}.")
                                    results[field_purpose] = f"error: file not found at {value}"
                                    continue
                                
                                await page.set_input_files(selector, value)
                                logger.info(f"   📎 Attached file for {field_purpose}: {value}")
                                results[field_purpose] = f"file_attached: {value}"
                                filled_count += 1
                            
                            elif action == "fill":
                                # Standard fill for text inputs, textareas, etc.
                                await page.fill(selector, str(value))
                                results[field_purpose] = f"filled: {value}"
                                filled_count += 1
                                logger.info(f"   ✅ Filled: {field_purpose} = {value}")
                                
                            elif action == "select":
                                # Handle dropdowns/selects
                                await page.select_option(selector, str(value))
                                results[field_purpose] = f"selected: {value}"
                                filled_count += 1
                                logger.info(f"   🔽 Selected: {field_purpose} = {value}")
                                
                            elif action == "check":
                                # Handle checkboxes/radio buttons
                                await page.check(selector)
                                results[field_purpose] = "checked"
                                filled_count += 1
                                logger.info(f"   ☑️  Checked: {field_purpose}")
                                
                            elif action == "radio":
                                radio_selector = f"{selector}[value='{value}']"
                                try:
                                    await page.check(radio_selector)
                                    results[field_purpose] = f"selected: {value}"
                                    filled_count += 1
                                    logger.info(f"   🔘 Selected radio: {field_purpose} = {value}")
                                except:
                                    await page.check(selector)
                                    results[field_purpose] = f"selected (general): {value}"
                                    filled_count += 1
                                    logger.info(f"   🔘 Selected radio (general): {field_purpose}")
                                
                            await page.wait_for_timeout(1000 if not self.headless else 500)
                            
                        except Exception as e:
                            results[field_purpose] = f"error: {str(e)}"
                            logger.warning(f"   ❌ Failed to fill field {field_purpose}: {e}")
                    
                    # Take screenshot
                    screenshot_path = f"data/filled_form_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    await page.screenshot(path=screenshot_path)
                    logger.info(f"📸 Screenshot saved: {screenshot_path}")
                    
                    # 🎉 BEAUTIFUL PERFORMANCE METRICS DISPLAY
                    logger.info(f"")
                    logger.info(f"{'='*60}")
                    logger.info(f"📊 OUTSTANDING PERFORMANCE METRICS:")
                    logger.info(f"{'='*60}")
                    
                    if real_data_percentage >= 80:
                        logger.info(f"🏆 {real_data_percentage:.1f}% Real Data Usage - EXCEEDS EXPECTATIONS!")
                    elif real_data_percentage >= 60:
                        logger.info(f"✅ {real_data_percentage:.1f}% Real Data Usage - EXCELLENT!")
                    elif real_data_percentage >= 40:
                        logger.info(f"👍 {real_data_percentage:.1f}% Real Data Usage - GOOD!")
                    else:
                        logger.info(f"⚠️  {real_data_percentage:.1f}% Real Data Usage - NEEDS IMPROVEMENT")
                    
                    logger.info(f"🤖 Only {generation_percentage:.1f}% Generation - {'MINIMAL AI generation' if generation_percentage <= 20 else 'MODERATE AI generation' if generation_percentage <= 40 else 'HIGH AI generation'}")
                    logger.info(f"")
                    logger.info(f"📈 DATA SOURCE BREAKDOWN:")
                    resume_pct = (resume_data_count/total_fields*100) if total_fields > 0 else 0
                    personal_pct = (personal_data_count/total_fields*100) if total_fields > 0 else 0
                    user_pct = (user_provided_count/total_fields*100) if total_fields > 0 else 0
                    logger.info(f"   📄 Resume Vector DB: {resume_data_count} fields ({resume_pct:.1f}%)")
                    logger.info(f"   👤 Personal Info DB: {personal_data_count} fields ({personal_pct:.1f}%)")
                    if user_provided_count > 0:
                        logger.info(f"   📝 User Provided: {user_provided_count} fields ({user_pct:.1f}%)")
                    logger.info(f"   🤖 AI Generated: {generated_count} fields ({generation_percentage:.1f}%)")
                    logger.info(f"")
                    logger.info(f"🎯 TOTAL FIELDS PROCESSED: {total_fields}")
                    
                    # Performance rating
                    if real_data_percentage >= 80:
                        logger.info(f"🏆 PERFORMANCE RATING: OUTSTANDING - Maximized authentic data usage!")
                    elif real_data_percentage >= 60:
                        logger.info(f"⭐ PERFORMANCE RATING: EXCELLENT - High authentic data usage!")
                    elif real_data_percentage >= 40:
                        logger.info(f"✅ PERFORMANCE RATING: GOOD - Moderate authentic data usage")
                    else:
                        logger.info(f"📈 PERFORMANCE RATING: IMPROVING - Consider adding more data to vector databases")
                    
                    logger.info(f"{'='*60}")
                    logger.info(f"🎉 FORM FILLING COMPLETED WITH INTELLIGENCE!")
                    
                    # Handle submission
                    submit_result = None
                    if submit:
                        logger.info(f"🚀 Attempting automatic submission...")
                        submit_result = await self._submit_form(page)
                    
                    await browser.close()
                    
                    return {
                        "status": "success",
                        "filled_fields": filled_count,
                        "results": results,
                        "screenshot": screenshot_path,
                        "submit_result": submit_result,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    logger.error(f"Form filling failed: {e}")
                    await browser.close()
                    return {"status": "error", "error": str(e)}
    
    async def _submit_form(self, page) -> Dict[str, Any]:
        """
        Attempt to submit the form after filling
        """
        try:
            # Look for common submit button selectors
            submit_selectors = [
                'input[type="submit"]',
                'button[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Apply")',
                'button:has-text("Send")',
                '.submit-btn',
                '#submit',
                '.btn-primary',
                '.apply-btn'
            ]
            
            for selector in submit_selectors:
                try:
                    submit_btn = page.locator(selector).first
                    if await submit_btn.is_visible():
                        await submit_btn.click()
                        await page.wait_for_timeout(3000)  # Wait for submission
                        
                        # Check if we're on a success page or if URL changed
                        current_url = page.url
                        return {
                            "status": "submitted",
                            "final_url": current_url,
                            "method": f"clicked {selector}"
                        }
                except:
                    continue
            
            return {"status": "no_submit_button_found"}
            
        except Exception as e:
            return {"status": "submit_error", "error": str(e)}
    
    async def preview_auto_fill(self, url: str, user_data: Dict) -> Dict[str, Any]:
        """
        Preview what values would be filled by automatically retrieving form analysis from cache
        """
        logger.info(f"Previewing auto-fill for: {url}")
        
        # Get form analysis from Redis cache
        cache_key = f"form:{url}"
        cached_data = self.cache.get(cache_key)
        
        if not cached_data:
            return {
                "status": "error", 
                "error": f"No form analysis found in cache for {url}. Please analyze the form first using /api/analyze-form"
            }
        
        # Extract field map from cached analysis
        analysis = cached_data.get("analysis", {})
        if analysis.get("status") != "success":
            return {
                "status": "error",
                "error": f"Cached form analysis failed: {analysis.get('error', 'Unknown error')}"
            }
        
        field_map = analysis.get("field_map")
        if not field_map:
            return {
                "status": "error",
                "error": "No field map found in cached analysis"
            }
        
        # Use existing preview method
        return await self.preview_form_filling(url, field_map, user_data)
    
    async def preview_form_filling(self, url: str, field_map: str, user_data: Dict) -> Dict[str, Any]:
        """
        Preview what values would be filled without actually filling the form
        """
        try:
            field_data = json.loads(field_map) if isinstance(field_map, str) else field_map
            if "fields" in field_data:
                fields = field_data["fields"]
            else:
                fields = field_data
        except Exception as e:
            return {"status": "error", "error": f"Invalid field map: {str(e)}"}
        
        # Generate field values using LLM
        field_values = await self._generate_field_values(fields, user_data)
        
        if field_values["status"] == "success":
            return {
                "status": "success",
                "preview": field_values["values"],
                "summary": field_values["summary"],
                "total_fields": len(field_values["values"]),
                "fillable_fields": len([f for f in field_values["values"] if f["action"] != "skip"])
            }
        
        return field_values
