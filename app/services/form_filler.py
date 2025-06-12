# smart_form_fill/app/services/form_filler.py

"""
Form Filler - AI-powered intelligent form filling service
Uses LLM to understand form context and fill fields professionally
"""

import os
import json
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from loguru import logger
from typing import Dict, Optional, Any, List
from datetime import datetime

class FormFiller:
    def __init__(self, openai_api_key: str, headless: bool = True):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.headless = headless
    
    async def fill_form(self, url: str, field_map: str, user_data: Dict, submit: bool = False) -> Dict[str, Any]:
        """
        Intelligently fill form fields using AI analysis and user data
        
        Args:
            url: Form URL to fill
            field_map: JSON string from form analyzer containing field information
            user_data: User's professional data (resume info, personal details)
            submit: Whether to submit the form after filling
        """
        logger.info(f"Starting intelligent form fill for: {url}")
        
        # Parse field map
        try:
            field_data = json.loads(field_map) if isinstance(field_map, str) else field_map
            if "fields" in field_data:
                fields = field_data["fields"]
            else:
                fields = field_data
        except Exception as e:
            logger.error(f"Failed to parse field map: {e}")
            return {"status": "error", "error": f"Invalid field map: {str(e)}"}
        
        # Generate intelligent field values using LLM
        field_values = await self._generate_field_values(fields, user_data)
        
        if field_values["status"] != "success":
            return field_values
        
        # Fill the form using Playwright
        return await self._fill_form_with_playwright(url, field_values["values"], submit)
    
    async def _generate_field_values(self, fields: List[Dict], user_data: Dict) -> Dict[str, Any]:
        """
        Use LLM to intelligently determine what values to fill in each field
        """
        prompt = f"""
        You are a professional form filler with expertise in job applications and professional forms.
        
        Your task is to analyze form fields and determine the most appropriate values to fill based on the user's professional data.
        
        FORM FIELDS TO ANALYZE:
        {json.dumps(fields, indent=2)}
        
        USER'S PROFESSIONAL DATA:
        {json.dumps(user_data, indent=2)}
        
        INSTRUCTIONS:
        1. For each field, determine the most appropriate value from the user's data
        2. If a field requires professional formatting (like phone numbers, addresses), format appropriately
        3. For file upload fields (resume, cover letter), indicate the file type needed
        4. For dropdown/select fields, choose the most relevant option if possible
        5. For text areas (cover letter, additional info), write professional content
        6. Skip fields that don't have relevant data or are optional
        7. Be professional and accurate - this is for job applications
        
        RESPONSE FORMAT (JSON):
        {{
            "field_mappings": [
                {{
                    "selector": "css_selector_from_field",
                    "field_type": "input_type",
                    "field_purpose": "what_this_field_is_for",
                    "value": "value_to_fill_or_null_if_skip",
                    "action": "fill|upload|select|skip",
                    "reasoning": "why_this_value_was_chosen"
                }}
            ],
            "summary": "brief_summary_of_filling_strategy"
        }}
        
        Be thorough but professional. Only fill fields where you have appropriate data.
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-0125-preview",  # Using GPT-4-turbo for better reasoning
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional form filling expert. You understand job application forms and can intelligently map user data to form fields. Always respond in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent, professional responses
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            logger.info(f"LLM generated field mappings: {result.get('summary', 'No summary')}")
            
            return {
                "status": "success",
                "values": result["field_mappings"],
                "summary": result.get("summary", "")
            }
            
        except Exception as e:
            logger.error(f"LLM field value generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _fill_form_with_playwright(self, url: str, field_mappings: List[Dict], submit: bool = False) -> Dict[str, Any]:
        """
        Use Playwright to actually fill the form with the LLM-generated values
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            
            try:
                # Load the page
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(3000)
                
                results = {}
                filled_count = 0
                
                # Process each field mapping
                for mapping in field_mappings:
                    selector = mapping["selector"]
                    action = mapping["action"]
                    value = mapping["value"]
                    field_purpose = mapping["field_purpose"]
                    
                    try:
                        if action == "skip" or not value:
                            results[field_purpose] = "skipped"
                            continue
                        
                        # Wait for element to be available
                        await page.wait_for_selector(selector, timeout=5000)
                        
                        if action == "fill":
                            await page.fill(selector, str(value))
                            results[field_purpose] = f"filled: {value}"
                            filled_count += 1
                            
                        elif action == "upload":
                            # Handle file uploads (resume, cover letter, etc.)
                            if os.path.exists(str(value)):
                                await page.set_input_files(selector, str(value))
                                results[field_purpose] = f"uploaded: {os.path.basename(str(value))}"
                                filled_count += 1
                            else:
                                results[field_purpose] = f"file not found: {value}"
                                
                        elif action == "select":
                            # Handle dropdown selections
                            await page.select_option(selector, str(value))
                            results[field_purpose] = f"selected: {value}"
                            filled_count += 1
                            
                        # Small delay between fields
                        await page.wait_for_timeout(500)
                        
                    except Exception as e:
                        results[field_purpose] = f"error: {str(e)}"
                        logger.warning(f"Failed to fill field {field_purpose}: {e}")
                
                # Take screenshot of filled form
                screenshot_path = f"data/filled_form_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                await page.screenshot(path=screenshot_path)
                
                # Submit form if requested
                submit_result = None
                if submit and filled_count > 0:
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
                '#submit'
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
