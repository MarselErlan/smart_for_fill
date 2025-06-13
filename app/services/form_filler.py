# smart_form_fill/app/services/form_filler.py

"""
Form Filler - AI-powered intelligent form filling service
Uses LLM to understand form context and fill fields professionally
Integrates with Redis cache to retrieve form analysis data automatically
"""

import os
import json
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from loguru import logger
from typing import Dict, Optional, Any, List
from datetime import datetime

from app.services.cache_service import CacheService

class FormFiller:
    def __init__(self, openai_api_key: str, cache_service: CacheService = None, headless: bool = True, keep_browser_open: bool = False):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.cache = cache_service or CacheService()
        self.headless = headless
        self.keep_browser_open = keep_browser_open
    
    async def auto_fill_form(self, url: str, user_data: Dict, submit: bool = False, manual_submit: bool = False) -> Dict[str, Any]:
        """
        Automatically fill form by retrieving analysis from Redis cache
        
        Args:
            url: Form URL to fill
            user_data: User's professional data (resume info, personal details)
            submit: Whether to submit the form automatically
            manual_submit: If True, keeps browser open for manual submission
        """
        logger.info(f"Starting auto-fill for: {url}")
        
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
        
        logger.info(f"Retrieved form analysis from Redis cache for {url}")
        
        # Use the existing fill_form method with cached data
        return await self.fill_form(url, field_map, user_data, submit, manual_submit)
    
    async def fill_form(self, url: str, field_map: str, user_data: Dict, submit: bool = False, manual_submit: bool = False) -> Dict[str, Any]:
        """
        Intelligently fill form fields using AI analysis and user data
        
        Args:
            url: Form URL to fill
            field_map: JSON string from form analyzer containing field information
            user_data: User's professional data (resume info, personal details)
            submit: Whether to submit the form automatically
            manual_submit: If True, keeps browser open for manual submission
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
        return await self._fill_form_with_playwright(url, field_values["values"], submit, manual_submit)
    
    async def _generate_field_values(self, fields: List[Dict], user_data: Dict) -> Dict[str, Any]:
        """
        🧠 INTELLIGENT 3-TIER DATA RETRIEVAL SYSTEM
        
        1. First: Search resume/vectordb for relevant professional information
        2. Second: Search info/vectordb for personal details and preferences  
        3. Third: Generate professional content if still insufficient
        
        This ensures we use real user data when available, then generate intelligently
        """
        
        # Step 1: Extract field purposes to understand what we need
        field_purposes = []
        for field in fields:
            purpose = field.get("field_purpose", field.get("name", field.get("selector", "unknown")))
            field_purposes.append(purpose)
        
        logger.info(f"🔍 Starting 3-tier intelligent data retrieval for {len(fields)} fields")
        logger.info(f"📋 Field purposes: {', '.join(field_purposes)}")
        
        # Step 2: Search resume vector database for professional information
        resume_data = await self._search_resume_vectordb(field_purposes)
        
        # Step 3: Search personal info vector database for additional details
        personal_data = await self._search_personal_info_vectordb(field_purposes)
        
        # Step 4: Combine all available data
        combined_data = {
            "provided_user_data": user_data,
            "resume_vectordb_data": resume_data,
            "personal_info_vectordb_data": personal_data
        }
        
        # Step 5: Assess data completeness and quality
        data_assessment = self._assess_data_completeness(fields, combined_data)
        
        logger.info(f"📊 Data Assessment: {data_assessment['summary']}")
        
        # If user_data is minimal or empty, let LLM know to generate professional data
        has_minimal_data = not user_data or len(user_data) <= 3
        
        prompt = f"""
        You are a professional form filler with expertise in job applications and professional forms.
        You have advanced contextual understanding and can generate intelligent, professional responses.
        
        🧠 INTELLIGENT DATA PRIORITY SYSTEM:
        1. **FIRST PRIORITY**: Use data from resume vector database (real professional experience)
        2. **SECOND PRIORITY**: Use data from personal info vector database (real personal details)  
        3. **THIRD PRIORITY**: Generate professional content only if real data is insufficient
        
        DATA COMPLETENESS ASSESSMENT:
        {data_assessment['summary']}
        
        FORM FIELDS TO ANALYZE:
        {json.dumps(fields, indent=2)}
        
        AVAILABLE DATA SOURCES (prioritized):
        
        1. RESUME VECTOR DATABASE DATA (HIGHEST PRIORITY - USE FIRST):
        {json.dumps(resume_data, indent=2) if resume_data else "No resume data found"}
        
        2. PERSONAL INFO VECTOR DATABASE DATA (SECOND PRIORITY):
        {json.dumps(personal_data, indent=2) if personal_data else "No personal info data found"}
        
        3. PROVIDED USER DATA (THIRD PRIORITY):
        {json.dumps(user_data, indent=2) if user_data else "No user data provided"}
        
        INTELLIGENT FILLING INSTRUCTIONS:
        - **ALWAYS prioritize real data from vector databases over generated content**
        - **Use resume data for**: work experience, skills, education, professional background
        - **Use personal info data for**: contact details, work authorization, salary expectations, preferences
        - **Only generate content when**: real data is missing or insufficient for specific fields
        - **Maintain consistency**: Ensure all generated content aligns with real data found
        
        ADVANCED INSTRUCTIONS:
        1. **Contextual Analysis**: Analyze each field's purpose and the job/form context
        2. **Professional Generation**: Generate realistic, professional information when user data is minimal
        3. **Job-Relevant Responses**: Create responses that match the job requirements and industry
        4. **Intelligent Reasoning**: Think through the best response based on form context and job type
        5. **Work Authorization**: Generate appropriate work authorization responses for the target country
        6. **Professional Formatting**: Format all data professionally (phone numbers, addresses, etc.)
        7. **Realistic Details**: Create believable professional profiles with consistent information
        8. **Industry Alignment**: Match generated content to the job industry and role requirements
        9. **Geographic Relevance**: Generate location-appropriate information
        10. **Professional Writing**: Write compelling cover letters and professional summaries
        
        GENERATION GUIDELINES (when user data is minimal):
        - **Name**: Generate a professional, realistic name
        - **Contact Info**: Create realistic email, phone number for the target region
        - **Location**: Generate appropriate location based on job location/requirements
        - **Professional Background**: Create relevant work experience and company names
        - **Skills**: Generate skills that match the job requirements
        - **URLs**: Create realistic professional URLs (LinkedIn, GitHub, portfolio)
        - **Work Authorization**: Provide appropriate responses based on job location
        - **Salary**: Generate realistic salary expectations for the role and location
        - **Cover Letter**: Write compelling, job-specific cover letters
        - **Additional Info**: Create relevant professional summaries
        
        SPECIAL HANDLING FOR COMMON FORM PATTERNS:
        - **Work Authorization**: Analyze job location and provide appropriate visa/sponsorship responses
        - **Salary Questions**: Research typical salaries for the role and location
        - **Cover Letters**: Write personalized cover letters that highlight relevant experience for the specific job
        - **Additional Info**: Generate professional summaries that match the job requirements
        - **Hybrid Work**: Provide positive responses about work flexibility
        - **Demographics**: Handle optional demographic questions appropriately (can skip or decline)
        
        RESPONSE FORMAT (JSON):
        {{
            "field_mappings": [
                {{
                    "selector": "css_selector_from_field",
                    "field_type": "input_type",
                    "field_purpose": "what_this_field_is_for",
                    "value": "intelligent_value_prioritizing_real_data_or_null_if_skip",
                    "action": "fill|upload|select|skip|check|radio",
                    "data_source": "resume_vectordb|personal_info_vectordb|generated|user_provided",
                    "reasoning": "detailed_explanation_of_data_source_and_value_selection_logic"
                }}
            ],
            "data_usage_summary": {{
                "resume_data_used": "count_of_fields_using_resume_data",
                "personal_data_used": "count_of_fields_using_personal_data", 
                "generated_content": "count_of_fields_requiring_generation",
                "data_quality": "assessment_of_overall_data_completeness"
            }},
            "summary": "comprehensive_summary_emphasizing_real_data_usage_vs_generation"
        }}
        
        🎯 CRITICAL SUCCESS FACTORS:
        1. **Maximize real data usage** - Extract every possible detail from vector databases
        2. **Minimize generation** - Only generate when absolutely necessary
        3. **Maintain consistency** - Ensure generated content aligns with real data
        4. **Professional quality** - All content must be job-application ready
        5. **Data transparency** - Clearly indicate data sources in reasoning
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-0125-preview",  # Using GPT-4-turbo for better reasoning
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional form filling expert with advanced contextual understanding and creative generation capabilities. You can analyze job forms and generate realistic, professional information that creates strong job applications. You understand job markets, professional communication, and can create compelling professional profiles. Always respond in valid JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Slightly higher temperature for creative generation while maintaining professionalism
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Log enhanced summary with data usage statistics
            data_usage = result.get("data_usage_summary", {})
            summary = result.get("summary", "No summary")
            
            logger.info(f"🧠 LLM field mapping completed:")
            logger.info(f"   📊 Data Usage: Resume={data_usage.get('resume_data_used', 0)}, Personal={data_usage.get('personal_data_used', 0)}, Generated={data_usage.get('generated_content', 0)}")
            logger.info(f"   📝 Summary: {summary}")
            
            return {
                "status": "success",
                "values": result["field_mappings"],
                "summary": summary,
                "data_usage": data_usage,
                "total_fields": len(result["field_mappings"]),
                "data_sources_used": {
                    "resume_vectordb": data_usage.get('resume_data_used', 0),
                    "personal_info_vectordb": data_usage.get('personal_data_used', 0),
                    "generated": data_usage.get('generated_content', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"LLM field value generation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _search_resume_vectordb(self, field_purposes: List[str]) -> Dict[str, Any]:
        """
        🎯 Search resume vector database for relevant professional information
        """
        try:
            from resume_extractor import ResumeExtractor
            
            resume_extractor = ResumeExtractor()
            
            # Create intelligent search queries based on field purposes
            search_queries = []
            
            # Map field purposes to relevant search terms
            purpose_mapping = {
                "name": ["name", "full name", "contact information"],
                "email": ["email", "contact", "email address"],
                "phone": ["phone", "contact", "phone number", "mobile"],
                "location": ["location", "address", "city", "residence"],
                "company": ["company", "employer", "work experience", "current job"],
                "position": ["position", "title", "role", "job title", "current position"],
                "experience": ["experience", "work history", "professional background", "career"],
                "skills": ["skills", "technical skills", "expertise", "technologies"],
                "education": ["education", "degree", "university", "school", "academic"],
                "linkedin": ["linkedin", "profile", "social media"],
                "github": ["github", "portfolio", "projects", "code"],
                "portfolio": ["portfolio", "website", "projects", "work samples"],
                "salary": ["salary", "compensation", "pay", "income"],
                "work_authorization": ["work authorization", "visa", "citizenship", "eligibility"],
                "cover_letter": ["summary", "objective", "about", "professional summary"],
                "additional_info": ["achievements", "accomplishments", "highlights", "summary"]
            }
            
            # Generate search queries based on field purposes
            for purpose in field_purposes:
                purpose_lower = purpose.lower()
                for key, terms in purpose_mapping.items():
                    if any(term in purpose_lower for term in [key] + terms):
                        search_queries.extend(terms[:2])  # Take top 2 relevant terms
                        break
            
            # Remove duplicates and limit queries
            search_queries = list(set(search_queries))[:5]
            
            if not search_queries:
                search_queries = ["professional experience", "work history", "skills", "education"]
            
            logger.info(f"🔍 Resume search queries: {search_queries}")
            
            # Perform searches and combine results
            all_results = []
            for query in search_queries:
                try:
                    results = resume_extractor.search_resume(query, k=3)
                    if results:
                        all_results.extend(results)
                except Exception as e:
                    logger.warning(f"Resume search failed for '{query}': {e}")
            
            # Process and structure the results
            if all_results:
                # Combine content and remove duplicates
                unique_content = []
                seen_content = set()
                
                for result in all_results:
                    # Handle both dict and string results
                    if isinstance(result, dict):
                        content = result.get("content", "").strip()
                    elif isinstance(result, str):
                        content = result.strip()
                    else:
                        content = str(result).strip()
                    
                    if content and content not in seen_content:
                        unique_content.append(content)
                        seen_content.add(content)
                
                resume_data = {
                    "status": "found",
                    "total_results": len(all_results),
                    "unique_content_pieces": len(unique_content),
                    "content": " ".join(unique_content[:10]),  # Limit content length
                    "search_queries_used": search_queries,
                    "data_source": "resume_vectordb"
                }
                
                logger.info(f"✅ Resume data found: {len(unique_content)} unique pieces")
                return resume_data
            else:
                logger.warning("⚠️ No resume data found in vector database")
                return {"status": "not_found", "message": "No resume data available"}
                
        except Exception as e:
            logger.error(f"❌ Resume vector search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _search_personal_info_vectordb(self, field_purposes: List[str]) -> Dict[str, Any]:
        """
        🎯 Search personal info vector database for contact details and preferences
        """
        try:
            from personal_info_extractor import PersonalInfoExtractor
            
            personal_extractor = PersonalInfoExtractor()
            
            # Create targeted search queries for personal information
            personal_queries = []
            
            # Map field purposes to personal info search terms
            personal_mapping = {
                "contact": ["contact information", "email", "phone", "address"],
                "location": ["location", "address", "city", "residence", "current location"],
                "work_authorization": ["work authorization", "visa status", "sponsorship", "eligibility"],
                "salary": ["salary expectations", "compensation", "desired salary", "pay range"],
                "preferences": ["work preferences", "remote work", "hybrid", "availability"],
                "personal": ["personal information", "about me", "background"],
                "additional": ["additional information", "notes", "preferences", "comments"]
            }
            
            # Generate search queries
            for purpose in field_purposes:
                purpose_lower = purpose.lower()
                for key, terms in personal_mapping.items():
                    if any(term in purpose_lower for term in [key] + terms):
                        personal_queries.extend(terms[:2])
                        break
            
            # Remove duplicates and add default queries
            personal_queries = list(set(personal_queries))[:5]
            
            if not personal_queries:
                personal_queries = ["contact information", "work authorization", "salary expectations"]
            
            logger.info(f"🔍 Personal info search queries: {personal_queries}")
            
            # Perform searches
            all_results = []
            for query in personal_queries:
                try:
                    results = personal_extractor.search_personal_info(query, k=3)
                    if results and "results" in results:
                        all_results.extend(results["results"])
                except Exception as e:
                    logger.warning(f"Personal info search failed for '{query}': {e}")
            
            # Process results
            if all_results:
                unique_content = []
                seen_content = set()
                
                for result in all_results:
                    content = result.get("content", "").strip()
                    if content and content not in seen_content:
                        unique_content.append(content)
                        seen_content.add(content)
                
                personal_data = {
                    "status": "found",
                    "total_results": len(all_results),
                    "unique_content_pieces": len(unique_content),
                    "content": " ".join(unique_content[:8]),  # Limit content length
                    "search_queries_used": personal_queries,
                    "data_source": "personal_info_vectordb"
                }
                
                logger.info(f"✅ Personal info data found: {len(unique_content)} unique pieces")
                return personal_data
            else:
                logger.warning("⚠️ No personal info data found in vector database")
                return {"status": "not_found", "message": "No personal info data available"}
                
        except Exception as e:
            logger.error(f"❌ Personal info vector search failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _assess_data_completeness(self, fields: List[Dict], combined_data: Dict) -> Dict[str, Any]:
        """
        📊 Assess the completeness and quality of available data
        """
        try:
            resume_data = combined_data.get("resume_vectordb_data", {})
            personal_data = combined_data.get("personal_info_vectordb_data", {})
            user_data = combined_data.get("provided_user_data", {})
            
            # Count available data sources
            sources_available = 0
            data_quality = []
            
            if resume_data.get("status") == "found":
                sources_available += 1
                data_quality.append(f"Resume: {resume_data.get('unique_content_pieces', 0)} pieces")
            
            if personal_data.get("status") == "found":
                sources_available += 1
                data_quality.append(f"Personal: {personal_data.get('unique_content_pieces', 0)} pieces")
            
            if user_data:
                sources_available += 1
                data_quality.append(f"User: {len(user_data)} fields")
            
            # Determine data sufficiency
            if sources_available >= 2:
                sufficiency = "EXCELLENT - Multiple data sources available"
                generation_needed = "MINIMAL"
            elif sources_available == 1:
                sufficiency = "GOOD - One data source available"
                generation_needed = "MODERATE"
            else:
                sufficiency = "LIMITED - No vector data found"
                generation_needed = "EXTENSIVE"
            
            assessment = {
                "sources_available": sources_available,
                "data_quality": data_quality,
                "sufficiency": sufficiency,
                "generation_needed": generation_needed,
                "summary": f"{sufficiency} | Generation needed: {generation_needed} | Sources: {', '.join(data_quality) if data_quality else 'None'}"
            }
            
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Data assessment failed: {e}")
            return {
                "sources_available": 0,
                "sufficiency": "ERROR",
                "generation_needed": "FULL",
                "summary": f"Assessment failed: {str(e)}"
            }
    
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
                        
                        # Wait for element to be available
                        await page.wait_for_selector(selector, timeout=5000)
                        
                        # Highlight the field being filled (visual feedback)
                        if not self.headless:
                            try:
                                element = page.locator(selector).first
                                await element.highlight()
                                await page.wait_for_timeout(500)
                            except Exception as highlight_error:
                                logger.debug(f"   ⚠️  Could not highlight field: {highlight_error}")
                        
                        if action == "fill":
                            await page.fill(selector, str(value))
                            results[field_purpose] = f"filled: {value}"
                            filled_count += 1
                            logger.info(f"   ✅ Filled: {field_purpose} = {value}")
                            
                        elif action == "upload":
                            if os.path.exists(str(value)):
                                await page.set_input_files(selector, str(value))
                                results[field_purpose] = f"uploaded: {os.path.basename(str(value))}"
                                filled_count += 1
                                logger.info(f"   📎 Uploaded: {os.path.basename(str(value))}")
                            else:
                                results[field_purpose] = f"file not found: {value}"
                                logger.warning(f"   ❌ File not found: {value}")
                                
                        elif action == "select":
                            await page.select_option(selector, str(value))
                            results[field_purpose] = f"selected: {value}"
                            filled_count += 1
                            logger.info(f"   🔽 Selected: {field_purpose} = {value}")
                            
                        elif action == "check":
                            if str(value).lower() in ['true', '1', 'yes', 'on']:
                                await page.check(selector)
                                results[field_purpose] = "checked"
                                filled_count += 1
                                logger.info(f"   ☑️  Checked: {field_purpose}")
                            else:
                                await page.uncheck(selector)
                                results[field_purpose] = "unchecked"
                                logger.info(f"   ☐ Unchecked: {field_purpose}")
                                
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
                            
                            # Small delay between fields for visual effect
                            await page.wait_for_timeout(1000 if not self.headless else 500)
                            
                    except Exception as e:
                        results[field_purpose] = f"error: {str(e)}"
                        logger.warning(f"   ❌ Failed to fill field {field_purpose}: {e}")
                
                # Manual submission mode - keep browser open (no screenshot needed)
                logger.info(f"🖱️  Browser kept open for manual submission")
                logger.info(f"   You can now manually review and submit the form")
                logger.info(f"   Close the browser when done")
                logger.info(f"   ⚠️  IMPORTANT: Browser will stay open - DO NOT close this terminal!")
                logger.info(f"   🌐 Browser is running and visible")
                logger.info(f"   🔗 Current URL: {page.url}")
                
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
                            
                            if action == "fill":
                                await page.fill(selector, str(value))
                                results[field_purpose] = f"filled: {value}"
                                filled_count += 1
                                logger.info(f"   ✅ Filled: {field_purpose} = {value}")
                                
                            elif action == "upload":
                                if os.path.exists(str(value)):
                                    await page.set_input_files(selector, str(value))
                                    results[field_purpose] = f"uploaded: {os.path.basename(str(value))}"
                                    filled_count += 1
                                    logger.info(f"   📎 Uploaded: {os.path.basename(str(value))}")
                                else:
                                    results[field_purpose] = f"file not found: {value}"
                                    logger.warning(f"   ❌ File not found: {value}")
                                    
                            elif action == "select":
                                await page.select_option(selector, str(value))
                                results[field_purpose] = f"selected: {value}"
                                filled_count += 1
                                logger.info(f"   🔽 Selected: {field_purpose} = {value}")
                                
                            elif action == "check":
                                if str(value).lower() in ['true', '1', 'yes', 'on']:
                                    await page.check(selector)
                                    results[field_purpose] = "checked"
                                    filled_count += 1
                                    logger.info(f"   ☑️  Checked: {field_purpose}")
                                else:
                                    await page.uncheck(selector)
                                    results[field_purpose] = "unchecked"
                                    logger.info(f"   ☐ Unchecked: {field_purpose}")
                                    
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
