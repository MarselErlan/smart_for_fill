# smart_form_fill/app/services/form_analyzer.py

"""
Form Analyzer - AI-powered form field detection
Uses GPT-4-turbo to analyze form structure and create mapping for auto-fill
"""

import os
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from loguru import logger
from typing import Dict, Optional, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import DictCursor

from app.services.cache_service import CacheService

class FormAnalyzer:
    def __init__(self, openai_api_key: str, db_url: str, cache_service: CacheService = None):
        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.cache = cache_service or CacheService()
        self.db_url = db_url

    def _get_db_connection(self):
        """Create and return a database connection"""
        return psycopg2.connect(self.db_url, cursor_factory=DictCursor)

    def _create_or_update_form_record(self, url: str, status: str = 'analyzed') -> None:
        """Create or update a form submission record"""
        with self._get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO form_submissions (url, status)
                    VALUES (%s, %s)
                    ON CONFLICT (url) 
                    DO UPDATE SET status = %s, updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                """, (url, status, status))
                conn.commit()
                record_id = cur.fetchone()[0]
                logger.info(f"Created/updated form record {record_id} for {url} with status {status}")

    async def analyze_form(self, url: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze form structure using AI.
        If the URL is in cache and force_refresh is False, returns the cached analysis.
        Otherwise, fetches the HTML, analyzes it, and caches the result.
        """
        cache_key = f"form:{url}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Returning cached analysis for {url} (from Redis).")
                return cached["analysis"]

        logger.info(f"Analyzing form at: {url} (force_refresh={force_refresh})")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                # Load the page
                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(3000)
                form_html = None
                try:
                    form = page.locator("form").first
                    form_html = await form.inner_html()
                except:
                    # If no form found, get full page content
                    form_html = await page.content()
                # Use GPT-4-turbo to analyze the form
                analysis = await self._analyze_with_gpt4(form_html)
                
                if analysis["status"] == "success":
                    # Create or update form record with analyzed status
                    self._create_or_update_form_record(url, "analyzed")
                else:
                    # If analysis failed, mark the form as failed
                    self._create_or_update_form_record(url, "failed")
                
                # Take a screenshot for verification (optional)
                await page.screenshot(path="data/form_snapshot.png")
                await browser.close()
                self.cache.set(cache_key, {"html": form_html, "analysis": analysis}, ttl_seconds=3600)
                return analysis
            except Exception as e:
                logger.error(f"Form analysis failed: {e}")
                # Mark form as failed in case of exception
                self._create_or_update_form_record(url, "failed")
                await browser.close()
                return {"status": "error", "error": str(e)}
    
    def _filter_form_content(self, html: str) -> str:
        """
        🎯 Enhanced form content filtering with better context preservation
        """
        import re
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unnecessary elements that don't contribute to form analysis
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'meta', 'link']):
                tag.decompose()
            
            # Enhanced form-related content extraction
            form_elements = []
            processed_elements = set()
            
            # 1. Find actual form tags with full context
            forms = soup.find_all('form')
            if forms:
                for form in forms:
                    form_elements.append(str(form))
                    processed_elements.add(id(form))
            
            # 2. Find input fields and their contextual containers
            inputs = soup.find_all(['input', 'textarea', 'select', 'button'])
            for inp in inputs:
                if id(inp) in processed_elements:
                    continue
                    
                # Get broader context - look for question text in surrounding elements
                context_element = None
                
                # Strategy 1: Look for parent containers with question text
                for parent_level in ['div', 'section', 'fieldset', 'li', 'tr', 'td']:
                    parent = inp.find_parent(parent_level)
                    if parent and id(parent) not in processed_elements:
                        # Check if this parent contains question-like text
                        parent_text = parent.get_text(strip=True)
                        if len(parent_text) > 5 and ('?' in parent_text or 
                                                   any(word in parent_text.lower() for word in 
                                                       ['what', 'which', 'how', 'do you', 'are you', 'have you', 
                                                        'please', 'enter', 'select', 'choose', 'provide'])):
                            context_element = parent
                            break
                
                # Strategy 2: If no good parent, look for nearby labels or text
                if not context_element:
                    # Look for associated label
                    field_id = inp.get('id')
                    if field_id:
                        label = soup.find('label', {'for': field_id})
                        if label:
                            # Get the container that includes both label and input
                            common_parent = label.find_parent(['div', 'section', 'fieldset'])
                            if common_parent and id(common_parent) not in processed_elements:
                                context_element = common_parent
                
                # Strategy 3: Look for preceding text elements
                if not context_element:
                    # Find the closest container that has meaningful text
                    current = inp
                    for _ in range(3):  # Look up to 3 levels up
                        parent = current.find_parent(['div', 'section', 'fieldset', 'li'])
                        if parent and id(parent) not in processed_elements:
                            parent_text = parent.get_text(strip=True)
                            if len(parent_text) > 10:  # Has substantial text
                                context_element = parent
                                break
                        current = parent if parent else current
                
                # Add the best context element we found
                if context_element:
                    form_elements.append(str(context_element))
                    processed_elements.add(id(context_element))
                else:
                    # Fallback: just add the input with minimal context
                    minimal_parent = inp.find_parent(['div', 'span'])
                    if minimal_parent and id(minimal_parent) not in processed_elements:
                        form_elements.append(str(minimal_parent))
                        processed_elements.add(id(minimal_parent))
                    else:
                        form_elements.append(str(inp))
            
            # 3. Find standalone labels that might have been missed
            labels = soup.find_all('label')
            for label in labels:
                if id(label) not in processed_elements:
                    # Include label with its context
                    label_parent = label.find_parent(['div', 'section', 'fieldset'])
                    if label_parent and id(label_parent) not in processed_elements:
                        form_elements.append(str(label_parent))
                        processed_elements.add(id(label_parent))
                    else:
                        form_elements.append(str(label))
            
            # 4. Look for form containers with question-like patterns
            question_patterns = [
                r'form.*field', r'application.*field', r'question', r'input.*group',
                r'field.*entry', r'form.*section', r'survey.*item'
            ]
            
            for pattern in question_patterns:
                containers = soup.find_all(['div', 'section'], class_=re.compile(pattern, re.I))
                for container in containers[:15]:  # Limit to prevent bloat
                    if id(container) not in processed_elements:
                        container_text = container.get_text(strip=True)
                        # Only include if it has substantial content
                        if len(container_text) > 10:
                            form_elements.append(str(container))
                            processed_elements.add(id(container))
            
            # 5. Look for text elements that contain questions
            question_elements = soup.find_all(text=re.compile(r'\?|what\s+is|which\s+|how\s+|do\s+you|are\s+you|please\s+', re.I))
            for text_element in question_elements[:10]:  # Limit to prevent bloat
                parent = text_element.parent
                if parent and id(parent) not in processed_elements:
                    # Get a reasonable container around the question
                    question_container = parent.find_parent(['div', 'section', 'p', 'span'])
                    if question_container and id(question_container) not in processed_elements:
                        form_elements.append(str(question_container))
                        processed_elements.add(id(question_container))
            
            # Combine and deduplicate
            filtered_html = '\n'.join(form_elements)
            
            # If still too long, truncate intelligently
            if len(filtered_html) > 60000:  # Increased limit for better context
                logger.warning(f"HTML still large ({len(filtered_html)} chars), truncating...")
                # Keep first 45k chars and last 15k chars to preserve structure
                filtered_html = filtered_html[:45000] + "\n...[TRUNCATED]...\n" + filtered_html[-15000:]
            
            logger.info(f"📊 Enhanced HTML filtering: {len(html)} → {len(filtered_html)} characters")
            return filtered_html
            
        except Exception as e:
            logger.warning(f"Enhanced HTML filtering failed: {e}, using simple truncation")
            # Fallback: simple truncation
            return html[:50000] if len(html) > 50000 else html

    def _validate_and_fix_selector(self, selector: str) -> str:
        """
        🔧 Validate and fix CSS selectors, especially UUID-based ones
        """
        import re
        
        # If it's a UUID-like selector, ensure it's properly formatted
        if selector.startswith('#') and len(selector) > 10:
            # Extract the ID part
            id_part = selector[1:]  # Remove the #
            
            # Check if it looks like a UUID (contains hyphens and alphanumeric)
            if re.match(r'^[a-f0-9\-]+$', id_part, re.I):
                # For UUID selectors, try different escaping approaches
                escaped_options = [
                    f'#{id_part}',  # Original
                    f'[id="{id_part}"]',  # Attribute selector
                    f'input[id="{id_part}"]',  # More specific
                    f'*[id="{id_part}"]',  # Universal selector
                ]
                return escaped_options[1]  # Use attribute selector as it's most reliable
        
        return selector

    async def _analyze_with_gpt4(self, html: str) -> Dict[str, Any]:
        """Use GPT-4-turbo to analyze form structure with intelligent content filtering"""
        
        # Filter and reduce HTML content
        filtered_html = self._filter_form_content(html)
        
        prompt = f"""
        You are a form analysis expert. Analyze this HTML form content and create a detailed mapping of fields with contextual understanding.
        
        CRITICAL REQUIREMENTS:
        1. **Extract Question Text**: For each field, find the actual question or label text that describes what the field is asking
        2. **Create Meaningful Purposes**: Create descriptive field purposes based on the question text, NOT the CSS selector
        3. **Separate Purpose from Selector**: The "purpose" field should be human-readable, the "selector" field should be the technical CSS selector
        
        IMPORTANT DISTINCTION:
        - **purpose**: Human-readable description (e.g., "primary_language", "work_experience", "salary_expectations")
        - **selector**: Technical CSS selector for targeting the element (e.g., "#5dd27251-0fb1-4f7b-8489-b68536d46c78", "input[name='email']")
        
        For each field, identify:
        1. Field type (text, email, file, radio, checkbox, select, textarea, etc.)
        2. **Purpose**: Create a meaningful, descriptive purpose based on the question text (NEVER use CSS selectors as purposes)
        3. **Selector**: The actual CSS selector to target the element (keep the technical ID/selector)
        4. **Question Text**: Extract the exact question or label text from the HTML
        5. **Context**: Explain what this field is asking for in plain English
        6. Options available (for radio, checkbox, select fields)
        7. Validation requirements and attributes

        Format your response as a JSON object with the following structure:
        {{
            "fields": [
                {{
                    "type": "field_type",
                    "purpose": "descriptive_purpose_based_on_question_NOT_selector",
                    "selector": "actual_css_selector_or_id",
                    "question_text": "exact_question_or_label_from_html",
                    "context": "what_this_field_is_asking_for_in_plain_english",
                    "options": ["available_options_if_applicable"],
                    "validation": ["validation_rules"],
                    "attributes": {{}}
                }}
            ]
        }}
        
        EXAMPLES of CORRECT purpose extraction:
        - If HTML contains "What is your primary language?" with selector "#abc123" → purpose: "primary_language", selector: "#abc123"
        - If HTML contains "Do you have a secondary language?" with selector "#def456" → purpose: "secondary_language", selector: "#def456"
        - If HTML contains "Upload your resume" with selector "#resume-upload" → purpose: "resume_upload", selector: "#resume-upload"
        - If HTML contains "Full Name" with selector "#name-field" → purpose: "full_name", selector: "#name-field"
        - If HTML contains "Email Address" with selector "#email" → purpose: "email", selector: "#email"
        - If HTML contains "Desired Annual Salary" with selector "#salary123" → purpose: "desired_annual_salary", selector: "#salary123"
        
        EXAMPLES of WRONG purpose extraction:
        - ❌ purpose: "#5dd27251-0fb1-4f7b-8489-b68536d46c78" (this is a selector, not a purpose!)
        - ❌ purpose: "textarea#abc123" (this is a selector, not a purpose!)
        - ❌ purpose: "input[type='text']" (this is a selector, not a purpose!)
        
        Pay special attention to:
        - Language preference questions ("What is your primary language?", "Secondary language?")
        - Work authorization questions and their exact wording
        - Radio button groups and their available options
        - Dropdown/select field options
        - File upload fields (resume, cover letter, etc.)
        - Personal information fields (name, email, phone)
        - Salary and compensation questions
        - Experience and background questions
        - Demographic survey questions
        
        CRITICAL: The "purpose" field must ALWAYS be a human-readable description of what the field is asking for, NEVER a CSS selector or technical identifier!
        
        HTML Content (filtered for form elements):
        {filtered_html}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-0125-preview",  # Using GPT-4-turbo with 128k context
                messages=[
                    {"role": "system", "content": "You are a form analysis expert. Provide responses in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            analysis = response.choices[0].message.content
            
            # Post-process the analysis to fix CSS selectors
            try:
                import json
                parsed_analysis = json.loads(analysis)
                
                # Fix selectors in the fields
                if "fields" in parsed_analysis:
                    for field in parsed_analysis["fields"]:
                        if "selector" in field:
                            original_selector = field["selector"]
                            fixed_selector = self._validate_and_fix_selector(original_selector)
                            field["selector"] = fixed_selector
                            
                            # Log if we made a change
                            if original_selector != fixed_selector:
                                logger.info(f"🔧 Fixed selector: {original_selector} → {fixed_selector}")
                
                # Convert back to JSON string
                analysis = json.dumps(parsed_analysis)
                
            except json.JSONDecodeError:
                logger.warning("Could not parse analysis for selector fixing")
            
            return {
                "status": "success",
                "field_map": analysis,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"GPT-4-turbo analysis failed: {e}")
            return {"status": "error", "error": str(e)}
