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
        🎯 Filter HTML content to focus on form-related elements and reduce token count
        """
        import re
        from bs4 import BeautifulSoup
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unnecessary elements that don't contribute to form analysis
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'meta', 'link']):
                tag.decompose()
            
            # Focus on form-related content
            form_elements = []
            
            # 1. Find actual form tags
            forms = soup.find_all('form')
            if forms:
                for form in forms:
                    form_elements.append(str(form))
            
            # 2. Find input fields outside forms (common in modern SPAs)
            inputs = soup.find_all(['input', 'textarea', 'select', 'button'])
            for inp in inputs:
                # Get parent context for better understanding
                parent = inp.find_parent(['div', 'section', 'fieldset', 'form'])
                if parent and str(parent) not in form_elements:
                    form_elements.append(str(parent))
            
            # 3. Find labels and form-related text
            labels = soup.find_all('label')
            for label in labels:
                form_elements.append(str(label))
            
            # 4. Look for common form containers
            form_containers = soup.find_all(['div', 'section'], class_=re.compile(r'form|field|input|application', re.I))
            for container in form_containers[:10]:  # Limit to first 10 to avoid bloat
                form_elements.append(str(container))
            
            # Combine and deduplicate
            filtered_html = '\n'.join(set(form_elements))
            
            # If still too long, truncate intelligently
            if len(filtered_html) > 50000:  # ~25k tokens max
                logger.warning(f"HTML still large ({len(filtered_html)} chars), truncating...")
                # Keep first 40k chars and last 10k chars to preserve structure
                filtered_html = filtered_html[:40000] + "\n...[TRUNCATED]...\n" + filtered_html[-10000:]
            
            logger.info(f"📊 HTML filtered: {len(html)} → {len(filtered_html)} characters")
            return filtered_html
            
        except Exception as e:
            logger.warning(f"HTML filtering failed: {e}, using truncated original")
            # Fallback: simple truncation
            return html[:50000] if len(html) > 50000 else html

    async def _analyze_with_gpt4(self, html: str) -> Dict[str, Any]:
        """Use GPT-4-turbo to analyze form structure with intelligent content filtering"""
        
        # Filter and reduce HTML content
        filtered_html = self._filter_form_content(html)
        
        prompt = f"""
        You are a form analysis expert. Analyze this HTML form content and create a detailed mapping of fields with contextual understanding.
        
        For each field, identify:
        1. Field type (text, email, file, radio, checkbox, select, textarea, etc.)
        2. Purpose (name, email, phone, resume, work_authorization, salary, etc.)
        3. Best CSS selector to target it
        4. Any validation requirements
        5. Any special attributes or requirements
        6. **Context and Question Text**: Extract the actual question or label text associated with the field
        7. **Options Available**: For radio buttons, checkboxes, and select fields, list available options
        8. **Field Context**: Understand what the field is asking and provide context

        Format your response as a JSON object with the following structure:
        {{
            "fields": [
                {{
                    "type": "field_type",
                    "purpose": "field_purpose",
                    "selector": "css_selector",
                    "validation": ["validation_rules"],
                    "attributes": {{}},
                    "question_text": "actual_question_or_label_from_form",
                    "options": ["available_options_if_applicable"],
                    "context": "what_this_field_is_asking_for"
                }}
            ]
        }}
        
        Pay special attention to:
        - Work authorization questions and their exact wording
        - Radio button groups and their available options
        - Dropdown/select field options
        - Complex textarea questions that need contextual understanding
        - Salary and compensation questions
        - Demographic survey questions
        
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
            return {
                "status": "success",
                "field_map": analysis,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"GPT-4-turbo analysis failed: {e}")
            return {"status": "error", "error": str(e)}
