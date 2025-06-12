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

    async def _analyze_with_gpt4(self, html: str) -> Dict[str, Any]:
        """Use GPT-4-turbo to analyze form structure with larger context window"""
        prompt = f"""
        You are a form analysis expert. Analyze this HTML form and create a detailed mapping of fields.
        For each field, identify:
        1. Field type (text, email, file, etc.)
        2. Purpose (name, email, phone, resume, etc.)
        3. Best CSS selector to target it
        4. Any validation requirements
        5. Any special attributes or requirements

        Format your response as a JSON object with the following structure:
        {{
            "fields": [
                {{
                    "type": "field_type",
                    "purpose": "field_purpose",
                    "selector": "css_selector",
                    "validation": ["validation_rules"],
                    "attributes": {{}}
                }}
            ]
        }}

        HTML:
        {html}  # Using full HTML since GPT-4-turbo has larger context
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
