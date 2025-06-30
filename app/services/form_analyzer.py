# smart_form_fill/app/services/form_analyzer.py

"""
Form Analyzer - AI-powered form field detection
Uses GPT-4-turbo to analyze form structure and create mapping for auto-fill
"""

import os
from playwright.async_api import async_playwright
from loguru import logger
from typing import Dict, Optional, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import DictCursor
from bs4 import BeautifulSoup

from app.services.cache_service import CacheService

class FormAnalyzer:
    def __init__(self, cache_service: CacheService = None):
        self.cache = cache_service or CacheService()

    async def analyze_form(self, url: str, force_refresh: bool = False) -> dict:
        """
        Analyzes a form to extract fields, labels, and actions.
        It uses Playwright to get the page content and BeautifulSoup to parse it.
        """
        cache_key = f"form_analysis:{url}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Returning cached form analysis for {url}.")
                return cached
        
        logger.info(f"Analyzing form at: {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url)
            
            # Wait for dynamic content to load and perform a reload just in case
            await page.wait_for_timeout(3000)
            await page.reload()
            await page.wait_for_timeout(2000)

            content = await page.content()
            await browser.close()

        soup = BeautifulSoup(content, 'html.parser')
        fields = []
        
        # Find all labels and their associated inputs
        for label in soup.find_all('label'):
            field_id = label.get('for')
            if not field_id:
                continue

            question = label.text.strip()
            input_el = soup.find(id=field_id)
            if not input_el:
                continue

            action = "fill" # Default action
            input_type = input_el.get('type')
            
            if input_el.name == 'select':
                action = 'select'
            elif input_el.name == 'textarea':
                action = 'fill'
            elif input_type in ['text', 'email', 'password', 'tel', 'url', 'search', 'number']:
                action = 'fill'
            elif input_type == 'file':
                action = 'upload'
            elif input_type == 'checkbox':
                action = 'check'
            elif input_type == 'radio':
                action = 'radio'
            else:
                action = 'skip' # Skip unknown types

            fields.append({
                "name": field_id,
                "question": question,
                "selector": f'[id="{field_id}"]',
                "field_purpose": question,
                "action": action
            })

        result = {
            "status": "success", 
            "field_count": len(fields),
            "questions": [f["question"] for f in fields]
        }
        
        # Cache the raw field data, not the summary
        self.cache.set(cache_key, {"status": "success", "fields": fields}, ttl_seconds=3600)
        logger.info(f"Analyzed {len(fields)} fields from {url}")
        
        return result

    async def analyze_labels_fast(self, url: str, force_refresh: bool = False) -> dict:
        """
        Fast label extraction using Playwright: returns all <label> elements in <body> as HTML strings.
        Blocks images, stylesheets, and fonts for speed.
        """
        cache_key = f"labels:{url}:fast"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Returning cached label analysis for {url} (from Redis, fast mode).")
                return cached["labels"]
        logger.info(f"Analyzing labels at: {url} (force_refresh={force_refresh}, fast mode)")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            # Block images, stylesheets, and fonts for speed
            async def route_handler(route, request):
                if request.resource_type in ["image", "stylesheet", "font"]:
                    await route.abort()
                else:
                    await route.continue_()
            await page.route("**/*", route_handler)
            await page.goto(url)
            await page.wait_for_selector("body label")  # Wait for at least one label in body
            labels = await page.query_selector_all("body label")
            label_htmls = []
            if labels:
                for label in labels:
                    outer_html = await label.evaluate('el => el.outerHTML')
                    label_htmls.append(outer_html)
            await browser.close()
        result = {"status": "success", "labels": label_htmls, "label_count": len(label_htmls)}
        self.cache.set(cache_key, {"labels": result}, ttl_seconds=3600)
        logger.info(f"analyze_labels_fast: Found {len(label_htmls)} labels at {url}")
        return result
