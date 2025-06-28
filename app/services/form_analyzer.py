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
