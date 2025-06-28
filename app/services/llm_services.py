import os
from typing import Optional
from bs4 import BeautifulSoup
import re

# If you want to use OpenAI, import openai and set your API key
try:
    import openai
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except ImportError:
    openai = None


def extract_question_from_label_html(label_html: str, use_llm: bool = True) -> str:
    """
    Extract the human-readable question from a <label> HTML string.
    If use_llm is True and OpenAI is available, use LLM to extract the question.
    Otherwise, fallback to BeautifulSoup text extraction.
    """
    # Fallback: just get the text content, but ensure space before parenthesis
    def fallback_extract(label_html: str) -> str:
        soup = BeautifulSoup(label_html, "html.parser")
        text = soup.get_text(strip=True)
        # Insert space before ( if missing
        text = re.sub(r"(?<!\s)\(", " (", text)
        return text

    if not use_llm or openai is None or not OPENAI_API_KEY:
        return fallback_extract(label_html)

    prompt = f"""
You are an expert at extracting clear, human-readable questions from HTML <label> tags. Given the following HTML, extract only the question or prompt that a human would see on a form. Remove any extra HTML, tooltips, or irrelevant text. Return only the question, nothing else.

<label> HTML:
{label_html}

Question:
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.0,
        )
        question = response["choices"][0]["message"]["content"].strip()
        return question
    except Exception as e:
        # Fallback to BeautifulSoup if LLM fails
        return fallback_extract(label_html) 