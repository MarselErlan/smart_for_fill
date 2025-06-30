import os
from typing import Optional
from bs4 import BeautifulSoup
import re
import json

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

def create_llm_prompt(fields_to_generate: list, combined_data: dict) -> str:
    """Creates a detailed prompt for the LLM to generate missing form field values."""
    
    prompt = f"""
You are an expert at professionally filling out job applications. Your task is to generate values for the following form fields based on the provided user data.

**User Data:**
- **Resume Data:** {combined_data.get('resume_vectordb_data', 'Not available')}
- **Personal Info:** {combined_data.get('personal_info_vectordb_data', 'Not available')}
- **Additional Data:** {combined_data.get('provided_user_data', 'Not available')}

**Fields to Fill:**
Please provide a professional, concise, and relevant value for each of the following fields. Output the results in a simple key-value JSON format, where the key is the field 'purpose'.

"""
    
    for field in fields_to_generate:
        prompt += f"- **{field.get('field_purpose', field.get('name', 'unknown'))}**: (type: {field.get('field_type', 'text')})\n"
        
    prompt += "\n**JSON Output:**\n"
    
    return prompt

def parse_llm_response(response: str) -> dict:
    """Parses the LLM's JSON response to extract field values."""
    try:
        # The response might be a JSON string enclosed in ```json ... ```
        if response.strip().startswith("```json"):
            response = response.strip()[7:-4]
        
        return json.loads(response)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}\nResponse:\n{response}")
        return {}

def generate_professional_value(purpose: str) -> str:
    """Fallback function to generate a plausible professional value for a given field purpose."""
    
    # Simple rules for generating fallback values
    if "email" in purpose.lower():
        return "professional.email@example.com"
    if "phone" in purpose.lower():
        return "123-456-7890"
    if "name" in purpose.lower():
        return "Taylor Professional"
    if "linkedin" in purpose.lower():
        return "https://linkedin.com/in/taylor-professional"
    if "website" in purpose.lower() or "portfolio" in purpose.lower():
        return "https://example.com"
        
    # Generic fallback
    return f"Professional response for {purpose}" 