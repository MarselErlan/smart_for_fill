import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_services import extract_question_from_label_html

def test_extract_question_from_label_html_fallback():
    label_html = '<label for="email">Email Address <span class="tooltip">(required)</span></label>'
    question = extract_question_from_label_html(label_html, use_llm=False)
    print("Extracted question (fallback):", question)
    assert question == "Email Address (required)"

def test_extract_question_from_label_html_llm():
    label_html = '<label for="email">Email Address <span class="tooltip">(required)</span></label>'
    mock_response = {
        "choices": [
            {"message": {"content": "Email Address"}}
        ]
    }
    with patch("app.services.llm_services.openai") as mock_openai:
        mock_openai.ChatCompletion.create.return_value = mock_response
        question = extract_question_from_label_html(label_html, use_llm=True)
        print("Extracted question (LLM):", question)
        assert question == "Email Address" 