import pytest
from unittest.mock import patch, MagicMock
from app.services.embed_questions_service import embed_question
import os
import numpy as np

def test_embed_question_returns_vector():
    question = "What is your email address?"
    fake_embedding = [0.1, 0.2, 0.3]
    with patch("app.services.embed_questions_service.get_embeddings") as mock_get_embeddings:
        mock_model = MagicMock()
        mock_encode = MagicMock()
        mock_encode.tolist.return_value = fake_embedding
        mock_model.encode.return_value = mock_encode
        mock_get_embeddings.return_value = mock_model
        result = embed_question(question)
        assert result == fake_embedding
        mock_model.encode.assert_called_once_with(question)

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="No OpenAI API key set; skipping real embedding test."
)
def test_embed_question_real_embedding():
    question = "What is your email address?"
    embedding = embed_question(question)
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
    assert len(embedding) > 10
    # Check that not all values are zero
    assert any(abs(x) > 1e-6 for x in embedding) 