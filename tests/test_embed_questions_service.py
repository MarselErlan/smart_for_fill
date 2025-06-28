import pytest
from unittest.mock import patch, MagicMock
from app.services.embed_questions_service import embed_question

def test_embed_question_returns_vector():
    question = "What is your email address?"
    fake_embedding = [0.1, 0.2, 0.3]
    with patch("app.services.embed_questions_service.get_embeddings") as mock_get_embeddings:
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = fake_embedding
        mock_get_embeddings.return_value = mock_embeddings
        result = embed_question(question)
        assert result == fake_embedding
        mock_embeddings.embed_query.assert_called_once_with(question) 