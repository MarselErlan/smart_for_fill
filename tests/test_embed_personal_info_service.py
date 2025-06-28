import pytest
from unittest.mock import patch, MagicMock
from app.services.embed_personal_info_service import PersonalInfoEmbedService

def test_embed_personal_info_returns_embedding_data():
    fake_text = "Name: John Doe\nEmail: john@example.com"
    fake_embedding = [0.1, 0.2, 0.3]
    fake_chunks = [MagicMock(page_content=fake_text, metadata={})]
    with patch("app.services.embed_personal_info_service.TextLoader") as mock_loader, \
         patch("app.services.embed_personal_info_service.RecursiveCharacterTextSplitter") as mock_splitter, \
         patch("app.services.embed_personal_info_service.OpenAIEmbeddings") as mock_embeddings_class:
        # Mock loader
        mock_loader.return_value.load.return_value = [MagicMock(page_content=fake_text, metadata={})]
        # Mock splitter
        mock_splitter.return_value.split_documents.return_value = fake_chunks
        # Mock embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [fake_embedding]
        mock_embeddings.embed_query.return_value = fake_embedding
        mock_embeddings_class.return_value = mock_embeddings
        service = PersonalInfoEmbedService(openai_api_key="test-key")
        result = service.embed_personal_info()
        assert result["embeddings"] == [[0.1, 0.2, 0.3]]
        assert result["query_embedding"] == [0.1, 0.2, 0.3]
        assert result["texts"] == [fake_text] 