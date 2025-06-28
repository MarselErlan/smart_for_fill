import pytest
from unittest.mock import MagicMock
from app.services.similarity_search_service import similarity_search

class FakeDoc:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def test_similarity_search_returns_results():
    fake_embedding = [0.1, 0.2, 0.3]
    fake_docs = [
        FakeDoc("Doc 1 content", {"id": 1}),
        FakeDoc("Doc 2 content", {"id": 2}),
        FakeDoc("Doc 3 content", {"id": 3}),
    ]
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_by_vector.return_value = fake_docs
    results = similarity_search(fake_embedding, mock_vectorstore, k=3)
    assert len(results) == 3
    assert results[0]["content"] == "Doc 1 content"
    assert results[1]["metadata"] == {"id": 2}
    mock_vectorstore.similarity_search_by_vector.assert_called_once_with(fake_embedding, k=3) 