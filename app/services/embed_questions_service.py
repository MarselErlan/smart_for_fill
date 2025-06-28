import os
from typing import List
from langchain_openai import OpenAIEmbeddings

# Singleton for embeddings
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        _embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )
    return _embeddings


def embed_question(question: str) -> List[float]:
    """
    Embed a question string using the project's embedding model.
    Returns a list of floats (embedding vector).
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(question) 