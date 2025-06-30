import os
from typing import List
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings

# Singleton for embeddings
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Always use the local model, wrapped for LangChain compatibility
        model_name = "all-MiniLM-L6-v2"
        _embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return _embeddings


def embed_question(question: str) -> List[float]:
    """
    Embed a question string using the local sentence-transformers model.
    Returns a list of floats (embedding vector).
    """
    embeddings = get_embeddings()
    result = embeddings.embed_query(question) # Use embed_query for single questions
    print(f"[DEBUG] Embedding for '{question}': {result[:10]}... (len={len(result)})")
    return result 