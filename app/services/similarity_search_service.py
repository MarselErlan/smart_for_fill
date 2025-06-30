from typing import List, Dict, Any

# Assumes vectorstore is a LangChain FAISS object

def similarity_search(embedding: List[float], vectorstore, k: int = 3) -> List[Dict[str, Any]]:
    """
    Perform a similarity search over a FAISS vectorstore using a given embedding vector.
    Returns the top-k most similar documents with their content and metadata.
    """
    # LangChain FAISS expects a query vector for similarity_search_by_vector
    results = vectorstore.similarity_search_by_vector(embedding, k=k)
    # Each result is a Document object
    return [
        {"content": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ] 