import os
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import json
import pickle
from loguru import logger

class ResumeEmbedService:
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
        if self.openai_api_key == "your_openai_api_key_here":
            logger.warning("⚠️ OpenAI API key not set. Embeddings will not work.")
            self.embeddings = None
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-3-small"
            )
        self.docs_path = Path("docs/resume")
        self.resume_file = self.docs_path / "ERIC _ABRAM33.docx"
        self.vectordb_path = Path("resume/vectordb")
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def embed_resume(self) -> Dict[str, Any]:
        """Load, split, and embed the resume document."""
        logger.info("🚀 Embedding resume document...")
        loader = Docx2txtLoader(str(self.resume_file))
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        embeddings_list = self.embeddings.embed_documents(texts)
        full_text = "\n\n".join(texts)
        query_embedding = self.embeddings.embed_query(full_text)
        embedding_data = {
            "embeddings": embeddings_list,
            "query_embedding": query_embedding,
            "texts": texts,
            "metadatas": metadatas,
            "model": "text-embedding-3-small",
            "total_chunks": len(texts),
            "embedding_dimension": len(embeddings_list[0]) if embeddings_list else 0,
            "creation_timestamp": datetime.now().isoformat(),
            "source_resume": str(self.resume_file),
        }
        return embedding_data

    def save_resume_embeddings(self, embedding_data: Dict[str, Any], vectorstore: Any = None) -> str:
        """Save embeddings and vector store to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        embeddings_file = self.vectordb_path / f"embeddings_{timestamp}.json"
        json_data = embedding_data.copy()
        if "embeddings" in json_data:
            json_data["embeddings"] = [emb if isinstance(emb, list) else emb.tolist() for emb in json_data["embeddings"]]
        if "query_embedding" in json_data:
            json_data["query_embedding"] = json_data["query_embedding"] if isinstance(json_data["query_embedding"], list) else json_data["query_embedding"].tolist()
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        pickle_file = self.vectordb_path / f"embeddings_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(embedding_data, f)
        faiss_path = None
        if vectorstore:
            faiss_path = self.vectordb_path / f"faiss_store_{timestamp}"
            vectorstore.save_local(str(faiss_path))
            logger.info(f"   💾 FAISS store: {faiss_path}")
        metadata = {
            "embeddings_json": str(embeddings_file),
            "embeddings_pickle": str(pickle_file),
            "faiss_store": str(faiss_path) if faiss_path else None,
            "timestamp": timestamp,
            "creation_date": datetime.now().isoformat(),
            "source_resume": str(self.resume_file),
            "total_chunks": embedding_data.get("total_chunks", 0),
            "embedding_dimension": embedding_data.get("embedding_dimension", 0),
            "model": embedding_data.get("model", "unknown")
        }
        metadata_file = self.vectordb_path / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        index_file = self.vectordb_path / "index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
        else:
            index_data = {"entries": []}
        index_data["entries"].append(metadata)
        index_data["last_updated"] = datetime.now().isoformat()
        index_data["total_entries"] = len(index_data["entries"])
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"✅ Saved resume embeddings and metadata.")
        return timestamp 