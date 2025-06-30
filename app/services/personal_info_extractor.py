#!/usr/bin/env python3
"""
Personal Info Extractor - Uses LangChain to extract content from personal_information.txt, creates embeddings, and saves to vector database
Reads from docs/info/personal_information.txt and saves embeddings to info/vectordb
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Other imports
import numpy as np
from loguru import logger
from langchain_community.document_loaders import Docx2txtLoader
from app.services.resume_extractor import ResumeExtractor
from app.services.embed_personal_info_service import PersonalInfoEmbedService



# Embedding logic moved to app/services/embed_personal_info_service.py

class PersonalInfoExtractor:
    def __init__(self):
        self.docs_path = Path("docs/info")
        self.personal_info_file = self.docs_path / "personal_information.txt"
        self.vectordb_path = Path("info/vectordb")
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "=", " ", ""]
        )
        self.embed_service = PersonalInfoEmbedService()
        self.embeddings = self.embed_service.embeddings  # Add embeddings attribute

    def process_personal_info(self) -> Dict[str, Any]:
        """
        Main method to process the personal info file:
        - Load text from file
        - Split into chunks
        - Create embeddings
        - Save to FAISS vector store
        """
        try:
            logger.info("🚀 Starting personal info processing...")
            
            # 1. Load documents
            documents = self.load_personal_info_with_langchain()
            if not documents:
                return {"status": "error", "error": "No documents loaded"}
            
            # 2. Split documents
            chunks = self.split_documents(documents)
            if not chunks:
                return {"status": "error", "error": "No chunks created"}
            
            # 3. Create embeddings and vector store
            embedding_result = self.embed_service.embed_documents(chunks)
            if not embedding_result or not embedding_result.get("vectorstore"):
                return {"status": "error", "error": "Failed to create embeddings"}
            
            # 4. Save embeddings
            save_result = self.embed_service.save_embeddings(embedding_result)
            
            timestamp = datetime.now().isoformat()
            
            result = {
                "status": "success",
                "timestamp": timestamp,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "vectorstore_created": save_result.get("vectorstore_created", False),
                "files_created": save_result.get("files_created", {}),
                "total_chunks": len(chunks),
                "dimension": embedding_result.get("dimension"),
                "model": embedding_result.get("model"),
                "chunk_stats": embedding_result.get("chunk_stats")
            }
            
            logger.info("✅ Personal info processing completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Personal info processing failed: {e}")
            return {"status": "error", "error": str(e)}

    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search the personal info vector database for relevant content"""
        try:
            # Load the latest FAISS store
            index_file = self.vectordb_path / "index.json"
            if not index_file.exists():
                return {"error": "No vector database found. Run process_personal_info() first."}
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            if not index_data["files"]:
                return {"error": "No entries in vector database."}
            
            # Get the latest entry
            latest_entry = index_data["files"][-1]
            faiss_path = latest_entry.get("faiss_store")
            
            if not faiss_path:
                return {"error": "No FAISS store available."}
            
            # Load FAISS store
            vectorstore = FAISS.load_local(faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            
            # Search
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error searching personal info: {e}")
            return {"error": str(e)}

    def embed_personal_info(self):
        return self.embed_service.embed_personal_info()

    def save_personal_info_embeddings(self, embedding_data: Dict[str, Any], vectorstore: Any = None):
        return self.embed_service.save_personal_info_embeddings(embedding_data, vectorstore)

    def load_personal_info_with_langchain(self) -> List[Any]:
        """Load personal information using LangChain TextLoader"""
        try:
            if not self.personal_info_file.exists():
                raise FileNotFoundError(f"Personal info file not found: {self.personal_info_file}")
            
            logger.info(f"📄 Loading personal info with LangChain from: {self.personal_info_file}")
            
            # Use LangChain's TextLoader
            loader = TextLoader(str(self.personal_info_file), encoding='utf-8')
            documents = loader.load()
            
            logger.info(f"✅ Loaded {len(documents)} document(s) with LangChain")
            
            # Log document details
            for i, doc in enumerate(documents):
                logger.info(f"   Document {i+1}: {len(doc.page_content)} characters")
                logger.info(f"   Metadata: {doc.metadata}")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error loading personal info with LangChain: {e}")
            raise
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks using LangChain text splitter"""
        try:
            logger.info("🔪 Splitting documents into chunks...")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"✅ Created {len(chunks)} chunks from documents")
            
            # Log chunk details
            total_chars = sum(len(chunk.page_content) for chunk in chunks)
            avg_chunk_size = total_chars / len(chunks) if chunks else 0
            
            logger.info(f"   Total characters: {total_chars}")
            logger.info(f"   Average chunk size: {avg_chunk_size:.0f} characters")
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error splitting documents: {e}")
            raise
    
    def search_personal_info(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search the personal info vector database for relevant content"""
        try:
            # Load the latest FAISS store
            index_file = self.vectordb_path / "index.json"
            if not index_file.exists():
                return {"error": "No vector database found. Run process_personal_info() first."}
            
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            
            if not index_data["entries"]:
                return {"error": "No entries in vector database."}
            
            # Get the latest entry
            latest_entry = index_data["entries"][-1]
            faiss_path = latest_entry.get("faiss_store")
            
            if not faiss_path:
                return {"error": "No FAISS store available."}
            
            # Load FAISS store
            vectorstore = FAISS.load_local(faiss_path, self.embed_service.embeddings, allow_dangerous_deserialization=True)
            
            # Search
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score)
                })
            
            return {
                "query": query,
                "results": formatted_results,
                "total_results": len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"❌ Error searching personal info: {e}")
            return {"error": str(e)}

def main():
    """Main function to run the personal info extractor"""
    print("🚀 LangChain Personal Info Extractor - personal_information.txt")
    print("=" * 70)
    
    # Initialize extractor
    extractor = PersonalInfoExtractor()
    
    # Process the personal info
    result = extractor.process_personal_info()
    
    # Display results
    print("\n📊 Processing Results:")
    print("=" * 30)
    
    if result["status"] == "success":
        print("✅ Status: SUCCESS")
        print(f"📅 Timestamp: {result['timestamp']}")
        print(f"📄 Documents loaded: {result['documents_loaded']}")
        print(f"🔪 Chunks created: {result['chunks_created']}")
        print(f"🔢 Embedding chunks: {result['total_chunks']}")
        print(f"📐 Embedding dimension: {result['dimension']}")
        print(f"🤖 Model: {result['model']}")
        print(f"🗄️ Vector store created: {result['vectorstore_created']}")
        
        # Show chunk statistics
        chunk_stats = result.get('chunk_stats', {})
        if chunk_stats:
            print(f"\n📈 Chunk Statistics:")
            print(f"   • Total characters: {chunk_stats.get('total_characters', 0):,}")
            print(f"   • Average chunk size: {chunk_stats.get('avg_chunk_size', 0):.0f}")
            print(f"   • Min chunk size: {chunk_stats.get('min_chunk_size', 0)}")
            print(f"   • Max chunk size: {chunk_stats.get('max_chunk_size', 0)}")
        
        print("\n📁 Files created:")
        for file_type, filename in result['files_created'].items():
            if filename:
                print(f"   • {file_type}: {filename}")
        
        # Test search functionality
        print("\n🔍 Testing search functionality...")
        search_queries = ["work authorization", "salary expectations", "contact information"]
        
        for query in search_queries:
            search_result = extractor.search_personal_info(query, k=2)
            if "error" not in search_result:
                print(f"   ✅ Search '{query}': {search_result['total_results']} results found")
            else:
                print(f"   ❌ Search '{query}' failed: {search_result['error']}")
                break
            
    else:
        print("❌ Status: ERROR")
        print(f"💥 Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("✨ LangChain personal info extraction completed!")

if __name__ == "__main__":
    main()

def extract_text_from_resume(docx_path):
    """Extract raw text from a resume DOCX file using LangChain."""
    extractor = ResumeExtractor()
    loader = Docx2txtLoader(str(docx_path))
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

def analyze_resume(docx_path):
    """Analyze the resume and return a structured object (stub)."""
    extractor = ResumeExtractor()
    documents = extractor.load_resume_with_langchain()
    return documents[0] if documents else None

def extract_contact_information(text):
    """Extract contact information from resume text (stub)."""
    class Contact:
        name = "John Doe"
        email = "john@example.com"
        phone = "123-456-7890"
        linkedin = "linkedin.com/in/johndoe"
        github = "github.com/johndoe"
        website = "johndoe.com"
    return Contact()

def save_resume_as_json(resume, output_path):
    """Save the resume analysis to a JSON file."""
    if hasattr(resume, 'page_content'):
        data = {"content": resume.page_content, "metadata": getattr(resume, "metadata", {})}
    else:
        data = resume
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2) 