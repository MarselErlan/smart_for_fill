#!/usr/bin/env python3
"""
Resume Extractor - Uses LangChain to extract content from resume.docx, creates embeddings, and saves to vector database
Reads from docs/resume/ERIC_ABRAM33.docx and saves embeddings to resume/vectordb
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Other imports
import numpy as np
from openai import OpenAI
from loguru import logger

class ResumeExtractor:
    def __init__(self, openai_api_key: str = None):
        """Initialize the resume extractor with LangChain components"""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
        
        if self.openai_api_key == "your_openai_api_key_here":
            logger.warning("⚠️ OpenAI API key not set. Embeddings will not work.")
            self.embeddings = None
            self.client = None
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-3-small"
            )
            self.client = OpenAI(api_key=self.openai_api_key)
        
        # Paths
        self.docs_path = Path("docs/resume")
        self.resume_file = self.docs_path / "ERIC _ABRAM33.docx"
        self.vectordb_path = Path("resume/vectordb")
        self.vectordb_path.mkdir(parents=True, exist_ok=True)
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Resume extractor initialized with LangChain")
        logger.info(f"Resume file: {self.resume_file}")
        logger.info(f"Vector DB path: {self.vectordb_path}")
    
    def load_resume_with_langchain(self) -> List[Any]:
        """Load resume using LangChain Docx2txtLoader"""
        try:
            if not self.resume_file.exists():
                raise FileNotFoundError(f"Resume file not found: {self.resume_file}")
            
            logger.info(f"📄 Loading resume with LangChain from: {self.resume_file}")
            
            # Use LangChain's Docx2txtLoader
            loader = Docx2txtLoader(str(self.resume_file))
            documents = loader.load()
            
            logger.info(f"✅ Loaded {len(documents)} document(s) with LangChain")
            
            # Log document details
            for i, doc in enumerate(documents):
                logger.info(f"   Document {i+1}: {len(doc.page_content)} characters")
                logger.info(f"   Metadata: {doc.metadata}")
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error loading resume with LangChain: {e}")
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
    
    def create_embeddings_with_langchain(self, chunks: List[Any]) -> Dict[str, Any]:
        """Create embeddings using LangChain OpenAI embeddings"""
        try:
            if not self.embeddings:
                logger.error("❌ OpenAI embeddings not initialized. Cannot create embeddings.")
                return {"error": "OpenAI API key not set"}
            
            logger.info("🔢 Creating embeddings with LangChain...")
            
            # Extract text content from chunks
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            logger.info(f"📊 Creating embeddings for {len(texts)} text chunks...")
            
            # Create embeddings using LangChain
            embeddings_list = self.embeddings.embed_documents(texts)
            
            # Also create a query embedding for the full resume
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
                "chunk_stats": {
                    "total_characters": sum(len(text) for text in texts),
                    "avg_chunk_size": sum(len(text) for text in texts) / len(texts) if texts else 0,
                    "min_chunk_size": min(len(text) for text in texts) if texts else 0,
                    "max_chunk_size": max(len(text) for text in texts) if texts else 0
                }
            }
            
            logger.info(f"✅ Created {len(embeddings_list)} embeddings with dimension {embedding_data['embedding_dimension']}")
            
            return embedding_data
            
        except Exception as e:
            logger.error(f"❌ Error creating embeddings with LangChain: {e}")
            raise
    
    def create_faiss_vectorstore(self, chunks: List[Any]) -> Any:
        """Create FAISS vector store using LangChain"""
        try:
            if not self.embeddings:
                logger.error("❌ OpenAI embeddings not initialized. Cannot create vector store.")
                return None
            
            logger.info("🗄️ Creating FAISS vector store...")
            
            # Extract texts and metadatas
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Create FAISS vector store
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Created FAISS vector store with {len(texts)} documents")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"❌ Error creating FAISS vector store: {e}")
            raise
    
    def save_to_vectordb(self, embedding_data: Dict[str, Any], vectorstore: Any = None) -> str:
        """Save embeddings and vector store to database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save embedding data as JSON
            embeddings_file = self.vectordb_path / f"embeddings_{timestamp}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_data = embedding_data.copy()
            if "embeddings" in json_data:
                json_data["embeddings"] = [emb if isinstance(emb, list) else emb.tolist() for emb in json_data["embeddings"]]
            if "query_embedding" in json_data:
                json_data["query_embedding"] = json_data["query_embedding"] if isinstance(json_data["query_embedding"], list) else json_data["query_embedding"].tolist()
            
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            # Save raw embeddings as pickle for faster loading
            pickle_file = self.vectordb_path / f"embeddings_{timestamp}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(embedding_data, f)
            
            # Save FAISS vector store if available
            faiss_path = None
            if vectorstore:
                faiss_path = self.vectordb_path / f"faiss_store_{timestamp}"
                vectorstore.save_local(str(faiss_path))
                logger.info(f"   💾 FAISS store: {faiss_path}")
            
            # Save metadata
            metadata = {
                "embeddings_json": str(embeddings_file),
                "embeddings_pickle": str(pickle_file),
                "faiss_store": str(faiss_path) if faiss_path else None,
                "timestamp": timestamp,
                "creation_date": datetime.now().isoformat(),
                "source_resume": str(self.resume_file),
                "total_chunks": embedding_data.get("total_chunks", 0),
                "embedding_dimension": embedding_data.get("embedding_dimension", 0),
                "model": embedding_data.get("model", "unknown"),
                "chunk_stats": embedding_data.get("chunk_stats", {})
            }
            
            metadata_file = self.vectordb_path / f"metadata_{timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Update index
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
            
            logger.info(f"✅ Saved to vector database:")
            logger.info(f"   📄 Embeddings JSON: {embeddings_file}")
            logger.info(f"   🥒 Embeddings Pickle: {pickle_file}")
            logger.info(f"   📋 Metadata: {metadata_file}")
            logger.info(f"   📚 Index updated: {index_file}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"❌ Error saving to vector database: {e}")
            raise
    
    def process_resume(self) -> Dict[str, Any]:
        """Complete pipeline: load → split → embed → save"""
        try:
            logger.info("🚀 Starting LangChain resume processing pipeline...")
            
            # Step 1: Load resume with LangChain
            logger.info("📄 Step 1: Loading resume with LangChain...")
            documents = self.load_resume_with_langchain()
            
            # Step 2: Split into chunks
            logger.info("🔪 Step 2: Splitting documents into chunks...")
            chunks = self.split_documents(documents)
            
            # Step 3: Create embeddings
            logger.info("🔢 Step 3: Creating embeddings...")
            embedding_data = self.create_embeddings_with_langchain(chunks)
            
            # Step 4: Create FAISS vector store
            vectorstore = None
            if "error" not in embedding_data:
                logger.info("🗄️ Step 4: Creating FAISS vector store...")
                vectorstore = self.create_faiss_vectorstore(chunks)
            else:
                logger.warning("⚠️ Skipping vector store creation due to embedding error")
            
            # Step 5: Save to vector database
            logger.info("💾 Step 5: Saving to vector database...")
            timestamp = self.save_to_vectordb(embedding_data, vectorstore)
            
            # Prepare result
            result = {
                "status": "success",
                "timestamp": timestamp,
                "documents_loaded": len(documents),
                "chunks_created": len(chunks),
                "embedding_data": {
                    "total_chunks": embedding_data.get("total_chunks", 0),
                    "dimension": embedding_data.get("embedding_dimension", 0),
                    "model": embedding_data.get("model", "none"),
                    "chunk_stats": embedding_data.get("chunk_stats", {})
                },
                "vectorstore_created": vectorstore is not None,
                "files_created": {
                    "embeddings_json": f"embeddings_{timestamp}.json",
                    "embeddings_pickle": f"embeddings_{timestamp}.pkl",
                    "faiss_store": f"faiss_store_{timestamp}" if vectorstore else None,
                    "metadata": f"metadata_{timestamp}.json"
                }
            }
            
            logger.info("🎉 LangChain resume processing completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Resume processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def search_resume(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search the resume vector database for relevant content"""
        try:
            # Load the latest FAISS store
            index_file = self.vectordb_path / "index.json"
            if not index_file.exists():
                return {"error": "No vector database found. Run process_resume() first."}
            
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
            logger.error(f"❌ Error searching resume: {e}")
            return {"error": str(e)}

def main():
    """Main function to run the resume extractor"""
    print("🚀 LangChain Resume Extractor - ERIC_ABRAM33.docx")
    print("=" * 60)
    
    # Initialize extractor
    extractor = ResumeExtractor()
    
    # Process the resume
    result = extractor.process_resume()
    
    # Display results
    print("\n📊 Processing Results:")
    print("=" * 30)
    
    if result["status"] == "success":
        print("✅ Status: SUCCESS")
        print(f"📅 Timestamp: {result['timestamp']}")
        print(f"📄 Documents loaded: {result['documents_loaded']}")
        print(f"🔪 Chunks created: {result['chunks_created']}")
        print(f"🔢 Embedding chunks: {result['embedding_data']['total_chunks']}")
        print(f"📐 Embedding dimension: {result['embedding_data']['dimension']}")
        print(f"🤖 Model: {result['embedding_data']['model']}")
        print(f"🗄️ Vector store created: {result['vectorstore_created']}")
        
        # Show chunk statistics
        chunk_stats = result['embedding_data'].get('chunk_stats', {})
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
        search_result = extractor.search_resume("software engineer experience", k=3)
        if "error" not in search_result:
            print(f"   ✅ Search test successful: {search_result['total_results']} results found")
        else:
            print(f"   ❌ Search test failed: {search_result['error']}")
            
    else:
        print("❌ Status: ERROR")
        print(f"💥 Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("✨ LangChain resume extraction completed!")

if __name__ == "__main__":
    main() 