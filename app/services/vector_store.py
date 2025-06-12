# smart_form_fill/app/services/vector_store.py

"""
Vector Store - Manages personal information in vector database
Uses embeddings to find the most relevant information for each form field
"""

import os
from typing import Dict, List
import numpy as np
from openai import OpenAI
import json
from loguru import logger

class VectorStore:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.vector_db_path = "data/vector_db/personal_info.json"
        self.load_vector_db()
    
    def load_vector_db(self):
        """Load existing vector database"""
        if os.path.exists(self.vector_db_path):
            with open(self.vector_db_path, 'r') as f:
                self.db = json.load(f)
        else:
            self.db = {
                "embeddings": [],
                "data": []
            }
    
    def add_personal_info(self, info: Dict):
        """
        Add new personal information to the vector database
        
        Args:
            info: Dictionary of personal information
        """
        # Create text representation
        text = " ".join(f"{k}: {v}" for k, v in info.items())
        
        # Generate embedding
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        
        # Store in database
        self.db["embeddings"].append(embedding)
        self.db["data"].append(info)
        
        # Save to disk
        self._save_db()
    
    def find_best_match(self, field_name: str, field_type: str) -> str:
        """
        Find the most relevant personal information for a form field
        
        Args:
            field_name: Name of the form field
            field_type: Type of the form field (text, email, etc.)
        """
        # Create query embedding
        query = f"field name: {field_name}, type: {field_type}"
        response = self.client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        
        # Find closest match
        similarities = []
        for emb in self.db["embeddings"]:
            similarity = np.dot(query_embedding, emb)
            similarities.append(similarity)
        
        if similarities:
            best_idx = np.argmax(similarities)
            return self.db["data"][best_idx]
        return None
    
    def _save_db(self):
        """Save vector database to disk"""
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        with open(self.vector_db_path, 'w') as f:
            json.dump(self.db, f, indent=2)
