#!/usr/bin/env python3
"""
Test script for Vector Database API endpoints
Demonstrates how to use the FastAPI endpoints for re-embedding and searching
"""

import requests
import json
import time
from typing import Dict, Any

class VectorDBAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_connection(self) -> bool:
        """Test if the API server is running"""
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information and available endpoints"""
        response = self.session.get(f"{self.base_url}/")
        return response.json()
    
    def get_resume_status(self) -> Dict[str, Any]:
        """Get resume vector database status"""
        response = self.session.get(f"{self.base_url}/api/v1/resume/status")
        return response.json()
    
    def get_personal_info_status(self) -> Dict[str, Any]:
        """Get personal info vector database status"""
        response = self.session.get(f"{self.base_url}/api/v1/personal-info/status")
        return response.json()
    
    def reembed_resume(self) -> Dict[str, Any]:
        """Re-embed resume vector database"""
        response = self.session.post(f"{self.base_url}/api/v1/resume/reembed")
        return response.json()
    
    def reembed_personal_info(self) -> Dict[str, Any]:
        """Re-embed personal info vector database"""
        response = self.session.post(f"{self.base_url}/api/v1/personal-info/reembed")
        return response.json()
    
    def reembed_all(self) -> Dict[str, Any]:
        """Re-embed both vector databases"""
        response = self.session.post(f"{self.base_url}/api/v1/reembed-all")
        return response.json()
    
    def search_resume(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search resume vector database"""
        params = {"query": query, "k": k}
        response = self.session.post(f"{self.base_url}/api/v1/resume/search", params=params)
        return response.json()
    
    def search_personal_info(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search personal info vector database"""
        params = {"query": query, "k": k}
        response = self.session.post(f"{self.base_url}/api/v1/personal-info/search", params=params)
        return response.json()

def print_json(data: Dict[str, Any], title: str = ""):
    """Pretty print JSON data"""
    if title:
        print(f"\n{'='*60}")
        print(f"📊 {title}")
        print('='*60)
    print(json.dumps(data, indent=2))

def main():
    """Main test function"""
    print("🚀 Vector Database API Tester")
    print("="*60)
    
    # Initialize tester
    tester = VectorDBAPITester()
    
    # Test connection
    print("\n🔗 Testing API connection...")
    if not tester.test_connection():
        print("❌ API server is not running!")
        print("💡 Start the server with: python main.py")
        return
    
    print("✅ API server is running!")
    
    # Get API info
    api_info = tester.get_api_info()
    print_json(api_info, "API Information")
    
    # Check current status
    print("\n📊 Checking current vector database status...")
    
    resume_status = tester.get_resume_status()
    print_json(resume_status, "Resume Database Status")
    
    personal_info_status = tester.get_personal_info_status()
    print_json(personal_info_status, "Personal Info Database Status")
    
    # Test search functionality (if databases exist)
    if resume_status.get("status") == "ready":
        print("\n🔍 Testing resume search...")
        search_queries = [
            "python programming experience",
            "automation testing tools",
            "software development"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching resume for: '{query}'")
            results = tester.search_resume(query, k=2)
            if results.get("status") == "success":
                print(f"✅ Found {results['total_results']} results")
                for i, result in enumerate(results["results"], 1):
                    print(f"   {i}. Score: {result['similarity_score']:.4f}")
                    print(f"      Content: {result['content'][:100]}...")
            else:
                print(f"❌ Search failed: {results}")
    
    if personal_info_status.get("status") == "ready":
        print("\n🔍 Testing personal info search...")
        search_queries = [
            "work authorization",
            "salary expectations", 
            "software engineer experience"
        ]
        
        for query in search_queries:
            print(f"\n🔎 Searching personal info for: '{query}'")
            results = tester.search_personal_info(query, k=2)
            if results.get("status") == "success":
                print(f"✅ Found {results['total_results']} results")
                for i, result in enumerate(results["results"], 1):
                    print(f"   {i}. Score: {result['similarity_score']:.4f}")
                    print(f"      Content: {result['content'][:100]}...")
            else:
                print(f"❌ Search failed: {results}")
    
    # Test re-embedding
    print("\n🔄 Testing re-embedding functionality...")
    
    print("\n📄 Re-embedding personal info...")
    reembed_result = tester.reembed_personal_info()
    if reembed_result.get("status") == "success":
        print(f"✅ Personal info re-embedded successfully!")
        print(f"   Processing time: {reembed_result['processing_time']:.2f}s")
        print(f"   Timestamp: {reembed_result['timestamp']}")
        print(f"   Chunks created: {reembed_result['details']['chunks_created']}")
    else:
        print(f"❌ Personal info re-embedding failed: {reembed_result}")
    
    print("\n📄 Re-embedding resume...")
    reembed_result = tester.reembed_resume()
    if reembed_result.get("status") == "success":
        print(f"✅ Resume re-embedded successfully!")
        print(f"   Processing time: {reembed_result['processing_time']:.2f}s")
        print(f"   Timestamp: {reembed_result['timestamp']}")
        print(f"   Chunks created: {reembed_result['details']['chunks_created']}")
    else:
        print(f"❌ Resume re-embedding failed: {reembed_result}")
    
    # Test batch re-embedding
    print("\n🔄 Testing batch re-embedding...")
    batch_result = tester.reembed_all()
    if batch_result.get("status") in ["success", "partial_success"]:
        print(f"✅ Batch re-embedding completed!")
        print(f"   Status: {batch_result['status']}")
        print(f"   Total processing time: {batch_result['processing_time']:.2f}s")
        print(f"   Resume status: {batch_result['results']['resume']['status']}")
        print(f"   Personal info status: {batch_result['results']['personal_info']['status']}")
    else:
        print(f"❌ Batch re-embedding failed: {batch_result}")
    
    print("\n" + "="*60)
    print("✨ Vector Database API testing completed!")
    print("\n💡 Available endpoints:")
    print("   • GET  /api/v1/resume/status")
    print("   • POST /api/v1/resume/reembed")
    print("   • POST /api/v1/resume/search?query=<query>&k=<num>")
    print("   • GET  /api/v1/personal-info/status")
    print("   • POST /api/v1/personal-info/reembed")
    print("   • POST /api/v1/personal-info/search?query=<query>&k=<num>")
    print("   • POST /api/v1/reembed-all")

if __name__ == "__main__":
    main() 