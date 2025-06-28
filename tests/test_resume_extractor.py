#!/usr/bin/env python3
"""
TEST RESUME EXTRACTOR - Demonstrates how to use the resume_extractor module

This script shows how to use the different functions of the resume_extractor module:
1. Basic text extraction from PDFs
2. Full resume analysis with section identification
3. Contact information extraction
4. Generating embeddings for AI matching
5. Saving resume data to JSON

Usage:
    python test_resume_extractor.py path/to/resume.pdf
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pytest

# Import the resume_extractor module
try:
    from app.services.resume_extractor import (
        extract_text_from_resume, 
        analyze_resume, 
        extract_contact_information,
        generate_embeddings,
        save_resume_as_json
    )
except ImportError:
    print("Error: Could not import resume_extractor module.")
    print("Make sure app/services/resume_extractor.py is in the correct location.")
    sys.exit(1)

@pytest.fixture
def pdf_path():
    # Provide a real or dummy path to a test resume file
    return "docs/resume/ERIC _ABRAM33.docx"

@pytest.fixture
def text(pdf_path):
    return extract_text_from_resume(pdf_path)

def test_basic_extraction(pdf_path):
    """Test basic text extraction from PDF"""
    print("\n" + "="*50)
    print("TESTING BASIC TEXT EXTRACTION")
    print("="*50)
    
    text = extract_text_from_resume(pdf_path)
    
    print(f"Successfully extracted {len(text)} characters from {pdf_path}")
    print("\nFirst 300 characters of the resume:")
    print("-"*50)
    print(text[:300] + "...")
    print("-"*50)
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Save raw text for reference
    txt_path = os.path.join("output", Path(pdf_path).stem + "_raw.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"Raw text saved to {txt_path}")
    
    return text

def test_contact_extraction(text):
    """Test contact information extraction"""
    print("\n" + "="*50)
    print("TESTING CONTACT INFORMATION EXTRACTION")
    print("="*50)
    
    contact = extract_contact_information(text)
    
    print("Extracted contact information:")
    print(f"Name:     {contact.name or 'Not found'}")
    print(f"Email:    {contact.email or 'Not found'}")
    print(f"Phone:    {contact.phone or 'Not found'}")
    print(f"LinkedIn: {contact.linkedin or 'Not found'}")
    print(f"GitHub:   {contact.github or 'Not found'}")
    print(f"Website:  {contact.website or 'Not found'}")
    
    return contact

def test_full_analysis(pdf_path):
    """Test full resume analysis"""
    print("\n" + "="*50)
    print("TESTING FULL RESUME ANALYSIS")
    print("="*50)

    resume = analyze_resume(pdf_path)

    print("Resume content preview:")
    print(resume.page_content[:300] + "...")
    print("Metadata:", resume.metadata)
    # Optionally, add more assertions or checks here

def test_embedding_generation(text):
    """Test generating embeddings for AI matching"""
    print("\n" + "="*50)
    print("TESTING EMBEDDING GENERATION")
    print("="*50)
    
    try:
        embedding = generate_embeddings(text)
        
        if embedding:
            print(f"Successfully generated embedding with {len(embedding)} dimensions")
            print("\nFirst 5 values of embedding vector:")
            print(embedding[:5])
            
            # Save embedding for reference
            output_path = os.path.join("output", "resume_embedding.json")
            with open(output_path, "w") as f:
                json.dump({"embedding": embedding}, f)
            print(f"\nEmbedding saved to {output_path}")
        else:
            print("Could not generate embeddings - OpenAI API key may be missing or invalid")
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")

def main():
    """Main function to run all tests"""
    parser = argparse.ArgumentParser(description="Test the resume_extractor module")
    parser.add_argument("pdf_path", help="Path to the PDF resume file")
    parser.add_argument("--skip-embeddings", action="store_true", 
                      help="Skip embedding generation (no OpenAI API needed)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} does not exist")
        return 1
    
    # Run tests
    print(f"Testing resume_extractor with {args.pdf_path}")
    text = test_basic_extraction(args.pdf_path)
    contact = test_contact_extraction(text)
    resume = test_full_analysis(args.pdf_path)
    
    if not args.skip_embeddings:
        test_embedding_generation(text)
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)
    print("Output files saved to the 'output' directory")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
