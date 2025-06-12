# smart_form_fill/examples/demo.py

from app.services.form_analyzer import FormAnalyzer
from app.services.form_filler import FormFiller
from app.services.vector_store import VectorStore
import os
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    analyzer = FormAnalyzer(os.getenv("OPENAI_API_KEY"))
    filler = FormFiller(headless=False)  # Show browser for demo
    vector_store = VectorStore(os.getenv("OPENAI_API_KEY"))
    
    # Add personal information to vector store
    vector_store.add_personal_info({
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "123-456-7890",
        "location": "San Francisco, CA",
        "resume_path": "path/to/resume.pdf",
        "linkedin": "https://linkedin.com/in/johndoe",
        "github": "https://github.com/johndoe",
        "website": "https://johndoe.com",
        "experience_years": "5",
        "education": "BS Computer Science",
        "skills": ["Python", "JavaScript", "Machine Learning"]
    })
    
    # Job application URL
    url = "https://example.com/jobs/apply"
    
    # 1. Analyze the form
    print("🔍 Analyzing form structure...")
    analysis = analyzer.analyze_form(url)
    
    if analysis["status"] == "success":
        field_map = analysis["field_map"]
        
        # 2. Prepare user data for each field
        user_data = {}
        for field_name, field_info in field_map.items():
            best_match = vector_store.find_best_match(
                field_name,
                field_info["type"]
            )
            if best_match:
                user_data[field_name] = best_match
        
        # 3. Fill the form
        print("✍️ Filling form fields...")
        result = filler.fill_form(url, field_map, user_data)
        
        if result["status"] == "success":
            print("✅ Form filled successfully!")
            print("\nField results:")
            for field, status in result["results"].items():
                print(f"  - {field}: {status}")
            print(f"\nScreenshot saved: {result['screenshot']}")
        else:
            print(f"❌ Form filling failed: {result['error']}")
    else:
        print(f"❌ Form analysis failed: {analysis['error']}")

if __name__ == "__main__":
    main()
