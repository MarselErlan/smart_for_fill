#!/usr/bin/env python3
"""
Test Enhanced Form Filler - 3-Tier Intelligent Data Retrieval System
Tests the new resume/vectordb → info/vectordb → generation pipeline
"""

import asyncio
import json
from app.services.form_filler import FormFiller
from app.services.cache_service import CacheService
import os
from dotenv import load_dotenv
import pytest
pytestmark = pytest.mark.asyncio

load_dotenv()

async def test_enhanced_form_filler():
    """Test the enhanced 3-tier form filling system"""
    
    print("🧪 Testing Enhanced Form Filler - 3-Tier System")
    print("=" * 60)
    
    # Initialize form filler
    openai_api_key = os.getenv("OPENAI_API_KEY")
    cache_service = CacheService()
    
    form_filler = FormFiller(
        openai_api_key=openai_api_key,
        cache_service=cache_service,
        headless=True
    )
    
    # Sample form fields to test
    test_fields = [
        {
            "selector": "#full_name",
            "field_type": "input",
            "field_purpose": "full_name",
            "name": "full_name"
        },
        {
            "selector": "#email",
            "field_type": "input", 
            "field_purpose": "email",
            "name": "email"
        },
        {
            "selector": "#current_company",
            "field_type": "input",
            "field_purpose": "current_company", 
            "name": "current_company"
        },
        {
            "selector": "#work_authorization",
            "field_type": "radio",
            "field_purpose": "work_authorization",
            "name": "work_authorization"
        },
        {
            "selector": "#desired_salary",
            "field_type": "input",
            "field_purpose": "desired_salary",
            "name": "desired_salary"
        },
        {
            "selector": "#cover_letter",
            "field_type": "textarea",
            "field_purpose": "cover_letter",
            "name": "cover_letter"
        }
    ]
    
    # Test with minimal user data to force vector database usage
    minimal_user_data = {
        "test_mode": True
    }
    
    print("📋 Test Fields:")
    for i, field in enumerate(test_fields, 1):
        print(f"   {i}. {field['field_purpose']} ({field['field_type']})")
    
    print(f"\n🔍 Testing 3-Tier Data Retrieval...")
    print("   1️⃣ Resume Vector Database Search")
    print("   2️⃣ Personal Info Vector Database Search") 
    print("   3️⃣ AI Generation (if needed)")
    
    try:
        # Test the enhanced field value generation
        result = await form_filler._generate_field_values(test_fields, minimal_user_data)
        
        if result["status"] == "success":
            print(f"\n✅ SUCCESS - Enhanced Form Filling Completed!")
            print(f"📊 Data Usage Statistics:")
            
            data_usage = result.get("data_usage", {})
            data_sources = result.get("data_sources_used", {})
            
            print(f"   📄 Resume Data Used: {data_sources.get('resume_vectordb', 0)} fields")
            print(f"   👤 Personal Info Used: {data_sources.get('personal_info_vectordb', 0)} fields")
            print(f"   🤖 Generated Content: {data_sources.get('generated', 0)} fields")
            print(f"   📝 Total Fields: {result.get('total_fields', 0)}")
            
            print(f"\n📝 Summary: {result.get('summary', 'No summary')}")
            
            print(f"\n🎯 Field Mappings:")
            for i, mapping in enumerate(result["values"], 1):
                field_purpose = mapping.get("field_purpose", "unknown")
                value = mapping.get("value", "")
                data_source = mapping.get("data_source", "unknown")
                action = mapping.get("action", "unknown")
                
                # Truncate long values for display
                display_value = value[:50] + "..." if len(str(value)) > 50 else value
                
                print(f"   {i}. {field_purpose}")
                print(f"      📊 Source: {data_source}")
                print(f"      🎯 Action: {action}")
                print(f"      💬 Value: {display_value}")
                print(f"      🧠 Reasoning: {mapping.get('reasoning', 'No reasoning')[:100]}...")
                print()
            
            # Calculate effectiveness metrics
            total_fields = len(result["values"])
            real_data_fields = data_sources.get('resume_vectordb', 0) + data_sources.get('personal_info_vectordb', 0)
            generated_fields = data_sources.get('generated', 0)
            
            if total_fields > 0:
                real_data_percentage = (real_data_fields / total_fields) * 100
                print(f"🎯 EFFECTIVENESS METRICS:")
                print(f"   📊 Real Data Usage: {real_data_percentage:.1f}% ({real_data_fields}/{total_fields} fields)")
                print(f"   🤖 Generation Required: {100-real_data_percentage:.1f}% ({generated_fields}/{total_fields} fields)")
                
                if real_data_percentage >= 70:
                    print(f"   🏆 EXCELLENT - High real data usage!")
                elif real_data_percentage >= 40:
                    print(f"   ✅ GOOD - Moderate real data usage")
                else:
                    print(f"   ⚠️  NEEDS IMPROVEMENT - Low real data usage")
        
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_form_filler()) 