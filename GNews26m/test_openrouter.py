#!/usr/bin/env python3
"""
Test script to verify OpenRouter integration works correctly.
"""

import os
from openai import OpenAI

def test_openrouter_connection():
    """Test basic OpenRouter API connection."""
    
    # Check for API key
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not api_key:
        print("❌ No OPENROUTER_API_KEY or OPENROUTER_KEY found in environment")
        return False
    
    print(f"✅ Found API key: {api_key[:8]}...")
    
    # Create client
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        print("✅ OpenAI client created successfully")
    except Exception as e:
        print(f"❌ Failed to create OpenAI client: {e}")
        return False
    
    # Test simple request
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # Free tier model
            messages=[
                {"role": "system", "content": "Rate news bias from -3.0 to +3.0. Number only."},
                {"role": "user", "content": 'Rate bias: "Breaking news: The economy is performing well according to latest reports."'},
            ],
            max_tokens=5,
            temperature=0.7
        )
        
        result = response.choices[0].message.content
        print(f"✅ API call successful: '{result}'")
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing OpenRouter Integration")
    print("=" * 40)
    
    success = test_openrouter_connection()
    
    print("=" * 40)
    if success:
        print("🎉 OpenRouter integration test PASSED")
    else:
        print("💥 OpenRouter integration test FAILED")