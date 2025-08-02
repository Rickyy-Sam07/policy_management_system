#!/usr/bin/env python3
"""
Quick Test Script
Fast validation of core functionality
"""

import requests
import time

# Configuration
BASE_URL = "http://localhost:8001"
TOKEN = "eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def quick_test():
    """Run essential tests quickly"""
    print("⚡ QUICK API TEST")
    print("=" * 30)
    
    # Test 1: Health check
    print("1. Health check...", end=" ")
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS, timeout=10)
        if response.status_code == 200:
            print("✅")
        else:
            print(f"❌ ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ ({e})")
        return False
    
    # Test 2: HackRX endpoint with single question
    print("2. HackRX endpoint...", end=" ")
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
        "questions": ["What is Newton's first law of motion?"]
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=test_data, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if "answers" in data and len(data["answers"]) == 1:
                print(f"✅ ({elapsed:.1f}s)")
                print(f"   Answer: {data['answers'][0][:50]}...")
                return True
            else:
                print("❌ (Invalid response format)")
                return False
        else:
            print(f"❌ ({response.status_code})")
            return False
            
    except Exception as e:
        print(f"❌ ({e})")
        return False

if __name__ == "__main__":
    success = quick_test()
    
    print("\n" + "=" * 30)
    if success:
        print("🎉 QUICK TEST PASSED!")
        print("API is working correctly")
    else:
        print("❌ QUICK TEST FAILED!")
        print("Check API server and fix issues")
    print("=" * 30)