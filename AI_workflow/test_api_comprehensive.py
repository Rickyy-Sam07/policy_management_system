#!/usr/bin/env python3
"""
Comprehensive API Testing Script
Tests all critical endpoints and functionality
"""

import requests
import time
import json

# Test configuration
BASE_URL = "http://localhost:8001"
TOKEN = "eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Test data
HACKRX_TEST_DATA = {
    "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "questions": [
        "How does Newton define 'quantity of motion'?",
        "What are Newton's three laws of motion?",
        "Who was the grandfather of Isaac Newton?"
    ]
}

def test_endpoint(name, method, url, data=None, expected_status=200):
    """Test a single endpoint"""
    print(f"\n[TEST] {name}...")
    
    try:
        start_time = time.time()
        
        if method == "GET":
            response = requests.get(url, headers=HEADERS, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=HEADERS, json=data, timeout=60)
        
        elapsed = time.time() - start_time
        
        print(f"   Status: {response.status_code}")
        print(f"   Time: {elapsed:.2f}s")
        
        if response.status_code == expected_status:
            print(f"   [PASS]")
            
            # Show response preview
            try:
                resp_json = response.json()
                if isinstance(resp_json, dict):
                    if "answers" in resp_json:
                        print(f"   Answers: {len(resp_json['answers'])} received")
                    elif "status" in resp_json:
                        print(f"   Status: {resp_json['status']}")
                    elif "service" in resp_json:
                        print(f"   Service: {resp_json['service']}")
            except:
                pass
                
            return True
        else:
            print(f"   [FAIL] - Expected {expected_status}, got {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   [ERROR]: {str(e)}")
        return False

def run_comprehensive_tests():
    """Run all critical tests"""
    print("[*] Starting Comprehensive API Tests")
    print("=" * 50)
    
    results = {}
    
    # 1. Root endpoint
    results["root"] = test_endpoint(
        "Root Endpoint", 
        "GET", 
        f"{BASE_URL}/"
    )
    
    # 2. Health check
    results["health"] = test_endpoint(
        "Health Check", 
        "GET", 
        f"{BASE_URL}/health"
    )
    
    # 3. HackRX endpoint (most critical)
    results["hackrx"] = test_endpoint(
        "HackRX Endpoint", 
        "POST", 
        f"{BASE_URL}/hackrx/run",
        HACKRX_TEST_DATA
    )
    
    # 4. Authentication test (should fail)
    print(f"\n[TEST] Authentication Failure...")
    try:
        bad_headers = {"Authorization": "Bearer invalid-token", "Content-Type": "application/json"}
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=bad_headers, json=HACKRX_TEST_DATA, timeout=30)
        if response.status_code == 401:
            print(f"   [PASS] - Authentication properly rejected")
            results["auth"] = True
        else:
            print(f"   [FAIL] - Expected 401, got {response.status_code}")
            results["auth"] = False
    except Exception as e:
        print(f"   [ERROR]: {str(e)}")
        results["auth"] = False
    
    # 5. Stress test - multiple questions
    stress_data = {
        "documents": HACKRX_TEST_DATA["documents"],
        "questions": [
            "What is Newton's first law?",
            "What is Newton's second law?", 
            "What is Newton's third law?",
            "How does gravity work according to Newton?",
            "What mathematical tools did Newton use?"
        ]
    }
    
    results["stress"] = test_endpoint(
        "Stress Test (5 questions)", 
        "POST", 
        f"{BASE_URL}/hackrx/run",
        stress_data
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("[*] TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"   {test_name.upper()}: {status}")
    
    print(f"\n[*] Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - API is ready for HackRX!")
    else:
        print("[WARNING] Some tests failed - check issues above")
    
    return results

if __name__ == "__main__":
    run_comprehensive_tests()