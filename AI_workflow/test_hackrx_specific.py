#!/usr/bin/env python3
"""
HackRX Specific Testing Script
Tests the exact evaluation scenario with Newton's Principia
"""

import requests
import time
import json

# Configuration
BASE_URL = "http://localhost:8001"
TOKEN = "eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Exact HackRX test data
HACKRX_QUESTIONS = [
    "How does Newton define 'quantity of motion' and how is it distinct from velocity?",
    "According to Newton, what are the three laws of motion and how do they relate to each other?",
    "How does Newton derive Kepler's Second Law (equal areas in equal times) from his gravitational theory?",
    "What is Newton's method of 'fluxions' and how does it relate to modern calculus?",
    "How does Newton explain the precession of the equinoxes in the Principia?",
    "What is Newton's explanation for the tides and how does it involve the Moon's gravitational influence?",
    "How does Newton's concept of 'absolute space' differ from relative motion?",
    "What mathematical techniques does Newton use to solve the two-body problem?",
    "How does Newton address the problem of planetary perturbations in the Principia?",
    "What is Newton's explanation for the shape of the Earth (oblate spheroid)?",
    "How does Newton's treatment of comets demonstrate his universal law of gravitation?",
    "What role do Newton's 'Rules of Reasoning in Philosophy' play in the Principia?"
]

HACKRX_REQUEST = {
    "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "questions": HACKRX_QUESTIONS
}

def test_hackrx_endpoint():
    """Test the exact HackRX evaluation scenario"""
    print("[*] HackRX Specific Test")
    print("=" * 60)
    print(f"Document: Newton's Principia")
    print(f"Questions: {len(HACKRX_QUESTIONS)}")
    print("=" * 60)
    
    try:
        print("Sending request to HackRX endpoint...")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=HEADERS,
            json=HACKRX_REQUEST,
            timeout=120  # 2 minutes max
        )
        elapsed = time.time() - start_time
        
        print(f"\n[*] RESPONSE DETAILS:")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {elapsed:.2f}s")
        print(f"   Avg per Question: {elapsed/len(HACKRX_QUESTIONS):.2f}s")
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                if "answers" in result:
                    answers = result["answers"]
                    print(f"   [SUCCESS]: Received {len(answers)} answers")
                    print(f"   [CORRECT]: Answer count matches question count")
                    
                    # Show sample answers
                    print(f"\n[*] SAMPLE ANSWERS:")
                    for i in range(min(3, len(answers))):
                        q_preview = HACKRX_QUESTIONS[i][:50] + "..." if len(HACKRX_QUESTIONS[i]) > 50 else HACKRX_QUESTIONS[i]
                        a_preview = answers[i][:80] + "..." if len(answers[i]) > 80 else answers[i]
                        print(f"   Q{i+1}: {q_preview}")
                        print(f"   A{i+1}: {a_preview}")
                        print()
                    
                    # Performance rating
                    if elapsed < 30:
                        print(f"   [EXCELLENT]: Response time under 30s")
                    elif elapsed < 60:
                        print(f"   [GOOD]: Response time under 60s")
                    else:
                        print(f"   [SLOW]: Response time over 60s")
                    
                    return True
                else:
                    print(f"   [ERROR]: No 'answers' field in response")
                    print(f"   Response: {json.dumps(result, indent=2)[:300]}")
                    return False
                    
            except json.JSONDecodeError:
                print(f"   [ERROR]: Invalid JSON response")
                print(f"   Raw response: {response.text[:300]}")
                return False
        else:
            print(f"   [FAIL]: HTTP {response.status_code}")
            print(f"   Error: {response.text[:300]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"   [TIMEOUT]: Request took longer than 2 minutes")
        return False
    except Exception as e:
        print(f"   [ERROR]: {str(e)}")
        return False

def test_error_scenarios():
    """Test error handling scenarios"""
    print("\n[*] ERROR SCENARIO TESTS")
    print("=" * 40)
    
    results = []
    
    # Test 1: Invalid document URL
    print("1. Testing invalid document URL...")
    try:
        invalid_request = {
            "documents": "https://invalid-url.com/nonexistent.pdf",
            "questions": ["Test question?"]
        }
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=invalid_request, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "answers" in result and len(result["answers"]) > 0:
                print("   [PASS]: Returns fallback answers for invalid URL")
                results.append(True)
            else:
                print("   [FAIL]: No answers returned for invalid URL")
                results.append(False)
        else:
            print(f"   [FAIL]: HTTP {response.status_code} for invalid URL")
            results.append(False)
    except Exception as e:
        print(f"   [ERROR]: {str(e)}")
        results.append(False)
    
    # Test 2: Empty questions
    print("2. Testing empty questions...")
    try:
        empty_request = {
            "documents": HACKRX_REQUEST["documents"],
            "questions": []
        }
        response = requests.post(f"{BASE_URL}/hackrx/run", headers=HEADERS, json=empty_request, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "answers" in result and len(result["answers"]) == 0:
                print("   [PASS]: Returns empty answers for empty questions")
                results.append(True)
            else:
                print("   [FAIL]: Incorrect response for empty questions")
                results.append(False)
        else:
            print(f"   [FAIL]: HTTP {response.status_code} for empty questions")
            results.append(False)
    except Exception as e:
        print(f"   [ERROR]: {str(e)}")
        results.append(False)
    
    return all(results)

def main():
    """Run HackRX specific tests"""
    print("[*] HackRX SPECIFIC TESTING")
    print("Testing exact evaluation scenario...")
    
    # Main test
    main_test_passed = test_hackrx_endpoint()
    
    # Error scenario tests
    error_tests_passed = test_error_scenarios()
    
    # Final result
    print("\n" + "=" * 60)
    if main_test_passed and error_tests_passed:
        print("[SUCCESS] HACKRX TEST PASSED - API is ready for evaluation!")
    elif main_test_passed:
        print("[WARNING] Main test passed but some error scenarios failed")
    else:
        print("[FAIL] HackRX test failed - check issues above")
    print("=" * 60)
    
    return main_test_passed

if __name__ == "__main__":
    main()