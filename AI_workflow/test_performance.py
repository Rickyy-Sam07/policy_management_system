#!/usr/bin/env python3
"""
Performance Testing Script
Tests API performance, reliability, and concurrent handling
"""

import requests
import time
import json
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BASE_URL = "http://localhost:8001"
TOKEN = "eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Test data
SINGLE_QUESTION_DATA = {
    "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "questions": ["What is Newton's first law of motion?"]
}

MULTIPLE_QUESTIONS_DATA = {
    "documents": "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
    "questions": [
        "What is Newton's first law of motion?",
        "What is Newton's second law of motion?",
        "What is Newton's third law of motion?"
    ]
}

def make_request(data, timeout=60):
    """Make a single API request"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=HEADERS,
            json=data,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            if "answers" in result:
                return {
                    "success": True,
                    "time": elapsed,
                    "answers": len(result["answers"]),
                    "status_code": response.status_code
                }
        
        return {
            "success": False,
            "time": elapsed,
            "status_code": response.status_code,
            "error": response.text[:100]
        }
        
    except Exception as e:
        return {
            "success": False,
            "time": 0,
            "error": str(e)
        }

def performance_test():
    """Run comprehensive performance tests"""
    print("[*] PERFORMANCE TESTING")
    print("=" * 50)
    
    results = []
    
    # Test 1: Single question performance
    print("1. Single Question Test...")
    result = make_request(SINGLE_QUESTION_DATA)
    if result["success"]:
        print(f"   [SUCCESS]: {result['time']:.2f}s")
        results.append(result)
    else:
        print(f"   [FAIL]: {result.get('error', 'Unknown error')}")
        return False
    
    # Test 2: Multiple questions performance
    print("2. Multiple Questions Test...")
    result = make_request(MULTIPLE_QUESTIONS_DATA)
    if result["success"]:
        avg_per_question = result["time"] / result["answers"]
        print(f"   [SUCCESS]: {result['time']:.2f}s ({avg_per_question:.2f}s per question)")
        results.append(result)
    else:
        print(f"   [FAIL]: {result.get('error', 'Unknown error')}")
        return False
    
    # Test 3: Reliability test (multiple repeated requests)
    print("3. Reliability Test (5 repeated requests)...")
    reliability_results = []
    for i in range(5):
        print(f"   Request {i+1}/5...", end=" ")
        result = make_request(SINGLE_QUESTION_DATA)
        if result["success"]:
            print(f"[SUCCESS] {result['time']:.2f}s")
            reliability_results.append(result)
            results.append(result)
        else:
            print(f"[FAIL] {result.get('error', 'Unknown error')}")
    
    # Test 4: Concurrent requests
    print("4. Concurrent Test (3 simultaneous requests)...")
    concurrent_results = []
    
    def concurrent_request():
        return make_request(SINGLE_QUESTION_DATA)
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(concurrent_request) for _ in range(3)]
        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                concurrent_results.append(result)
                results.append(result)
    
    concurrent_time = time.time() - start_time
    successful_concurrent = len(concurrent_results)
    print(f"   Results: {successful_concurrent}/3 successful in {concurrent_time:.2f}s total")
    
    # Performance analysis
    if results:
        print(f"\n[*] PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Success rate
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = (successful_requests / total_requests) * 100
        print(f"[SUCCESS] Success Rate: {successful_requests}/{total_requests} ({success_rate:.1f}%)")
        
        # Response times
        times = [r["time"] for r in results if r["success"]]
        if times:
            avg_time = statistics.mean(times)
            median_time = statistics.median(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"[*] Response Times:")
            print(f"   Average: {avg_time:.2f}s")
            print(f"   Median: {median_time:.2f}s")
            print(f"   Min: {min_time:.2f}s")
            print(f"   Max: {max_time:.2f}s")
            
            # Per question analysis
            question_times = []
            for r in results:
                if r["success"] and "answers" in r:
                    question_times.append(r["time"] / r["answers"])
            
            if question_times:
                avg_per_question = statistics.mean(question_times)
                median_per_question = statistics.median(question_times)
                print(f"[*] Per Question Times:")
                print(f"   Average: {avg_per_question:.2f}s")
                print(f"   Median: {median_per_question:.2f}s")
            
            # Performance rating
            if avg_time < 10:
                print(f"[EXCELLENT]: Average response time under 10s")
            elif avg_time < 30:
                print(f"[GOOD]: Average response time under 30s")
            elif avg_time < 60:
                print(f"[ACCEPTABLE]: Average response time under 60s")
            else:
                print(f"[SLOW]: Average response time over 60s")
        
        print("\n" + "=" * 50)
        if success_rate >= 90 and avg_time < 30:
            print("[SUCCESS] PERFORMANCE TEST PASSED - API is performing well!")
        elif success_rate >= 80:
            print("[WARNING] Performance acceptable but could be improved")
        else:
            print("[FAIL] Performance issues detected")
        print("=" * 50)
        
        return success_rate >= 80
    
    return False

if __name__ == "__main__":
    success = performance_test()