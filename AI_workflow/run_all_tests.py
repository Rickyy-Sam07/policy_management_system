#!/usr/bin/env python3
"""
Master Test Runner
Executes all test suites and provides comprehensive reporting
"""

import subprocess
import time
import requests
import sys

# Configuration
BASE_URL = "http://localhost:8001"
TEST_SCRIPTS = [
    ("test_api_comprehensive.py", "Comprehensive API Tests"),
    ("test_hackrx_specific.py", "HackRX Specific Tests"),
    ("test_performance.py", "Performance Tests")
]

def check_api_server():
    """Check if API server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_test_script(script_name, description):
    """Run a single test script and capture results"""
    print(f"\n{'='*60}")
    print(f"[TEST] RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Print stdout if available
        if result.stdout:
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\n[*] Test completed in {elapsed:.1f}s")
        
        # Determine success based on return code
        success = result.returncode == 0
        status = "[PASS]" if success else "[FAIL]"
        print(f"[*] Result: {status}")
        
        return success, elapsed
        
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Test script timed out after 5 minutes")
        return False, 300
    except Exception as e:
        print(f"[ERROR] Failed to run test script: {str(e)}")
        return False, 0

def main():
    """Run all test suites"""
    print("[*] COMPREHENSIVE API TESTING SUITE")
    print("Testing all critical functionality for HackRX readiness")
    print("="*60)
    
    # Check if API server is running
    print("[*] Checking if API server is running...")
    if not check_api_server():
        print("[ERROR] API server is not running!")
        print("Please start the server with: python rtx3050_advanced_api.py")
        return False
    
    print("[SUCCESS] API server is running")
    
    # Run all test scripts
    results = {}
    total_time = 0
    
    for script_name, description in TEST_SCRIPTS:
        success, elapsed = run_test_script(script_name, description)
        results[description] = success
        total_time += elapsed
    
    # Final report
    print(f"\n{'='*60}")
    print("[*] FINAL TEST REPORT")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n[*] Overall Results: {passed}/{total} test suites passed")
    print(f"[*] Total Testing Time: {total_time:.1f}s")
    
    if passed == total:
        print(f"\n[SUCCESS] ALL TESTS PASSED!")
        print("[SUCCESS] API is ready for HackRX submission")
        print("[SUCCESS] All functionality verified")
    else:
        failed = total - passed
        print(f"\n[WARNING] {failed} test suite(s) failed")
        print("[ERROR] Fix issues before HackRX submission")
        print("[ERROR] Review failed tests above")
    
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)