
import requests
import json
import time

def test_hackrx_official_input():
    """Test with the official HackRX input format"""
    
    print(" TESTING API WITH OFFICIAL HACKRX INPUT")
    print("=" * 60)
    
    # Your current public endpoint
    api_endpoint = "https://19f2590d8cd3.ngrok-free.app/hackrx/run"
    
    # Official HackRX headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer rtx3050-advanced-token"  # Your API key
    }
    
    # Official HackRX request body
    request_body = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    print(f" REQUEST DETAILS:")
    print(f"   Endpoint: {api_endpoint}")
    print(f"   Method: POST")
    print(f"   Content-Type: {headers['Content-Type']}")
    print(f"   Accept: {headers['Accept']}")
    print(f"   Authorization: {headers['Authorization']}")
    print(f"   Document: {request_body['documents'][:60]}...")
    print(f"   Questions: {len(request_body['questions'])}")
    print()
    
    # Display all questions
    print(" QUESTIONS TO PROCESS:")
    for i, question in enumerate(request_body['questions'], 1):
        print(f"   {i:2d}. {question}")
    print()
    
    # Make the API request
    print(" Making API request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            json=request_body,
            timeout=180  # 3 minutes timeout
        )
        
        request_time = time.time() - start_time
        
        print(f"⏱ Request completed in {request_time:.2f}s")
        print(f" Status Code: {response.status_code}")
        print()
        
        if response.status_code == 200:
            result = response.json()
            
            print(" SUCCESS! API RESPONSE RECEIVED")
            print("=" * 60)
            
            # Check response format
            if "answers" in result and isinstance(result["answers"], list):
                answers = result["answers"]
                
                print(f" RESPONSE FORMAT: {{ \"answers\": [...] }} ✅")
                print(f" Questions sent: {len(request_body['questions'])}")
                print(f" Answers received: {len(answers)}")
                print(f" Average per question: {request_time/len(request_body['questions']):.2f}s")
                print()
                
                # Display all answers
                print(" COMPLETE ANSWERS:")
                print("-" * 60)
                
                for i, (question, answer) in enumerate(zip(request_body['questions'], answers), 1):
                    print(f"\n{i:2d}. Q: {question}")
                    print(f"    A: {answer}")
                
                # Performance analysis
                avg_time = request_time / len(request_body['questions'])
                
                print("\n" + "=" * 60)
                print(" PERFORMANCE ANALYSIS")
                print("-" * 30)
                print(f"⏱ Total Processing Time: {request_time:.2f}s")
                print(f"⏱ Average per Question: {avg_time:.2f}s")
                print(f" Success Rate: {len(answers)}/{len(request_body['questions'])}")
                
                if avg_time <= 2.0:
                    print(" EXCELLENT: Sub-2s performance!")
                elif avg_time <= 3.0:
                    print(" GOOD: Under 3s target achieved!")
                else:
                    print(" Performance could be improved")
                
                # Check for performance info
                if "_performance" in result:
                    perf = result["_performance"]
                    print(f" API Performance Status: {perf.get('status', 'N/A')}")
                    print(f" Optimizations Applied: {perf.get('optimizations', [])}")
                
                # Save results
                test_results = {
                    "test_type": "official_hackrx_input",
                    "endpoint": api_endpoint,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "performance": {
                        "total_time": request_time,
                        "avg_per_question": avg_time,
                        "questions_count": len(request_body['questions']),
                        "answers_count": len(answers)
                    },
                    "request": {
                        "headers": headers,
                        "body": request_body
                    },
                    "response": result,
                    "status": "success"
                }
                
                with open('hackrx_official_test_results.json', 'w', encoding='utf-8') as f:
                    json.dump(test_results, f, indent=2, ensure_ascii=False)
                
                print(f"\n Complete results saved to: hackrx_official_test_results.json")
                print(" OFFICIAL HACKRX TEST COMPLETE!")
                
                return True
                
            else:
                print(" UNEXPECTED RESPONSE FORMAT")
                print(f"Expected: {{\"answers\": [...]}}")
                print(f"Received: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
                
        else:
            print(f" HTTP ERROR {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(" Request timed out after 180 seconds")
        return False
    except requests.exceptions.ConnectionError:
        print(" Connection error - check if ngrok tunnel is still active")
        return False
    except Exception as e:
        print(f" Error: {e}")
        return False

if __name__ == "__main__":
    print(" OFFICIAL HACKRX API TEST")
    print("Testing with exact input format provided")
    print()
    
    success = test_hackrx_official_input()
    
    if success:
        print("\n YOUR API IS READY FOR HACKRX SUBMISSION!")
    else:
        print("\n Please check your API and try again")
