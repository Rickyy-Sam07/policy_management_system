import requests
import json

# Test your API
api_url = input("Enter your ngrok URL (e.g. https://abc123.ngrok-free.app): ")
if not api_url.endswith('/hackrx/run'):
    api_url += '/hackrx/run'

# HackRX test request
data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d"
}

print(f"Testing: {api_url}")
response = requests.post(api_url, json=data, headers=headers)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    print("✅ SUCCESS!")
    print(json.dumps(result, indent=2))
else:
    print("❌ ERROR:")
    print(response.text)