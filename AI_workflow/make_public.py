#!/usr/bin/env python3
"""
🌐 Simple Ngrok Public Hosting
Make your API accessible to anyone for testing
"""

from pyngrok import ngrok
import time
import json

def setup_public_api():
    """Set up public access via ngrok"""
    
    print("🌐 SETTING UP PUBLIC API ACCESS")
    print("=" * 50)
    
    # Get authtoken
    authtoken = input("Enter your ngrok authtoken: ").strip()
    
    if not authtoken:
        print("❌ No authtoken provided")
        print("💡 Get one from: https://dashboard.ngrok.com/get-started/your-authtoken")
        return
    
    # Set authtoken
    ngrok.set_auth_token(authtoken)
    
    # Create tunnel
    try:
        print("🚀 Creating public tunnel...")
        tunnel = ngrok.connect(8001)
        public_url = str(tunnel.public_url)
        
        print(f"✅ SUCCESS! Your API is now public!")
        print("=" * 50)
        print(f"🌐 Public URL: {public_url}")
        print(f"🎯 API Endpoint: {public_url}/hackrx/run")
        print("=" * 50)
        
        # Create shareable info
        api_info = {
            "public_endpoint": f"{public_url}/hackrx/run",
            "method": "POST",
            "authentication": "Bearer rtx3050-advanced-token",
            "content_type": "application/json",
            "request_example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?",
                    "Does this policy cover maternity expenses?"
                ]
            },
            "expected_response": {
                "answers": [
                    "Answer to question 1",
                    "Answer to question 2", 
                    "Answer to question 3"
                ]
            },
            "performance": "~1.09s per question",
            "status": "ready_for_testing"
        }
        
        # Save public API info
        with open('public_api_ready.json', 'w', encoding='utf-8') as f:
            json.dump(api_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Public API info saved: public_api_ready.json")
        print()
        print(f"📋 SHARE THIS WITH TESTERS:")
        print(f"   Endpoint: {public_url}/hackrx/run")
        print(f"   Method: POST")
        print(f"   Auth: Bearer rtx3050-advanced-token")
        print(f"   Format: {{\"answers\": [...]}}")
        print()
        print(f"🔗 Ngrok Dashboard: http://127.0.0.1:4040")
        print(f"⚡ Press Ctrl+C to stop")
        
        # Keep tunnel alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping public tunnel...")
            ngrok.disconnect_all()
            print("✅ Tunnel stopped")
            
    except Exception as e:
        print(f"❌ Failed to create tunnel: {e}")
        print(f"💡 Make sure your API server is running on localhost:8001")

if __name__ == "__main__":
    print("🌐 NGROK PUBLIC API SETUP")
    print("=" * 50)
    print("This will make your API accessible from anywhere")
    print("Make sure your API server is running first!")
    print()
    
    setup_public_api()
