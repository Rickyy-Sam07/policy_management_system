#!/usr/bin/env python3
"""
RTX 3050 Distributed API Server - Computer 1 Main Server
Integrates distributed RAG with HackRX endpoint
"""

import os
import time
import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Import distributed RAG
from rtx3050_distributed_rag import RTX3050DistributedRAG

# Global processor instance
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize distributed processor on startup"""
    global processor
    
    print("🚀 Starting RTX 3050 Distributed API Server...")
    
    # Get Computer 2 worker URL from environment
    worker_url = os.getenv('COMPUTER2_WORKER_URL', 'https://your-computer2-ngrok-url.ngrok-free.app')
    
    # Initialize distributed processor
    processor = RTX3050DistributedRAG(worker_url=worker_url)
    
    print(f"✅ Distributed RAG initialized with worker: {worker_url}")
    
    yield
    
    print("🛑 Shutting down Distributed API Server...")

# Create FastAPI app
app = FastAPI(
    title="RTX 3050 Distributed API",
    description="Distributed processing across 2 computers with different GROQ API keys",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify HackRX token"""
    token = credentials.credentials
    
    valid_tokens = [
        'eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d',  # HackRX official
        'rtx3050-distributed-token'  # Default
    ]
    
    if token in valid_tokens:
        return token
    
    raise HTTPException(status_code=401, detail="Invalid authentication token")

# Request models
class HackRXRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "RTX 3050 Distributed API",
        "version": "1.0.0",
        "architecture": "2-Computer Distributed Processing",
        "computer1": "Document processing + Question splitting",
        "computer2": "Question processing with different GROQ API",
        "status": "ready" if processor else "initializing"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not processor:
        return {"status": "unhealthy", "reason": "Processor not initialized"}
    
    stats = processor.get_stats()
    
    return {
        "status": "healthy",
        "processor_ready": stats['ready'],
        "worker_configured": stats['worker_configured'],
        "device": stats['device'],
        "total_chunks": stats['total_chunks']
    }

@app.post("/hackrx/run")
async def hackrx_distributed_run(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """🚀 HackRX endpoint with distributed processing"""
    
    print(f"📥 Processing HackRX request with {len(request.questions)} questions (DISTRIBUTED)")
    print(f"📄 Document URL: {request.documents[:100]}...")
    
    try:
        # Step 1: Download document
        print("📡 Downloading document...")
        response = requests.get(request.documents, timeout=60)
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code}")
        
        doc_content = response.content
        print(f"📊 Downloaded: {len(doc_content)} bytes")
        
        # Step 2: Process document on Computer 1
        print("🔄 Processing document on Computer 1...")
        if not processor.process_document(doc_content):
            raise Exception("Document processing failed")
        
        # Step 3: Distributed question processing
        groq_api_key = os.getenv('GROQ_API_KEY', 'gsk_2qfmcYifn6s6LPsgpSyj4GH1eM1_2F3NQNuZ7KUqjsEjHTwH')
        
        print("🚀 Starting distributed question processing...")
        answers = processor.process_questions_distributed(request.questions, groq_api_key)
        
        print(f"✅ Distributed processing completed: {len(answers)} answers")
        return {"answers": answers}
        
    except Exception as e:
        print(f"🚨 Distributed processing error: {e}")
        # Fallback to single computer processing
        try:
            print("🔄 Falling back to single computer processing...")
            from rtx3050_optimized_rag import RTX3050OptimizedRAG
            
            fallback_processor = RTX3050OptimizedRAG()
            if fallback_processor.process_document(doc_content):
                groq_api_key = os.getenv('GROQ_API_KEY', 'gsk_2qfmcYifn6s6LPsgpSyj4GH1eM1_2F3NQNuZ7KUqjsEjHTwH')
                answers = fallback_processor.process_questions_batch(request.questions, groq_api_key)
                return {"answers": answers}
        except:
            pass
        
        # Final fallback
        return {"answers": [f"Processing error occurred for question {i+1}" for i in range(len(request.questions))]}

if __name__ == "__main__":
    import uvicorn
    
    try:
        print("🚀 Starting RTX 3050 Distributed API Server...")
        print("💡 Set COMPUTER2_WORKER_URL environment variable")
        print("💡 Set GROQ_API_KEY environment variable")
        
        uvicorn.run(
            "rtx3050_distributed_api:app",
            host="0.0.0.0",
            port=8001,
            workers=1,
            access_log=True,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Distributed API Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("🔧 Check environment variables")