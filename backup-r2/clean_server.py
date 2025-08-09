#!/usr/bin/env python3
"""
Clean FastAPI Server for RAG System
Simple, efficient, and focused
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

from clean_rag_system import create_rag_system

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clean RAG System", version="1.0.0")

# Initialize RAG system
api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCHgFhB-3WJeN1ld0MK2a0j8geEMO56anw')
rag_system = create_rag_system(api_key)

class BatchRequest(BaseModel):
    documents: str
    questions: List[str]

class BatchResponse(BaseModel):
    answers: List[str]

@app.post("/process-batch", response_model=BatchResponse)
async def process_batch(request: BatchRequest):
    """Process documents and answer questions"""
    try:
        logger.info(f"📄 Processing document: {request.documents[:50]}...")
        
        # Process document
        success = rag_system.process_document(request.documents)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        logger.info(f"❓ Processing {len(request.questions)} questions...")
        
        # Process questions
        answers = rag_system.process_questions(request.questions)
        
        logger.info("✅ Request completed successfully")
        return BatchResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hackrx/run", response_model=BatchResponse)
async def hackrx_run(request: BatchRequest):
    """HackRX endpoint - same as process-batch"""
    return await process_batch(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "features": [
            "GPU acceleration",
            "Parallel sub-query processing", 
            "Intelligent chunking",
            "Hybrid retrieval"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "chunks_loaded": len(rag_system.chunks),
        "device": rag_system.device,
        "config": rag_system.config
    }

if __name__ == "__main__":
    logger.info("🚀 Starting Clean RAG Server...")
    logger.info("🧠 Features: GPU acceleration + Parallel processing")
    logger.info("🎯 Hardware: GPU detection enabled")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
