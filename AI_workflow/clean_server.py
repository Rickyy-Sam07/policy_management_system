#!/usr/bin/env python3
"""
Clean FastAPI Server for RAG System with Eager Model Loading
Optimized for instant API responses with pre-loaded models
"""

import os
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

from clean_rag_system import create_rag_system

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clean RAG System", version="2.0.0")

# Global variables for system components
rag_system = None
startup_time = None
model_load_time = None

class BatchRequest(BaseModel):
    documents: str
    questions: List[str]

class BatchResponse(BaseModel):
    answers: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system and models during server startup"""
    global rag_system, startup_time, model_load_time
    
    logger.info("🚀 STARTING CLEAN RAG SERVER WITH EAGER LOADING")
    logger.info("=" * 60)
    
    startup_start = time.time()
    
    # Initialize RAG system
    api_key = os.getenv('GEMINI_API_KEY', 'your gemini api key')
    logger.info("📦 Initializing RAG system...")
    rag_system = create_rag_system(api_key)
    
    # Pre-load all models during startup
    logger.info("🧠 PRE-LOADING ALL MODELS (to avoid API delays)...")
    model_start = time.time()
    
    try:
        # This will load embedding model, reranker, and setup GPU
        rag_system.setup_models()
        model_load_time = time.time() - model_start
        
        logger.info(f"✅ ALL MODELS PRE-LOADED in {model_load_time:.2f}s")
        logger.info(f"   🎯 Device: {rag_system.device.upper()}")
        logger.info(f"   🧠 Embedding Model: BAAI/bge-small-en-v1.5")
        logger.info(f"   🔍 Reranker: BAAI/bge-reranker-base")
        if rag_system.device == "cuda":
            logger.info(f"   💾 GPU Memory: {rag_system.config['gpu_memory_fraction']*100}% allocated")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise e
    
    startup_time = time.time() - startup_start
    
    logger.info("🎉 SERVER READY FOR INSTANT API RESPONSES!")
    logger.info(f"⚡ Total startup time: {startup_time:.2f}s")
    logger.info(f"🚀 First API call will be INSTANT (no model loading delay)")
    logger.info("=" * 60)

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
    """Enhanced health check with model status"""
    global startup_time, model_load_time
    
    return {
        "status": "healthy",
        "server_info": {
            "startup_time": f"{startup_time:.2f}s" if startup_time else "N/A",
            "model_load_time": f"{model_load_time:.2f}s" if model_load_time else "N/A",
            "models_preloaded": True if rag_system and rag_system.embedding_model else False
        },
        "features": [
            "🎯 Eager model loading (instant API responses)",
            "🚀 GPU acceleration", 
            "⚡ Parallel sub-query processing",
            "🧠 Intelligent chunking",
            "🔍 Hybrid retrieval",
            "💾 Smart caching system"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Enhanced system statistics"""
    global startup_time, model_load_time
    
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    return {
        "system_info": {
            "startup_time": f"{startup_time:.2f}s" if startup_time else "N/A",
            "model_load_time": f"{model_load_time:.2f}s" if model_load_time else "N/A",
            "status": "ready_for_instant_responses"
        },
        "models": {
            "embedding_model_loaded": rag_system.embedding_model is not None,
            "reranker_loaded": rag_system.reranker is not None,
            "device": rag_system.device,
            "gpu_memory_fraction": rag_system.config.get('gpu_memory_fraction', 'N/A')
        },
        "cache": {
            "chunks_loaded": len(rag_system.chunks),
            "pdf_cache_entries": len(rag_system.pdf_cache),
            "embedding_cache_entries": len(rag_system.embedding_cache)
        },
        "config": rag_system.config
    }

if __name__ == "__main__":
    logger.info("🚀 LAUNCHING CLEAN RAG SERVER WITH EAGER LOADING...")
    logger.info("💡 Models will be pre-loaded during startup")
    logger.info("⚡ First API call will be INSTANT!")
    logger.info("🎯 Hardware: GPU detection enabled")
    logger.info("📡 Starting server on http://0.0.0.0:8001")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
