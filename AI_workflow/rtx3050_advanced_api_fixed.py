#!/usr/bin/env python3
"""
RTX 3050 Advanced Pipeline Production API Server
FastAPI integration with complete 5-stage pipeline
"""

import os
import time
import json
import hashlib
import requests
import fitz  # PyMuPDF
import concurrent.futures
from groq import Groq
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager

# Import the advanced pipeline
from rtx3050_advanced_pipeline import RTX3050AdvancedPipeline

# Global pipeline instance
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup"""
    global pipeline
    
    print("🚀 Starting RTX 3050 Advanced Pipeline Server...")
    
    # Get API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        raise RuntimeError("GROQ_API_KEY required")
    
    # Initialize pipeline
    pipeline = RTX3050AdvancedPipeline(groq_api_key)
    
    # Load documents
    pdf_files = [
        os.path.join(os.getcwd(), "doc1.pdf"),
        os.path.join(os.getcwd(), "doc2.pdf")
    ]
    
    # Check if vector index exists
    index_path = "vector_index"
    if os.path.exists(f"{index_path}.faiss") and os.path.exists(f"{index_path}.meta"):
        print("📂 Loading existing vector index...")
        if pipeline.load_vector_index(index_path):
            print("✅ Vector index loaded successfully")
        else:
            print("⚠️ Failed to load index, rebuilding...")
            if pipeline.load_documents(pdf_files):
                pipeline.build_vector_index()
                pipeline.save_vector_index(index_path)
    else:
        print("🏗️ Building new vector index...")
        if pipeline.load_documents(pdf_files):
            pipeline.build_vector_index()
            pipeline.save_vector_index(index_path)
    
    print("✅ RTX 3050 Advanced Pipeline ready!")
    
    yield
    
    print("🛑 Shutting down RTX 3050 Advanced Pipeline...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="RTX 3050 Advanced Insurance Query API",
    description="5-Stage Advanced Pipeline: Input → LLM Parser → Vector Search → Clause Matching → Logic Evaluation → JSON Output",
    version="2.0.0",
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
    """Flexible token verification for HackRX"""
    token = credentials.credentials
    
    # Accept multiple valid tokens for HackRX compatibility
    valid_tokens = [
        os.getenv('API_TOKEN', 'rtx3050-advanced-token'),
        'eb1793c521f670ca5d57867e68a3ae40418ae525d3dbd4bcaad8b8ff27b3998d',  # HackRX official token
        'rtx3050-advanced-token',  # Default token
        'hackrx-token',  # Alternative
    ]
    
    if token in valid_tokens:
        print(f"✅ Valid token accepted: {token[:10]}...")
        return token
    
    # Log failed attempts for debugging
    print(f"❌ Invalid token attempted: {token[:10]}...")
    raise HTTPException(status_code=401, detail="Invalid authentication token")

# Request models
class QueryRequest(BaseModel):
    question: str
    options: Optional[Dict[str, Any]] = {}

class BatchQueryRequest(BaseModel):
    questions: List[str]
    options: Optional[Dict[str, Any]] = {}

class DocumentUploadRequest(BaseModel):
    pdf_paths: List[str]
    rebuild_index: bool = True

class HackRXRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

# Response models
class QueryResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    performance: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "RTX 3050 Advanced Insurance Query API",
        "version": "2.0.0",
        "pipeline": "5-Stage Advanced Pipeline",
        "gpu_acceleration": "RTX 3050 6GB",
        "target_performance": "2-3 seconds, 90%+ accuracy",
        "status": "ready" if pipeline and pipeline.vector_index_ready else "initializing"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not pipeline:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Pipeline not initialized"}
        )
    
    stats = pipeline.get_performance_stats()
    
    return {
        "status": "healthy",
        "pipeline_ready": pipeline.vector_index_ready,
        "gpu_acceleration": stats['pipeline_status']['gpu_available'],
        "total_queries_processed": stats['performance_metrics']['total_queries'],
        "avg_response_time": round(stats['performance_metrics']['avg_response_time'], 3)
    }

@app.post("/query", response_model=QueryResponse)
async def process_single_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """Process a single insurance query with advanced 5-stage pipeline"""
    
    if not pipeline or not pipeline.vector_index_ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    start_time = time.time()
    
    try:
        print(f"📥 Processing query: {request.question[:100]}...")
        
        # Process with advanced pipeline
        result = pipeline.process_query(request.question)
        
        if result.get('error'):
            return QueryResponse(
                success=False,
                error=result.get('message', 'Processing error'),
                performance=result.get('performance')
            )
        
        processing_time = time.time() - start_time
        
        # Add API-level performance metrics
        if 'performance' not in result:
            result['performance'] = {}
        
        result['performance']['api_total_time'] = round(processing_time, 3)
        
        print(f"✅ Query processed in {processing_time:.3f}s")
        
        return QueryResponse(
            success=True,
            data=result,
            performance=result.get('performance')
        )
        
    except Exception as e:
        print(f"❌ Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# HackRX endpoint - Robust with better error handling
@app.post("/hackrx/run")
async def hackrx_run_robust(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """🚀 Robust HackRX endpoint with enhanced PDF processing"""
    
    print(f"📥 Processing HackRX request with {len(request.questions)} questions")
    print(f"📄 Document URL: {request.documents[:100]}...")
    
    try:
        # Step 1: Download document with retry
        doc_content = None
        for attempt in range(2):
            try:
                print(f"📡 Download attempt {attempt + 1}...")
                response = requests.get(request.documents, timeout=60, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                print(f"📊 Status: {response.status_code}, Size: {len(response.content)} bytes")
                
                if response.status_code == 200:
                    doc_content = response.content
                    break
            except Exception as e:
                print(f"❌ Download attempt {attempt + 1} failed: {e}")
                if attempt == 1:
                    raise Exception("Download failed after retries")
        
        # Step 2: OPTIMIZED RAG PIPELINE - PyMuPDF + GPU + Vector DB
        from rtx3050_optimized_rag import RTX3050OptimizedRAG
        
        rag_processor = RTX3050OptimizedRAG()
        
        # Process document with optimized pipeline
        if not rag_processor.process_document(doc_content):
            raise Exception("Optimized RAG processing failed")
        
        print(f"✅ Optimized RAG pipeline ready: {rag_processor.get_stats()}")
        
        # Step 3: RAG-based context retrieval
        use_knowledge = False  # Always use RAG for better accuracy
        
        # Step 4: OPTIMIZED 5-QUESTION BATCH PROCESSING
        groq_api_key = os.getenv('GROQ_API_KEY', 'gsk_2qfmcYifn6s6LPsgpSyj4GH1eM1_2F3NQNuZ7KUqjsEjHTwH')
        
        # Process questions in 5-question batches with 2s delays
        answers = rag_processor.process_questions_batch(request.questions, groq_api_key)
        
        print(f"⚡ Optimized batch processing completed: {len(answers)} answers")
        return {"answers": answers}
        
    except Exception as final_error:
        print(f"🚨 Final error: {final_error}")
        # Generate knowledge-based answers as final fallback
        return {"answers": [generate_smart_fallback(q) for q in request.questions]}

def generate_smart_fallback(question):
    """Generate generic fallback answers for any document type"""
    return f"I apologize, but I encountered an issue processing this question from the provided document. The question '{question}' requires specific information that may not be clearly available in the document excerpt I was able to analyze."

if __name__ == "__main__":
    import uvicorn
    
    try:
        print("🚀 Starting RTX 3050 Advanced API Server...")
        # Production server configuration with error handling
        uvicorn.run(
            "rtx3050_advanced_api_fixed:app",
            host="0.0.0.0",
            port=8001,
            workers=1,  # Single worker for RTX 3050 optimization
            access_log=True,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("🔧 Check GROQ_API_KEY and dependencies")