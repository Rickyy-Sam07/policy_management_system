#!/usr/bin/env python3
"""
Computer 2 Worker Server
Receives questions with context from Computer 1 and processes with GROQ API Key 2
WITH BATCHING AND COOLDOWN SYSTEM
"""

import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

# FastAPI app
app = FastAPI(
    title="Computer 2 Worker Server",
    description="Processes questions with context using GROQ API Key 2 with batching",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QuestionWithContext(BaseModel):
    question: str
    context: str

class WorkerRequest(BaseModel):
    questions_with_context: List[QuestionWithContext]

# Response model
class WorkerResponse(BaseModel):
    answers: List[str]
    processing_time: float
    questions_processed: int

class Computer2Worker:
    def __init__(self):
        """Initialize Computer 2 worker with GROQ API Key 2"""
        self.groq_api_key = os.getenv('GROQ_API_KEY_2')  # Different API key
        if not self.groq_api_key:
            print("❌ GROQ_API_KEY_2 environment variable not set")
            raise RuntimeError("GROQ_API_KEY_2 required for Computer 2")
        
        self.client = Groq(api_key=self.groq_api_key)
        print("🚀 Computer 2 Worker initialized with GROQ API Key 2")
    
    def answer_question(self, question: str, context: str) -> str:
        """Answer single question with context using GROQ API Key 2"""
        try:
            system_prompt = "You are a knowledgeable assistant. Answer questions directly and confidently based on the provided context. If the provided context contains relevant information, use it to answer the question. If the context is completely unrelated to the question or contains no relevant information, respond with exactly: 'This question is out of context.' Never use phrases like 'unfortunately', 'the text doesn't mention', or 'I cannot find' for context-related questions. Always provide a direct, factual response when context is relevant."
            
            if context:
                user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                user_prompt = f"Question: {question}\n\nAnswer:"
            
            # Retry with incremental backoff: 2s, 3s, 4s
            for attempt in range(3):
                try:
                    response = self.client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=45,
                        temperature=0.0,
                        timeout=10
                    )
                    
                    return response.choices[0].message.content.strip()
                    
                except Exception as api_error:
                    error_str = str(api_error)
                    if "429" in error_str and attempt < 2:
                        wait_time = 2 + attempt
                        print(f"⏳ Computer 2 rate limit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise api_error
            
        except Exception as e:
            print(f"Computer 2 GROQ error: {e}")
            return f"Unable to process question due to API error."
    
    def process_questions_batched(self, questions_with_context: List[Dict]) -> List[str]:
        """Process multiple questions with context using batching and cooldown"""
        print(f"💻 Computer 2 processing {len(questions_with_context)} questions with batching")
        
        answers = []
        batch_size = 3  # Reduced to 3 for rate limit safety
        
        for i in range(0, len(questions_with_context), batch_size):
            batch = questions_with_context[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(questions_with_context) + batch_size - 1) // batch_size
            
            print(f"📦 Computer 2 batch {batch_num}/{total_batches}: {len(batch)} questions")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.answer_question, item["question"], item["context"]): item for item in batch}
                
                batch_answers = []
                for future in as_completed(futures):
                    try:
                        answer = future.result(timeout=45)
                        batch_answers.append(answer)
                    except Exception as e:
                        print(f"Computer 2 batch error: {e}")
                        batch_answers.append("Unable to process question.")
                
                answers.extend(batch_answers)
            
            # Cooldown between batches (except last)
            if i + batch_size < len(questions_with_context):
                print("⏸️ Computer 2 cooling down 2s...")
                time.sleep(2)
        
        print(f"✅ Computer 2 completed {len(answers)} answers")
        return answers

# Initialize worker
worker = Computer2Worker()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Computer 2 Worker Server",
        "status": "ready",
        "groq_api_configured": worker.groq_api_key is not None,
        "batching_enabled": True
    }

@app.post("/process_questions", response_model=WorkerResponse)
async def process_questions(request: WorkerRequest):
    """Process questions with context from Computer 1 using batching"""
    
    start_time = time.time()
    
    try:
        print(f"📥 Received {len(request.questions_with_context)} questions from Computer 1")
        
        # Convert Pydantic models to dict
        questions_data = []
        for item in request.questions_with_context:
            questions_data.append({
                "question": item.question,
                "context": item.context
            })
        
        # Process questions with batching
        answers = worker.process_questions_batched(questions_data)
        
        processing_time = time.time() - start_time
        
        print(f"📤 Sending {len(answers)} answers back to Computer 1")
        
        return WorkerResponse(
            answers=answers,
            processing_time=processing_time,
            questions_processed=len(answers)
        )
        
    except Exception as e:
        print(f"❌ Worker processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "groq_api_key_configured": worker.groq_api_key is not None,
        "service": "Computer 2 Worker",
        "batching_enabled": True,
        "batch_size": 3,
        "cooldown_seconds": 2
    }

if __name__ == "__main__":
    import uvicorn
    
    try:
        print("🚀 Starting Computer 2 Worker Server with batching...")
        print("💡 Make sure to set GROQ_API_KEY_2 environment variable")
        print("🌐 Use ngrok to expose this server to Computer 1")
        print("📦 Batching: 3 questions per batch with 2s cooldown")
        
        # Start server on port 8002
        uvicorn.run(
            "computer2_worker:app",
            host="0.0.0.0",
            port=8002,
            workers=1,
            access_log=True,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Computer 2 Worker stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("🔧 Check GROQ_API_KEY_2 environment variable")