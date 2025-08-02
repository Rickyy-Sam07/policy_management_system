#!/usr/bin/env python3
"""
RTX 3050 Distributed RAG Pipeline - Computer 1 (Main Server)
Document processing + Question splitting + Context sharing + Batching
"""

import os
import time
import numpy as np
import fitz  # PyMuPDF
import faiss
import torch
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import gc
import requests
from groq import Groq

class RTX3050DistributedRAG:
    def __init__(self, worker_url: str = None):
        """Initialize distributed RAG with worker computer URL"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vector_index = None
        self.texts = []
        self.metadata = []
        self.worker_url = worker_url  # Computer 2 ngrok URL
        
        print(f"🚀 RTX 3050 Distributed RAG on {self.device.upper()}")
        
        # Initialize embedding model with GPU
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print(f"✅ GPU model loaded on {self.device.upper()}")
        except Exception as e:
            print(f"⚠️ GPU failed, using CPU: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.device = 'cpu'
    
    def extract_text_worker(self, page_num: int, pdf_content: bytes) -> tuple:
        """Extract text from single page using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            if page_num < len(doc):
                page = doc[page_num]
                text = page.get_text()
                doc.close()
                return page_num, text.strip()
        except Exception as e:
            print(f"Error page {page_num}: {e}")
            return page_num, ""
        return page_num, ""
    
    def process_document(self, pdf_content: bytes) -> bool:
        """Process document with PyMuPDF parallel extraction"""
        print("📄 Starting PyMuPDF parallel extraction...")
        start_time = time.time()
        
        try:
            # Get total pages
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()
            
            print(f"📚 Processing {total_pages} pages with 8 workers")
            
            # Parallel extraction with 8 workers
            page_texts = {}
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(self.extract_text_worker, i, pdf_content): i 
                          for i in range(total_pages)}
                
                for future in as_completed(futures):
                    page_num, text = future.result()
                    if text:
                        page_texts[page_num] = text
            
            # Combine pages in order
            full_text = ""
            for page_num in sorted(page_texts.keys()):
                full_text += f"[Page {page_num+1}] {page_texts[page_num]}\n\n"
            
            print(f"✅ Extracted {len(full_text)} chars in {time.time() - start_time:.2f}s")
            
            # Create chunks
            chunks = self.create_chunks(full_text)
            print(f"📝 Created {len(chunks)} chunks")
            
            # Generate embeddings with GPU
            embeddings = self.generate_embeddings(chunks)
            if embeddings is None:
                return False
            
            # Build vector index
            self.build_vector_index(embeddings)
            
            # Store data
            self.texts = [chunk['text'] for chunk in chunks]
            self.metadata = [{'page': chunk['page']} for chunk in chunks]
            
            print(f"🎉 Pipeline completed in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return False
    
    def create_chunks(self, text: str, chunk_size: int = 300) -> List[Dict]:
        """Create text chunks"""
        import re
        
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_page = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for page markers
            page_match = re.search(r'\[Page (\d+)\]', sentence)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': current_page
                })
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': current_page
            })
        
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings with GPU acceleration"""
        print(f"🧠 Generating embeddings on {self.device.upper()}")
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device
                )
            
            print(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return None
    
    def build_vector_index(self, embeddings: np.ndarray):
        """Build FAISS vector index"""
        print("🔍 Building FAISS index...")
        
        try:
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.vector_index.add(embeddings.astype('float32'))
            
            print(f"✅ Built index with {self.vector_index.ntotal} vectors")
            
        except Exception as e:
            print(f"❌ Index error: {e}")
    
    def get_relevant_context(self, question: str, top_k: int = 3) -> str:
        """Get relevant context using vector search"""
        try:
            # Generate question embedding
            with torch.no_grad():
                question_embedding = self.model.encode([question], device=self.device)
            question_embedding = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)
            
            # Search
            scores, indices = self.vector_index.search(question_embedding.astype('float32'), top_k)
            
            # Get context
            context_parts = []
            for idx in indices[0]:
                if idx < len(self.texts):
                    context_parts.append(f"[Page {self.metadata[idx]['page']}] {self.texts[idx]}")
            
            return "\n\n".join(context_parts)[:2600]
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""
    
    def answer_question_local(self, question: str, context: str, client: Groq) -> str:
        """Answer question locally with GROQ API Key 1"""
        try:
            system_prompt = "You are a knowledgeable assistant. Answer questions directly and confidently based on the provided context. If the provided context contains relevant information, use it to answer the question. If the context is completely unrelated to the question or contains no relevant information, respond with exactly: 'This question is out of context.' Never use phrases like 'unfortunately', 'the text doesn't mention', or 'I cannot find' for context-related questions. Always provide a direct, factual response when context is relevant."
            
            if context:
                user_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                user_prompt = f"Question: {question}\n\nAnswer:"
            
            # Retry with incremental backoff: 2s, 3s, 4s
            for attempt in range(3):
                try:
                    response = client.chat.completions.create(
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
                        print(f"⏳ Computer 1 rate limit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise api_error
            
        except Exception as e:
            print(f"Computer 1 GROQ error: {e}")
            return f"Unable to process question due to API error."
    
    def send_questions_to_worker(self, questions_with_context: List[Dict]) -> List[str]:
        """Send questions with context to Computer 2 with increased timeout"""
        try:
            if not self.worker_url:
                print("⚠️ No worker URL configured, processing locally")
                return ["Worker not available"] * len(questions_with_context)
            
            print(f"📡 Sending {len(questions_with_context)} questions to Computer 2")
            
            # Increased timeout for large question batches
            timeout_seconds = max(120, len(questions_with_context) * 10)  # 10s per question, min 120s
            print(f"⏱️ Using {timeout_seconds}s timeout for {len(questions_with_context)} questions")
            
            response = requests.post(
                f"{self.worker_url}/process_questions",
                json={"questions_with_context": questions_with_context},
                timeout=timeout_seconds  # Increased from 60s
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Received {len(result['answers'])} answers from Computer 2")
                return result["answers"]
            else:
                print(f"❌ Worker error: {response.status_code}")
                return ["Worker error"] * len(questions_with_context)
                
        except Exception as e:
            print(f"❌ Worker communication error: {e}")
            return ["Worker communication failed"] * len(questions_with_context)
    
    def process_questions_distributed(self, questions: List[str], groq_api_key: str) -> List[str]:
        """Process questions distributed across two computers"""
        print(f"🚀 Processing {len(questions)} questions distributed across 2 computers")
        
        # Split questions in half
        mid_point = len(questions) // 2
        computer1_questions = questions[:mid_point]
        computer2_questions = questions[mid_point:]
        
        print(f"💻 Computer 1: {len(computer1_questions)} questions")
        print(f"💻 Computer 2: {len(computer2_questions)} questions")
        
        # Prepare questions with context for both computers
        computer1_data = []
        computer2_data = []
        
        # Get context for Computer 1 questions
        for q in computer1_questions:
            context = self.get_relevant_context(q)
            computer1_data.append({"question": q, "context": context})
        
        # Get context for Computer 2 questions
        for q in computer2_questions:
            context = self.get_relevant_context(q)
            computer2_data.append({"question": q, "context": context})
        
        # Process in parallel with increased timeout
        client = Groq(api_key=groq_api_key)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit Computer 1 processing
            future1 = executor.submit(self.process_local_questions_batched, computer1_data, client)
            
            # Submit Computer 2 processing
            future2 = executor.submit(self.send_questions_to_worker, computer2_data)
            
            # Collect results with longer timeout
            try:
                computer1_answers = future1.result(timeout=300)  # 5 minutes
            except Exception as e:
                print(f"❌ Computer 1 timeout: {e}")
                computer1_answers = ["Computer 1 timeout"] * len(computer1_questions)
            
            try:
                computer2_answers = future2.result(timeout=600)  # 10 minutes
            except Exception as e:
                print(f"❌ Computer 2 timeout: {e}")
                computer2_answers = ["Computer 2 timeout"] * len(computer2_questions)
        
        # Combine answers in original order
        all_answers = computer1_answers + computer2_answers
        
        print(f"✅ Distributed processing completed: {len(all_answers)} total answers")
        return all_answers
    
    def process_local_questions_batched(self, questions_with_context: List[Dict], client: Groq) -> List[str]:
        """Process questions locally on Computer 1 with batching and cooldown"""
        print(f"💻 Computer 1 processing {len(questions_with_context)} questions with batching")
        
        answers = []
        batch_size = 3  # Reduced to 3 for rate limit safety
        
        for i in range(0, len(questions_with_context), batch_size):
            batch = questions_with_context[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(questions_with_context) + batch_size - 1) // batch_size
            
            print(f"📦 Computer 1 batch {batch_num}/{total_batches}: {len(batch)} questions")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.answer_question_local, item["question"], item["context"], client): item for item in batch}
                
                batch_answers = []
                for future in as_completed(futures):
                    try:
                        answer = future.result(timeout=45)
                        batch_answers.append(answer)
                    except Exception as e:
                        print(f"Computer 1 batch error: {e}")
                        batch_answers.append("Unable to process question.")
                
                answers.extend(batch_answers)
            
            # Cooldown between batches (except last)
            if i + batch_size < len(questions_with_context):
                print("⏸️ Computer 1 cooling down 2s...")
                time.sleep(2)
        
        print(f"✅ Computer 1 completed {len(answers)} answers")
        return answers
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'device': self.device,
            'total_chunks': len(self.texts),
            'index_size': self.vector_index.ntotal if self.vector_index else 0,
            'worker_configured': self.worker_url is not None,
            'ready': self.vector_index is not None and len(self.texts) > 0
        }

if __name__ == "__main__":
    # Example usage
    worker_url = "https://your-computer2-ngrok-url.ngrok-free.app"
    processor = RTX3050DistributedRAG(worker_url=worker_url)
    print("🧪 Distributed RAG Processor ready")
    print(f"📊 Stats: {processor.get_stats()}")