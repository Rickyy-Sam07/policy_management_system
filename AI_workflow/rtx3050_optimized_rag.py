#!/usr/bin/env python3
"""
RTX 3050 Optimized RAG Pipeline
PyMuPDF + Parallel Execution + Vector DB + 4-Question Batches
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
from groq import Groq

class RTX3050OptimizedRAG:
    def __init__(self):
        """Initialize optimized RAG with GPU acceleration"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vector_index = None
        self.texts = []
        self.metadata = []
        
        print(f"🚀 RTX 3050 Optimized RAG on {self.device.upper()}")
        
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
    
    def answer_question(self, question: str, client: Groq) -> str:
        """Answer single question with RAG context and incremental retry"""
        try:
            context = self.get_relevant_context(question)
            
            system_prompt = "You are a knowledgeable assistant. Answer questions directly and confidently. If the provided context doesn't contain the exact information, use your knowledge to provide an accurate answer. Never use phrases like 'unfortunately', 'the text doesn't mention', or 'I cannot find'. Always provide a direct, factual response."
            
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
                        wait_time = 2 + attempt  # 2s, 3s, 4s
                        print(f"⏳ Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise api_error
            
        except Exception as e:
            print(f"GROQ error: {e}")
            return f"Unable to process question due to API error."
    
    def process_questions_batch(self, questions: List[str], groq_api_key: str) -> List[str]:
        """Process questions in 4-question batches with 2s delays"""
        print(f"🚀 Processing {len(questions)} questions in batches of 4")
        
        client = Groq(api_key=groq_api_key)
        answers = []
        batch_size = 4
        
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📦 Processing batch {batch_num}: {len(batch)} questions")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(self.answer_question, q, client): q for q in batch}
                
                batch_answers = []
                for future in as_completed(futures):
                    try:
                        answer = future.result(timeout=45)
                        batch_answers.append(answer)
                    except Exception as e:
                        print(f"Batch error: {e}")
                        batch_answers.append("Unable to process question.")
                
                answers.extend(batch_answers)
            
            # Wait 2 seconds between batches
            if i + batch_size < len(questions):
                print("⏸️ Waiting 2s before next batch...")
                time.sleep(2)
        
        print(f"✅ Completed {len(answers)} answers")
        return answers
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'device': self.device,
            'total_chunks': len(self.texts),
            'index_size': self.vector_index.ntotal if self.vector_index else 0,
            'ready': self.vector_index is not None and len(self.texts) > 0
        }

if __name__ == "__main__":
    processor = RTX3050OptimizedRAG()
    print("🧪 Optimized RAG Processor ready")
    print(f"📊 Stats: {processor.get_stats()}")