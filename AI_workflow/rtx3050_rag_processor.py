#!/usr/bin/env python3
"""
RTX 3050 GPU-Accelerated RAG Pipeline
Optimized for 500+ page documents with semantic search
"""

import os
import time
import hashlib
import numpy as np
import fitz  # PyMuPDF
import faiss
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import re

class RTX3050RAGProcessor:
    def __init__(self):
        """Initialize GPU-accelerated RAG processor"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vector_index = None
        self.chunks = []
        self.chunk_metadata = []
        
        print(f"🚀 RTX 3050 RAG Processor initializing on {self.device.upper()}")
        
        # Initialize embedding model with GPU acceleration
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print(f"✅ Embedding model loaded on {self.device.upper()}")
        except Exception as e:
            print(f"⚠️ GPU model failed, falling back to CPU: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.device = 'cpu'
    
    def extract_text_parallel(self, pdf_content: bytes) -> str:
        """Extract text from PDF using parallel processing"""
        print("📄 Starting parallel PDF text extraction...")
        start_time = time.time()
        
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()
            
            print(f"📚 Processing {total_pages} pages with parallel extraction")
            
            def extract_page_text(page_num):
                """Extract text from single page"""
                try:
                    doc = fitz.open(stream=pdf_content, filetype="pdf")
                    page = doc[page_num]
                    text = page.get_text()
                    doc.close()
                    return page_num, text.strip() if text else ""
                except:
                    return page_num, ""
            
            # Parallel extraction with 4 workers
            page_texts = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_page = {
                    executor.submit(extract_page_text, page_num): page_num 
                    for page_num in range(total_pages)
                }
                
                for future in concurrent.futures.as_completed(future_to_page):
                    page_num, text = future.result()
                    if text:
                        page_texts[page_num] = text
            
            # Combine all pages in order
            full_text = ""
            for page_num in sorted(page_texts.keys()):
                full_text += f"[Page {page_num+1}] {page_texts[page_num]}\n\n"
            
            extraction_time = time.time() - start_time
            print(f"✅ Extracted {len(full_text)} chars from {len(page_texts)} pages in {extraction_time:.2f}s")
            
            return full_text
            
        except Exception as e:
            print(f"❌ PDF extraction error: {e}")
            return ""
    
    def chunk_text_parallel(self, text: str, chunk_size: int = 200) -> List[Dict]:
        """Split text into chunks with parallel processing"""
        print(f"✂️ Chunking text into {chunk_size}-token segments...")
        start_time = time.time()
        
        # Simple sentence-based chunking for better semantic coherence
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        page_pattern = r'\[Page (\d+)\]'
        current_page = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check for page markers
            page_match = re.search(page_pattern, sentence)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            sentence_tokens = len(sentence) // 4
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Save current chunk - minimal data only
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': current_page
                })
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'page': current_page
            })
        
        chunk_time = time.time() - start_time
        print(f"✅ Created {len(chunks)} chunks in {chunk_time:.2f}s")
        
        return chunks
    
    def generate_embeddings_gpu(self, chunks: List[Dict], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings using GPU acceleration"""
        print(f"🧠 Generating embeddings on {self.device.upper()} (batch size: {batch_size})")
        start_time = time.time()
        
        texts = [chunk['text'] for chunk in chunks]
        
        try:
            # GPU-accelerated batch processing - optimized
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,  # Disabled for speed
                convert_to_numpy=True,
                device=self.device
            )
            
            embedding_time = time.time() - start_time
            print(f"✅ Generated {len(embeddings)} embeddings ({embeddings.shape[1]}D) in {embedding_time:.2f}s")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return np.array([])
    
    def build_vector_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS vector index"""
        print("🔍 Building FAISS vector index...")
        start_time = time.time()
        
        try:
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for cosine similarity (faster for small datasets)
            index = faiss.IndexFlatIP(dimension)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            index.add(embeddings.astype('float32'))
            
            index_time = time.time() - start_time
            print(f"✅ Built FAISS index with {index.ntotal} vectors in {index_time:.2f}s")
            
            return index
            
        except Exception as e:
            print(f"❌ Index building error: {e}")
            return None
    
    def process_document(self, pdf_content: bytes) -> bool:
        """Complete document processing pipeline"""
        print("🚀 Starting RAG document processing pipeline...")
        total_start = time.time()
        
        # Stage 1: Extract text
        text = self.extract_text_parallel(pdf_content)
        if not text:
            print("❌ No text extracted from document")
            return False
        
        # Stage 2: Chunk text
        self.chunks = self.chunk_text_parallel(text)
        if not self.chunks:
            print("❌ No chunks created")
            return False
        
        # Stage 3: Generate embeddings
        embeddings = self.generate_embeddings_gpu(self.chunks)
        if embeddings.size == 0:
            print("❌ No embeddings generated")
            return False
        
        # Stage 4: Build vector index
        self.vector_index = self.build_vector_index(embeddings)
        if self.vector_index is None:
            print("❌ Failed to build vector index")
            return False
        
        # Store minimal metadata only
        self.chunk_metadata = [
            {'page': chunk['page']}
            for chunk in self.chunks
        ]
        
        total_time = time.time() - total_start
        print(f"🎉 RAG pipeline completed in {total_time:.2f}s")
        print(f"📊 Stats: {len(self.chunks)} chunks, {len(embeddings)} embeddings, {self.vector_index.ntotal} vectors")
        
        return True
    
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Perform semantic search using vector similarity - LOCAL ONLY"""
        if not self.vector_index or not self.model:
            return []
        
        try:
            # Generate query embedding locally (no API calls)
            with torch.no_grad():  # Disable gradients for inference
                query_embedding = self.model.encode(
                    [query], 
                    convert_to_numpy=True, 
                    device=self.device,
                    show_progress_bar=False,  # Disable progress bar for speed
                    batch_size=1
                )
            faiss.normalize_L2(query_embedding)
            
            # Search similar chunks (pure local FAISS operation)
            scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
            
            # Return relevant chunks with metadata
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    results.append({
                        'text': self.chunks[idx]['text'],
                        'page': self.chunks[idx]['page'],
                        'score': float(score),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Get relevant context for a query using semantic search - OPTIMIZED"""
        # Fast semantic search (local embeddings only)
        relevant_chunks = self.semantic_search(query, top_k=3)  # Top 3 chunks for balanced accuracy
        
        if not relevant_chunks:
            return ""
        
        # Combine relevant chunks into context efficiently
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            chunk_text = f"[Page {chunk['page']}] {chunk['text']}"
            if current_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'device': self.device,
            'total_chunks': len(self.chunks),
            'index_size': self.vector_index.ntotal if self.vector_index else 0,
            'model_loaded': self.model is not None,
            'ready': self.vector_index is not None and len(self.chunks) > 0
        }

if __name__ == "__main__":
    # Test the RAG processor
    processor = RTX3050RAGProcessor()
    print("🧪 RAG Processor initialized for testing")
    print(f"📊 Stats: {processor.get_stats()}")