#!/usr/bin/env python3
"""
RTX 3050 Streaming RAG Pipeline
Memory-efficient streaming processing with parallel extraction and vectorization
"""

import os
import time
import queue
import threading
import numpy as np
import fitz  # PyMuPDF
import pdfplumber
import faiss
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures
from typing import List, Dict, Tuple, Optional
import re
import gc

class RTX3050StreamingRAG:
    def __init__(self):
        """Initialize streaming RAG processor with memory optimization"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.vector_index = None
        self.chunks = []
        
        # Streaming queues
        self.extraction_queue = queue.Queue(maxsize=50)  # Limit memory
        self.vectorization_queue = queue.Queue(maxsize=20)
        self.completed_chunks = []
        
        print(f"🚀 RTX 3050 Streaming RAG initializing on {self.device.upper()}")
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print(f"✅ Embedding model loaded on {self.device.upper()}")
        except Exception as e:
            print(f"⚠️ GPU model failed, falling back to CPU: {e}")
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.device = 'cpu'
    
    def detect_tables_fast(self, page_content: bytes, page_num: int) -> bool:
        """Fast table detection using simple heuristics"""
        try:
            # Quick text-based detection
            doc = fitz.open(stream=page_content, filetype="pdf")
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            # Simple heuristics for table detection
            lines = text.split('\n')
            table_indicators = 0
            
            for line in lines:
                # Check for table-like patterns
                if len(line.split()) > 3 and any(char.isdigit() for char in line):
                    table_indicators += 1
                if '|' in line or '\t' in line:
                    table_indicators += 1
            
            # If >20% of lines look like table data
            return table_indicators > len(lines) * 0.2
            
        except:
            return False
    
    def extract_page_hybrid(self, pdf_content: bytes, page_num: int) -> Dict:
        """Hybrid extraction: PyMuPDF for text, pdfplumber for tables"""
        try:
            # Fast table detection
            has_tables = self.detect_tables_fast(pdf_content, page_num)
            
            if has_tables:
                # Use pdfplumber for table-rich pages
                with pdfplumber.open(pdf_content) as pdf:
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        text = page.extract_text() or ""
                        tables = page.extract_tables()
                        
                        # Convert tables to text
                        table_text = ""
                        for table in tables:
                            for row in table:
                                if row:
                                    table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                        
                        combined_text = text + "\n" + table_text
                        return {
                            'page': page_num,
                            'text': combined_text.strip(),
                            'has_tables': True,
                            'extraction_method': 'pdfplumber'
                        }
            else:
                # Use PyMuPDF for text-only pages (faster)
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                if page_num < len(doc):
                    page = doc[page_num]
                    text = page.get_text()
                    doc.close()
                    return {
                        'page': page_num,
                        'text': text.strip(),
                        'has_tables': False,
                        'extraction_method': 'pymupdf'
                    }
                doc.close()
            
            return {'page': page_num, 'text': '', 'has_tables': False, 'extraction_method': 'failed'}
            
        except Exception as e:
            print(f"❌ Page {page_num} extraction error: {e}")
            return {'page': page_num, 'text': '', 'has_tables': False, 'extraction_method': 'failed'}
    
    def extraction_worker(self, pdf_content: bytes, page_numbers: List[int]):
        """Worker thread for page extraction"""
        for page_num in page_numbers:
            try:
                page_data = self.extract_page_hybrid(pdf_content, page_num)
                if page_data['text']:
                    self.extraction_queue.put(page_data)
                    print(f"📄 Extracted page {page_num+1}: {len(page_data['text'])} chars ({page_data['extraction_method']})")
            except Exception as e:
                print(f"❌ Extraction worker error on page {page_num}: {e}")
        
        # Signal completion
        self.extraction_queue.put(None)
    
    def chunking_worker(self):
        """Worker thread for text chunking and queuing for vectorization"""
        while True:
            try:
                page_data = self.extraction_queue.get(timeout=30)
                if page_data is None:  # Completion signal
                    self.vectorization_queue.put(None)
                    break
                
                # Chunk the page text
                chunks = self.chunk_page_text(page_data['text'], page_data['page'])
                
                for chunk in chunks:
                    self.vectorization_queue.put(chunk)
                
                # Clear page data from memory immediately
                del page_data
                gc.collect()
                
            except queue.Empty:
                print("⚠️ Chunking worker timeout")
                break
            except Exception as e:
                print(f"❌ Chunking worker error: {e}")
                break
    
    def vectorization_worker(self):
        """Worker thread for embedding generation and vector index building"""
        batch_texts = []
        batch_metadata = []
        batch_size = 32
        
        while True:
            try:
                chunk_data = self.vectorization_queue.get(timeout=30)
                if chunk_data is None:  # Completion signal
                    # Process final batch
                    if batch_texts:
                        self.process_embedding_batch(batch_texts, batch_metadata)
                    break
                
                batch_texts.append(chunk_data['text'])
                batch_metadata.append({
                    'page': chunk_data['page'],
                    'chunk_id': len(self.completed_chunks)
                })
                
                # Process batch when full
                if len(batch_texts) >= batch_size:
                    self.process_embedding_batch(batch_texts, batch_metadata)
                    batch_texts = []
                    batch_metadata = []
                
            except queue.Empty:
                print("⚠️ Vectorization worker timeout")
                break
            except Exception as e:
                print(f"❌ Vectorization worker error: {e}")
                break
    
    def process_embedding_batch(self, texts: List[str], metadata: List[Dict]):
        """Process a batch of texts into embeddings and add to index"""
        try:
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=len(texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device
                )
            
            # Initialize or update vector index
            if self.vector_index is None:
                dimension = embeddings.shape[1]
                self.vector_index = faiss.IndexFlatIP(dimension)
            
            # Normalize and add to index
            faiss.normalize_L2(embeddings)
            self.vector_index.add(embeddings.astype('float32'))
            
            # Store chunk data
            for i, (text, meta) in enumerate(zip(texts, metadata)):
                self.completed_chunks.append({
                    'text': text,
                    'page': meta['page']
                })
            
            print(f"🧠 Vectorized batch: {len(texts)} chunks → {self.vector_index.ntotal} total vectors")
            
            # Clear batch data from memory
            del embeddings
            gc.collect()
            
        except Exception as e:
            print(f"❌ Embedding batch error: {e}")
    
    def chunk_page_text(self, text: str, page_num: int, chunk_size: int = 200) -> List[Dict]:
        """Chunk page text into smaller segments"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_tokens = len(sentence) // 4  # Rough token estimation
            
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'page': page_num
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
                'page': page_num
            })
        
        return chunks
    
    def process_document_streaming(self, pdf_content: bytes) -> bool:
        """Main streaming processing pipeline"""
        print("🚀 Starting streaming RAG pipeline...")
        start_time = time.time()
        
        try:
            # Get total pages
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            total_pages = len(doc)
            doc.close()
            
            print(f"📚 Processing {total_pages} pages with streaming pipeline")
            
            # Divide pages among extraction workers
            pages_per_worker = max(1, total_pages // 4)
            page_batches = [
                list(range(i, min(i + pages_per_worker, total_pages)))
                for i in range(0, total_pages, pages_per_worker)
            ]
            
            # Start worker threads
            threads = []
            
            # Start extraction workers
            for i, page_batch in enumerate(page_batches):
                if page_batch:  # Only start if batch has pages
                    thread = threading.Thread(
                        target=self.extraction_worker,
                        args=(pdf_content, page_batch),
                        name=f"Extractor-{i}"
                    )
                    thread.start()
                    threads.append(thread)
            
            # Start chunking worker
            chunking_thread = threading.Thread(target=self.chunking_worker, name="Chunker")
            chunking_thread.start()
            threads.append(chunking_thread)
            
            # Start vectorization worker
            vectorization_thread = threading.Thread(target=self.vectorization_worker, name="Vectorizer")
            vectorization_thread.start()
            threads.append(vectorization_thread)
            
            # Wait for all workers to complete
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            print(f"🎉 Streaming pipeline completed in {total_time:.2f}s")
            print(f"📊 Stats: {len(self.completed_chunks)} chunks, {self.vector_index.ntotal if self.vector_index else 0} vectors")
            
            return self.vector_index is not None and len(self.completed_chunks) > 0
            
        except Exception as e:
            print(f"❌ Streaming pipeline error: {e}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform semantic search using vector similarity"""
        if not self.vector_index or not self.model:
            return []
        
        try:
            # Generate query embedding
            with torch.no_grad():
                query_embedding = self.model.encode([query], convert_to_numpy=True, device=self.device)
            faiss.normalize_L2(query_embedding)
            
            # Search similar chunks
            scores, indices = self.vector_index.search(query_embedding.astype('float32'), top_k)
            
            # Return relevant chunks
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.completed_chunks):
                    results.append({
                        'text': self.completed_chunks[idx]['text'],
                        'page': self.completed_chunks[idx]['page'],
                        'score': float(score),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Get relevant context for a query using semantic search"""
        relevant_chunks = self.semantic_search(query, top_k=3)
        
        if not relevant_chunks:
            return ""
        
        # Combine relevant chunks into context
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            chunk_text = f"[Page {chunk['page']+1}] {chunk['text']}"
            if current_length + len(chunk_text) > max_context_length:
                break
            context_parts.append(chunk_text)
            current_length += len(chunk_text)
        
        return "\n\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'device': self.device,
            'total_chunks': len(self.completed_chunks),
            'index_size': self.vector_index.ntotal if self.vector_index else 0,
            'model_loaded': self.model is not None,
            'ready': self.vector_index is not None and len(self.completed_chunks) > 0
        }

if __name__ == "__main__":
    # Test the streaming RAG processor
    processor = RTX3050StreamingRAG()
    print("🧪 Streaming RAG Processor initialized for testing")
    print(f"📊 Stats: {processor.get_stats()}")