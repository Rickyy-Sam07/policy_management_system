#!/usr/bin/env python3
"""
Advanced RTX 3050 Optimized Pipeline
Stage 3: Embedding Search (FAISS/Vector DB)
"""

import os
import time
import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from rtx3050_optimizer import rtx_optimizer

class RTX3050VectorStore:
    """
    Stage 3: Embedding Search with FAISS
    RTX 3050 GPU-accelerated vector database
    """
    
    def __init__(self, use_gpu=True, model_name="all-MiniLM-L6-v2"):
        self.use_gpu = use_gpu and rtx_optimizer.gpu_available
        self.rtx_optimizer = rtx_optimizer if self.use_gpu else None
        
        # Lightweight model optimized for RTX 3050 6GB VRAM
        self.model_name = model_name  # 384 dimensions, fast
        self.embedding_model = None
        self.faiss_index = None
        self.document_chunks = []
        self.chunk_metadata = []
        
        # RTX 3050 optimized settings - INCREASED for better accuracy
        self.max_chunks = 2048  # Increased 4x for better document coverage
        self.batch_size = 32    # Optimal for RTX 3050
        self.embedding_dim = 384  # MiniLM dimension
        
        print(f"🔍 RTX 3050 Vector Store initializing...")
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize sentence transformer with RTX 3050 optimization"""
        
        try:
            # Set device
            device = 'cuda' if self.use_gpu else 'cpu'
            
            print(f"📥 Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(
                self.model_name,
                device=device
            )
            
            # RTX 3050 memory optimization
            if self.use_gpu:
                print(f"🎮 RTX 3050 GPU acceleration active")
                self.embedding_model.max_seq_length = 256  # Reduce for memory
            
            print(f"✅ Embedding model loaded on {device}")
            
        except Exception as e:
            print(f"⚠️ GPU model failed, falling back to CPU: {e}")
            self.use_gpu = False
            self.embedding_model = SentenceTransformer(self.model_name, device='cpu')
    
    def create_vector_index(self, document_data: Dict[str, Any]) -> bool:
        """Create/Update FAISS vector index from document sections with memory optimization"""
        
        start_time = time.time()
        
        try:
            # MEMORY CLEANUP: Clear previous document data
            if hasattr(self, 'document_chunks') and self.document_chunks:
                print(f"🧹 Clearing previous document data for memory optimization...")
                self.document_chunks.clear()
                self.chunk_metadata.clear()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # RTX 3050 memory cleanup
                if self.rtx_optimizer:
                    self.rtx_optimizer.optimize_memory()
            
            sections = document_data.get('sections', [])
            
            if not sections:
                print(f"❌ No document sections to index")
                return False
            
            # Prepare chunks for embedding
            print(f"📝 Preparing {len(sections)} sections for embedding...")
            
            chunks = []
            metadata = []
            
            # Enhanced chunking strategy for better accuracy
            for i, section in enumerate(sections):  # Process ALL sections, not limited
                text = section['text'].strip()
                
                if len(text) < 50:  # Skip very short sections
                    continue
                
                # Smart chunking with overlap for better context preservation
                if len(text) > 800:  # Increased threshold for larger chunks
                    # Split into overlapping chunks for better context
                    chunk_size = 600  # Increased chunk size
                    overlap = 100     # Overlap for context preservation
                    
                    start = 0
                    chunk_num = 0
                    
                    while start < len(text):
                        end = min(start + chunk_size, len(text))
                        
                        # Try to break at sentence boundaries
                        if end < len(text):
                            # Look for sentence endings within last 100 chars
                            sentence_break = -1
                            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                                punct_pos = text.rfind(punct, end-100, end)
                                if punct_pos > sentence_break:
                                    sentence_break = punct_pos + len(punct)
                            
                            if sentence_break > start:
                                end = sentence_break
                        
                        current_chunk = text[start:end].strip()
                        
                        if len(current_chunk) > 50:  # Only meaningful chunks
                            chunks.append(current_chunk)
                            metadata.append({
                                'section_id': i,
                                'page': section['page'],
                                'chunk_type': 'smart_split',
                                'chunk_number': chunk_num,
                                'length': len(current_chunk),
                                'start_pos': start,
                                'end_pos': end
                            })
                            chunk_num += 1
                        
                        # Move start position with overlap
                        start = max(start + chunk_size - overlap, end - overlap)
                        
                        if start >= len(text):
                            break
                            
                elif len(text) > 400:
                    # Medium sections - split at sentences but keep larger chunks
                    sentences = text.split('. ')
                    current_chunk = ""
                    chunk_num = 0
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 500:  # Increased chunk size
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                chunks.append(current_chunk.strip())
                                metadata.append({
                                    'section_id': i,
                                    'page': section['page'],
                                    'chunk_type': 'sentence_split',
                                    'chunk_number': chunk_num,
                                    'length': len(current_chunk)
                                })
                                chunk_num += 1
                            current_chunk = sentence + ". "
                    
                    # Add remaining chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        metadata.append({
                            'section_id': i,
                            'page': section['page'],
                            'chunk_type': 'sentence_split',
                            'chunk_number': chunk_num,
                            'length': len(current_chunk)
                        })
                else:
                    # Keep smaller sections intact
                    chunks.append(text)
                    metadata.append({
                        'section_id': i,
                        'page': section['page'],
                        'chunk_type': 'full_section',
                        'length': len(text)
                    })
            
            # Smart prioritization if we exceed max_chunks
            if len(chunks) > self.max_chunks:
                print(f"⚠️ Prioritizing top {self.max_chunks} chunks for RTX 3050 optimization")
                print(f"📊 Total chunks generated: {len(chunks)}")
                
                # Sort by relevance: prioritize full sections and longer chunks
                chunk_priorities = []
                for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
                    priority = 0
                    
                    # Higher priority for full sections
                    if meta['chunk_type'] == 'full_section':
                        priority += 100
                    elif meta['chunk_type'] == 'smart_split':
                        priority += 80
                    elif meta['chunk_type'] == 'sentence_split':
                        priority += 60
                    
                    # Favor longer chunks (more content)
                    priority += min(meta['length'] / 10, 50)
                    
                    # Slightly favor earlier sections (often more important)
                    priority += max(0, 20 - meta['section_id'] * 0.1)
                    
                    chunk_priorities.append((priority, i, chunk, meta))
                
                # Sort by priority and take top chunks
                chunk_priorities.sort(key=lambda x: x[0], reverse=True)
                chunks = [item[2] for item in chunk_priorities[:self.max_chunks]]
                metadata = [item[3] for item in chunk_priorities[:self.max_chunks]]
                
                print(f"✅ Selected {len(chunks)} highest-priority chunks")
            
            print(f"🔢 Created {len(chunks)} chunks for embedding")
            
            # Generate embeddings in batches (RTX 3050 optimization)
            print(f"🧠 Generating embeddings with RTX 3050...")
            embeddings = self._generate_embeddings_batch(chunks)
            
            if embeddings is None:
                return False
            
            # MEMORY OPTIMIZATION: Reuse existing index or create new one
            print(f"🏗️ Building/Updating FAISS index...")
            
            if self.faiss_index is not None:
                # Clear existing index to reuse memory
                print(f"🔄 Clearing existing index for memory reuse...")
                self.faiss_index.reset()
            else:
                # Create new index only if none exists
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index (replaces previous data)
            self.faiss_index.add(embeddings)
            
            # Store metadata
            self.document_chunks = chunks
            self.chunk_metadata = metadata
            
            # RTX 3050 memory cleanup
            if self.rtx_optimizer:
                self.rtx_optimizer.optimize_memory()
            
            indexing_time = time.time() - start_time
            
            print(f"✅ Vector index created: {len(chunks)} chunks in {indexing_time:.2f}s")
            print(f"🎮 RTX 3050 optimized: {self.use_gpu}")
            
            return True
            
        except Exception as e:
            print(f"❌ Vector indexing error: {e}")
            return False
    
    def _generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings in RTX 3050 optimized batches"""
        
        try:
            all_embeddings = []
            
            # Process in batches for RTX 3050 memory management
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                print(f"🔄 Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
                
                # Generate embeddings
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_tensor=False,
                    show_progress_bar=False,
                    batch_size=self.batch_size
                )
                
                all_embeddings.append(batch_embeddings)
                
                # RTX 3050 memory management
                if self.rtx_optimizer and i % (self.batch_size * 4) == 0:
                    self.rtx_optimizer.optimize_memory()
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings).astype('float32')
            
            print(f"✅ Generated {embeddings.shape[0]} embeddings ({embeddings.shape[1]}D)")
            return embeddings
            
        except Exception as e:
            print(f"❌ Embedding generation error: {e}")
            return None
    
    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content using FAISS vector similarity"""
        
        if not self.faiss_index or not self.document_chunks:
            print(f"❌ Vector index not initialized")
            return []
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_tensor=False,
                show_progress_bar=False
            ).astype('float32')
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Prepare results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.document_chunks):
                    result = {
                        'rank': i + 1,
                        'text': self.document_chunks[idx],
                        'similarity': float(similarity),
                        'metadata': self.chunk_metadata[idx],
                        'chunk_id': int(idx)
                    }
                    results.append(result)
            
            search_time = time.time() - start_time
            
            print(f"🔍 Vector search: {len(results)} results in {search_time:.3f}s")
            return results
            
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []
    
    def save_index(self, filepath: str) -> bool:
        """Save vector index and metadata"""
        
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.meta", 'wb') as f:
                pickle.dump({
                    'chunks': self.document_chunks,
                    'metadata': self.chunk_metadata,
                    'model_name': self.model_name,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            print(f"💾 Vector index saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def load_index(self, filepath: str) -> bool:
        """Load vector index and metadata"""
        
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.meta", 'rb') as f:
                data = pickle.load(f)
                self.document_chunks = data['chunks']
                self.chunk_metadata = data['metadata']
            
            print(f"📂 Vector index loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Load error: {e}")
            return False
    
    def clear_index(self):
        """Clear vector index and free memory"""
        try:
            if self.faiss_index is not None:
                self.faiss_index.reset()
            
            self.document_chunks.clear()
            self.chunk_metadata.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # RTX 3050 memory cleanup
            if self.rtx_optimizer:
                self.rtx_optimizer.optimize_memory()
            
            print(f"🧹 Vector index cleared and memory freed")
            
        except Exception as e:
            print(f"⚠️ Clear index error: {e}")

# Export class
__all__ = ['RTX3050VectorStore']
