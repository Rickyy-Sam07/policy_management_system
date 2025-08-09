#!/usr/bin/env python3
"""
Clean, Efficient RAG System with Parallel Sub-Query Processing
Optimized for RTX 3050 GPU and Complex Insurance Questions
"""

import os
import re
import time
import json
import torch
import faiss
import logging
import tempfile
import requests
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure environment to minimize verbose output
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Disable all warnings
import warnings
warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
import nltk

# Import tqdm first, then monkey patch it
from tqdm import tqdm as original_tqdm
from tqdm.auto import tqdm as auto_tqdm

# Create silent tqdm classes
class SilentTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.n = 0
        self.total = len(iterable) if hasattr(iterable, '__len__') else None
        
    def __iter__(self):
        if self.iterable:
            for item in self.iterable:
                self.n += 1
                yield item
    
    def update(self, n=1):
        self.n += n
    
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass

# Monkey patch tqdm to be silent
import tqdm
tqdm.tqdm = SilentTqdm
tqdm.auto.tqdm = SilentTqdm

# Now import sentence transformers with silenced tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configure environment
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Disable progress bars
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Document chunk with metadata"""
    content: str
    page_num: int
    chunk_id: int
    section_title: str = ""

@dataclass
class RetrievalResult:
    """Search result with scoring"""
    chunk: Chunk
    score: float
    method: str

class CleanRAGSystem:
    """Clean, efficient RAG system with parallel processing"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configuration optimized for insurance documents
        self.config = {
            'chunk_size': 300,        # Smaller chunks for better granularity
            'chunk_overlap': 50,      # Minimal overlap
            'top_k_retrieval': 15,    # More chunks for comprehensive coverage
            'final_k': 8,             # More context for complex questions
            'max_workers': 6,         # Optimal for RTX 3050
            'context_length': 8000,   # Larger context window
            'embedding_batch_size': 64
        }
        
        # Initialize models
        self.chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_model = None
        self.reranker = None
        
        logger.info(f"🚀 Clean RAG System initialized on {self.device.upper()}")
    
    def setup_models(self):
        """Initialize embedding and reranking models"""
        logger.info("📦 Loading models...")
        
        # Load embedding model with GPU support and disabled progress bars
        self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=self.device)
        self.embedding_model.encode(['test'], show_progress_bar=False)  # Initialize without progress bar
        logger.info("✅ BGE embedding model loaded")
        
        # Load reranker with disabled progress
        self.reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
        logger.info("✅ BGE reranker loaded")
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
    
    def download_pdf(self, url: str) -> str:
        """Download PDF from URL to temporary file"""
        logger.info(f"🌐 Downloading PDF from URL...")
        
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            temp_path = tmp_file.name
        
        logger.info(f"✅ PDF downloaded to {temp_path}")
        return temp_path
    
    def extract_text(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page information"""
        logger.info(f"📄 Extracting text from PDF...")
        
        pages = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if text.strip():  # Only include pages with content
                pages.append({
                    'page_num': page_num + 1,
                    'content': text.strip()
                })
        
        doc.close()
        logger.info(f"✅ Extracted {len(pages)} pages")
        return pages
    
    def create_chunks(self, pages: List[Dict]) -> List[Chunk]:
        """Create intelligent chunks from pages"""
        logger.info("✂️ Creating intelligent chunks...")
        
        from nltk.tokenize import sent_tokenize
        
        chunks = []
        chunk_id = 0
        
        for page_data in pages:
            page_num = page_data['page_num']
            content = page_data['content']
            
            # Split into sentences
            try:
                sentences = sent_tokenize(content)
            except:
                sentences = content.split('. ')
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                sentence_length = len(sentence)
                
                # Check if adding this sentence exceeds chunk size
                if current_length + sentence_length > self.config['chunk_size'] and current_chunk:
                    # Create chunk from current sentences
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text.strip()) > 50:  # Only meaningful chunks
                        chunks.append(Chunk(
                            content=chunk_text,
                            page_num=page_num,
                            chunk_id=chunk_id,
                            section_title=self._extract_section_title(chunk_text)
                        ))
                        chunk_id += 1
                    
                    # Start new chunk with overlap
                    if len(current_chunk) >= 2:
                        current_chunk = current_chunk[-1:]  # Keep last sentence for overlap
                        current_length = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.strip()) > 50:
                    chunks.append(Chunk(
                        content=chunk_text,
                        page_num=page_num,
                        chunk_id=chunk_id,
                        section_title=self._extract_section_title(chunk_text)
                    ))
                    chunk_id += 1
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        return chunks
    
    def _extract_section_title(self, text: str) -> str:
        """Extract section title from chunk"""
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and len(line) < 100:
                # Check if it looks like a title/header
                if any(word in line.upper() for word in ['SECTION', 'PART', 'CHAPTER', 'COVERAGE', 'BENEFIT', 'EXCLUSION', 'CLAIM']):
                    return line[:50]
        return ""
    
    def generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """Generate embeddings with GPU acceleration"""
        logger.info("🧠 Generating embeddings with GPU acceleration...")
        
        if not self.embedding_model:
            self.setup_models()
        
        texts = [chunk.content for chunk in chunks]
        batch_size = self.config['embedding_batch_size']
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    device=self.device,
                    normalize_embeddings=True
                )
            
            all_embeddings.append(batch_embeddings)
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        embeddings = np.vstack(all_embeddings)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search"""
        logger.info("📊 Building FAISS index...")
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings.astype('float32'))
        
        logger.info(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def process_document(self, pdf_path_or_url: str) -> bool:
        """Process document from file path or URL"""
        try:
            logger.info("🚀 Starting document processing...")
            
            # Handle URL downloads
            if pdf_path_or_url.startswith('http'):
                pdf_path = self.download_pdf(pdf_path_or_url)
                cleanup_temp = True
            else:
                pdf_path = pdf_path_or_url
                cleanup_temp = False
            
            # Extract text
            pages = self.extract_text(pdf_path)
            
            # Create chunks
            self.chunks = self.create_chunks(pages)
            
            # Generate embeddings
            self.embeddings = self.generate_embeddings(self.chunks)
            
            # Build index
            self.build_index(self.embeddings)
            
            # Cleanup temporary file
            if cleanup_temp:
                try:
                    os.unlink(pdf_path)
                except:
                    pass
            
            logger.info("✅ Document processing completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return False
    
    def retrieve_chunks(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant chunks using hybrid search"""
        if not self.chunks or self.faiss_index is None:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=False
        )
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            min(self.config['top_k_retrieval'], len(self.chunks))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append(RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    method="semantic"
                ))
        
        # Rerank results
        if len(results) > self.config['final_k'] and self.reranker:
            pairs = [(query, result.chunk.content) for result in results]
            rerank_scores = self.reranker.predict(pairs)
            
            # Update scores and re-sort
            for i, new_score in enumerate(rerank_scores):
                results[i].score = float(new_score)
                results[i].method = "reranked"
            
            results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:self.config['final_k']]
    
    def get_context(self, query: str) -> str:
        """Get formatted context for query"""
        results = self.retrieve_chunks(query)
        
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in results:
            chunk = result.chunk
            section_prefix = f"[{chunk.section_title}] " if chunk.section_title else ""
            text = f"[Page {chunk.page_num}] {section_prefix}{chunk.content}"
            
            if current_length + len(text) > self.config['context_length']:
                break
            
            context_parts.append(text)
            current_length += len(text)
        
        return "\n\n".join(context_parts)
    
    def is_complex_query(self, query: str) -> bool:
        """Detect if query is complex and needs decomposition"""
        query_lower = query.lower()
        
        # Check for multiple coordination
        multi_indicators = ['while', 'also', 'simultaneously', 'at the same time', 'additionally']
        if any(indicator in query_lower for indicator in multi_indicators):
            return True
        
        # Check for multiple questions
        if query.count('?') > 1:
            return True
        
        # Check for multiple action verbs
        action_verbs = ['check', 'confirm', 'provide', 'list', 'describe', 'submit', 'update', 'inquire']
        verb_count = sum(1 for verb in action_verbs if verb in query_lower)
        if verb_count >= 3:
            return True
        
        # Check for long multi-concept queries
        if len(query.split()) > 20 and any(word in query_lower for word in [' and ', ', ', 'also']):
            return True
        
        return False
    
    def decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-queries"""
        if not self.is_complex_query(query):
            return [query]
        
        sub_queries = []
        
        # Split on coordination words
        multi_indicators = ['while', 'also', 'simultaneously', 'at the same time', 'additionally']
        for indicator in multi_indicators:
            if indicator in query.lower():
                parts = query.split(indicator, 1)
                if len(parts) == 2:
                    # Clean up parts
                    first_part = parts[0].strip().rstrip(',').rstrip('?') + '?'
                    second_part = parts[1].strip()
                    
                    # Fix second part if needed
                    if not second_part.endswith('?'):
                        second_part += '?'
                    if not second_part[0].isupper():
                        second_part = second_part.capitalize()
                    
                    if len(first_part) > 15 and len(second_part) > 15:
                        sub_queries = [first_part, second_part]
                        break
        
        # Fallback to sentence splitting
        if not sub_queries and ',' in query:
            parts = [p.strip() for p in query.split(',')]
            if len(parts) >= 2:
                current_query = ""
                for part in parts:
                    if any(word in part.lower() for word in ['and', 'also']):
                        if current_query:
                            sub_queries.append(current_query.rstrip('?') + '?')
                        current_query = part
                    else:
                        current_query += (', ' if current_query else '') + part
                
                if current_query:
                    sub_queries.append(current_query.rstrip('?') + '?')
        
        # Clean up results
        cleaned_queries = []
        for sq in sub_queries:
            sq = sq.strip()
            if sq and len(sq) > 10 and ('?' in sq or len(sq) > 20):
                if not sq.endswith('?'):
                    sq += '?'
                cleaned_queries.append(sq)
        
        return cleaned_queries[:4] if cleaned_queries else [query]
    
    def call_gemini_api(self, prompt: str, max_tokens: int = 100) -> str:
        """Call Gemini API with retry logic"""
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1
            }
        }
        
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content.strip()
                else:
                    return "No answer found"
                    
            except Exception as e:
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    continue
                logger.error(f"Gemini API error: {e}")
                return "No answer found"
    
    def answer_question(self, question: str, context: str) -> str:
        """Generate answer using Gemini with context"""
        if not context or len(context.strip()) < 50:
            return "No answer found"
        
        prompt = f"""Based on the insurance policy context provided, answer the question directly and concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.call_gemini_api(prompt, max_tokens=150)
    
    def process_single_question(self, question: str) -> str:
        """Process a single question with parallel sub-query handling"""
        try:
            if not self.chunks:
                return "No document loaded"
            
            # Check if complex query
            if self.is_complex_query(question):
                logger.info(f"🧠 Complex query detected: '{question[:50]}...'")
                
                # Decompose into sub-queries
                sub_queries = self.decompose_query(question)
                logger.info(f"🔍 Decomposed into {len(sub_queries)} sub-queries")
                
                if len(sub_queries) > 1:
                    # Process sub-queries in parallel
                    max_workers = min(len(sub_queries), self.config['max_workers'])
                    
                    def process_sub_query(sub_query: str) -> Tuple[str, str]:
                        context = self.get_context(sub_query)
                        if context:
                            answer = self.answer_question(sub_query, context)
                            return sub_query, answer
                        return sub_query, "No answer found"
                    
                    sub_answers = {}
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_query = {
                            executor.submit(process_sub_query, sq): sq 
                            for sq in sub_queries
                        }
                        
                        for future in as_completed(future_to_query):
                            try:
                                sub_query, answer = future.result()
                                sub_answers[sub_query] = answer
                            except Exception as e:
                                logger.error(f"Sub-query processing error: {e}")
                    
                    # Synthesize final answer
                    valid_answers = {q: a for q, a in sub_answers.items() if a and "No answer found" not in a}
                    
                    if valid_answers:
                        # Create comprehensive answer
                        answer_parts = []
                        for i, (sub_q, answer) in enumerate(valid_answers.items(), 1):
                            answer_parts.append(f"{i}. {answer}")
                        
                        return "\n\n".join(answer_parts)
                    else:
                        return "No answer found after comprehensive analysis"
            
            # Simple query processing
            context = self.get_context(question)
            return self.answer_question(question, context)
            
        except Exception as e:
            logger.error(f"Question processing error: {e}")
            return "No answer found"
    
    def process_questions(self, questions: List[str]) -> List[str]:
        """Process multiple questions with progress tracking"""
        logger.info(f"🚀 Processing {len(questions)} questions...")
        
        answers = []
        for i, question in enumerate(questions):
            answer = self.process_single_question(question)
            answers.append(answer)
            
            if (i + 1) % 5 == 0:
                logger.info(f"📊 Progress: {i + 1}/{len(questions)} completed")
        
        logger.info("✅ All questions processed")
        return answers


# Factory function for easy usage
def create_rag_system(api_key: str) -> CleanRAGSystem:
    """Create and return a clean RAG system instance"""
    return CleanRAGSystem(api_key)


if __name__ == "__main__":
    # Example usage
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    rag = create_rag_system(api_key)
    
    # Example document processing
    # success = rag.process_document("path/to/document.pdf")
    # if success:
    #     answers = rag.process_questions(["What is the claim process?"])
    #     print(answers[0])
