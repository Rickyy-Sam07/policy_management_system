#!/usr/bin/env python3
"""
Advanced RTX 3050 Optimized Pipeline
Complete 5-Stage Implementation
Input → LLM Parser → Embedding Search → Clause Matching → Logic Evaluation → JSON Output
"""

import os
import time
import json
from typing import Dict, List, Any, Optional

# Import all 5 stages
from advanced_processor import AdvancedDocumentProcessor
from rtx3050_vector_store import RTX3050VectorStore
from rtx3050_clause_matcher import RTX3050ClauseMatcher
from rtx3050_logic_evaluator import RTX3050LogicEvaluator
from rtx3050_optimizer import rtx_optimizer

class RTX3050AdvancedPipeline:
    """
    Complete 5-Stage Advanced Pipeline
    Optimized for RTX 3050 GPU acceleration
    Target: 2-3 seconds, 90%+ accuracy
    """
    
    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        
        # Initialize all 5 stages
        print(f"🚀 Initializing RTX 3050 Advanced Pipeline...")
        
        # Stage 1 & 2: Document Input + LLM Parser
        self.document_processor = AdvancedDocumentProcessor(groq_api_key)
        
        # Stage 3: Vector Database
        self.vector_store = RTX3050VectorStore(use_gpu=True)
        
        # Stage 4: Clause Matching
        self.clause_matcher = RTX3050ClauseMatcher()
        
        # Stage 5: Logic Evaluation
        self.logic_evaluator = RTX3050LogicEvaluator(groq_api_key)
        
        # Pipeline state
        self.documents_loaded = False
        self.vector_index_ready = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'gpu_acceleration': rtx_optimizer.gpu_available if rtx_optimizer else False
        }
        
        print(f"✅ RTX 3050 Advanced Pipeline initialized")
        print(f"🎮 GPU Acceleration: {self.performance_metrics['gpu_acceleration']}")
    
    def load_documents(self, pdf_paths: List[str]) -> bool:
        """Stage 1: Load and process documents"""
        
        start_time = time.time()
        
        try:
            print(f"📄 Stage 1: Loading {len(pdf_paths)} documents...")
            
            # Process documents with RTX 3050 optimization
            document_data = self.document_processor.process_documents(pdf_paths)
            
            if not document_data or not document_data.get('sections'):
                print(f"❌ Failed to load documents")
                return False
            
            # Store document data
            self.document_data = document_data
            self.documents_loaded = True
            
            loading_time = time.time() - start_time
            print(f"✅ Stage 1 complete: {len(document_data['sections'])} sections in {loading_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Document loading error: {e}")
            return False
    
    def build_vector_index(self) -> bool:
        """Stage 3: Build vector database index"""
        
        if not self.documents_loaded:
            print(f"❌ Documents must be loaded first")
            return False
        
        start_time = time.time()
        
        try:
            print(f"🔍 Stage 3: Building vector index...")
            
            # Create FAISS vector index with RTX 3050 optimization
            success = self.vector_store.create_vector_index(self.document_data)
            
            if not success:
                print(f"❌ Failed to build vector index")
                return False
            
            self.vector_index_ready = True
            
            indexing_time = time.time() - start_time
            print(f"✅ Stage 3 complete: Vector index built in {indexing_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Vector indexing error: {e}")
            return False
    
    async def get_optimized_answer(self, query: str) -> Dict[str, Any]:
        """🚀 OPTIMIZED query processing with async support and performance enhancements"""
        
        if not self.vector_index_ready:
            return self._generate_error_response("Pipeline not ready. Load documents and build index first.")
        
        start_time = time.time()
        
        try:
            # Stage 2: LLM Parser (optimized with reduced tokens)
            parsed_query = self.document_processor.parse_query_fast(query)
            
            if not parsed_query:
                return self._generate_error_response("Failed to parse query")
            
            stage2_time = time.time()
            
            # Stage 3: Vector Search (same performance)
            vector_results = self.vector_store.search_similar(query, top_k=6)  # Reduced from 8 to 6
            
            if not vector_results:
                return self._generate_no_results_response(parsed_query)
            
            stage3_time = time.time()
            
            # Stage 4: Clause Matching (optimized)
            matched_clauses = self.clause_matcher.match_clauses_fast(vector_results, parsed_query)
            
            stage4_time = time.time()
            
            # Stage 5: Logic Evaluation (optimized with reduced tokens)
            final_response = self.logic_evaluator.evaluate_fast(parsed_query, matched_clauses)
            
            stage5_time = time.time()
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Return with performance info
            return {
                'answer': final_response.get('answer', 'Unable to determine'),
                'confidence': final_response.get('confidence', 0.7),
                'source': final_response.get('source', ''),
                'clause_info': final_response.get('clause_info', {}),
                'performance': {
                    'total_time': round(total_time, 3),
                    'optimized': True
                }
            }
            
        except Exception as e:
            print(f"❌ Optimized pipeline error: {e}")
            return self._generate_error_response(str(e))

    def process_query(self, query: str) -> Dict[str, Any]:
        """Complete 5-stage query processing"""
        
        if not self.vector_index_ready:
            return self._generate_error_response("Pipeline not ready. Load documents and build index first.")
        
        start_time = time.time()
        
        try:
            print(f"🔥 Processing query: '{query[:50]}...'")
            
            # Stage 2: LLM Parser
            print(f"🧠 Stage 2: Parsing query...")
            parsed_query = self.document_processor.parse_query(query)
            
            if not parsed_query:
                return self._generate_error_response("Failed to parse query")
            
            stage2_time = time.time()
            print(f"✅ Stage 2: Query parsed in {stage2_time - start_time:.3f}s")
            
            # Stage 3: Vector Search
            print(f"🔍 Stage 3: Vector search...")
            vector_results = self.vector_store.search_similar(query, top_k=8)
            
            if not vector_results:
                return self._generate_no_results_response(parsed_query)
            
            stage3_time = time.time()
            print(f"✅ Stage 3: Found {len(vector_results)} results in {stage3_time - stage2_time:.3f}s")
            
            # Stage 4: Clause Matching
            print(f"🎯 Stage 4: Clause matching...")
            matched_clauses = self.clause_matcher.match_clauses(vector_results, parsed_query)
            
            # Enhance clauses with context
            matched_clauses = self.clause_matcher.enhance_clause_context(matched_clauses)
            
            stage4_time = time.time()
            print(f"✅ Stage 4: Matched {len(matched_clauses)} clauses in {stage4_time - stage3_time:.3f}s")
            
            # Stage 5: Logic Evaluation
            print(f"🧠 Stage 5: Logic evaluation...")
            final_response = self.logic_evaluator.evaluate_and_generate_response(
                parsed_query, matched_clauses
            )
            
            stage5_time = time.time()
            print(f"✅ Stage 5: Logic evaluation in {stage5_time - stage4_time:.3f}s")
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(total_time)
            
            # Add pipeline timing to response
            final_response['performance'] = {
                'total_time': round(total_time, 3),
                'stage_times': {
                    'query_parsing': round(stage2_time - start_time, 3),
                    'vector_search': round(stage3_time - stage2_time, 3),
                    'clause_matching': round(stage4_time - stage3_time, 3),
                    'logic_evaluation': round(stage5_time - stage4_time, 3)
                },
                'gpu_accelerated': self.performance_metrics['gpu_acceleration'],
                'query_count': self.performance_metrics['total_queries']
            }
            
            print(f"🚀 Pipeline complete: {total_time:.3f}s (Target: 2-3s)")
            
            return final_response
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return self._generate_error_response(str(e))
    
    def process_batch_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries with RTX 3050 batch optimization"""
        
        if not self.vector_index_ready:
            return [self._generate_error_response("Pipeline not ready") for _ in queries]
        
        start_time = time.time()
        results = []
        
        print(f"📊 Processing batch of {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            print(f"🔄 Query {i+1}/{len(queries)}")
            result = self.process_query(query)
            results.append(result)
            
            # RTX 3050 memory management between queries
            if rtx_optimizer and i % 3 == 0:  # Every 3 queries
                rtx_optimizer.optimize_memory()
        
        batch_time = time.time() - start_time
        avg_time = batch_time / len(queries)
        
        print(f"✅ Batch complete: {len(queries)} queries in {batch_time:.2f}s (avg: {avg_time:.2f}s)")
        
        return results
    
    def save_vector_index(self, filepath: str) -> bool:
        """Save vector index for faster startup"""
        
        if not self.vector_index_ready:
            print(f"❌ No vector index to save")
            return False
        
        try:
            success = self.vector_store.save_index(filepath)
            if success:
                print(f"💾 Vector index saved to {filepath}")
            return success
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def load_vector_index(self, filepath: str) -> bool:
        """Load pre-built vector index"""
        
        try:
            success = self.vector_store.load_index(filepath)
            if success:
                self.vector_index_ready = True
                print(f"📂 Vector index loaded from {filepath}")
            return success
            
        except Exception as e:
            print(f"❌ Load error: {e}")
            return False
    
    def _update_performance_metrics(self, query_time: float):
        """Update pipeline performance tracking"""
        
        self.performance_metrics['total_queries'] += 1
        
        # Calculate running average
        total = self.performance_metrics['total_queries']
        current_avg = self.performance_metrics['avg_response_time']
        
        new_avg = ((current_avg * (total - 1)) + query_time) / total
        self.performance_metrics['avg_response_time'] = new_avg
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'target_metrics': {
                'target_time': '2-3 seconds',
                'target_accuracy': '90%+',
                'gpu_optimization': 'RTX 3050 6GB'
            },
            'pipeline_status': {
                'documents_loaded': self.documents_loaded,
                'vector_index_ready': self.vector_index_ready,
                'gpu_available': rtx_optimizer.gpu_available if rtx_optimizer else False
            }
        }
    
    def _generate_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate standardized error response"""
        
        return {
            "error": True,
            "message": error_msg,
            "pipeline_status": "error",
            "timestamp": time.time()
        }
    
    def _generate_no_results_response(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response for no results found"""
        
        return self.logic_evaluator._generate_no_results_response(parsed_query)

# Export main class
__all__ = ['RTX3050AdvancedPipeline']

# Demo usage
if __name__ == "__main__":
    # Example usage
    import os
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("❌ GROQ_API_KEY environment variable not set")
        exit(1)
    
    # Initialize pipeline
    pipeline = RTX3050AdvancedPipeline(groq_api_key)
    
    # Load documents
    pdf_files = ["doc1.pdf", "doc2.pdf"]  # Replace with actual paths
    if pipeline.load_documents(pdf_files):
        # Build vector index
        if pipeline.build_vector_index():
            # Process test query
            test_query = "What is covered under this insurance policy?"
            result = pipeline.process_query(test_query)
            
            print(f"\n🎯 Final Result:")
            print(json.dumps(result, indent=2))
            
            # Show performance stats
            stats = pipeline.get_performance_stats()
            print(f"\n📊 Performance Stats:")
            print(json.dumps(stats, indent=2))
