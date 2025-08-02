#!/usr/bin/env python3
"""
Advanced RTX 3050 Optimized Pipeline
Stage 5: Logic Evaluation & JSON Output
"""

import time
import json
from typing import Dict, List, Any, Optional
from groq import Groq
from rtx3050_optimizer import rtx_optimizer

class RTX3050LogicEvaluator:
    """
    Stage 5: Logic Evaluation & Structured JSON Output
    Final reasoning and decision making
    """
    
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        self.rtx_optimizer = rtx_optimizer if rtx_optimizer.gpu_available else None
        
        # Logic evaluation prompts
        self.evaluation_prompts = {
            'coverage': """
            Query: {query}
            Policy Text: {clauses}
            
            Answer briefly: Is this covered? YES/NO/PARTIAL + 1-2 key details + page reference.
            JSON format: {{"answer": "...", "analysis": "...", "confidence": 0.8}}
            """,
            
            'claim': """
            Query: {query}
            Policy Text: {clauses}
            
            Answer briefly: Claim process + requirements + timeline.
            JSON format: {{"answer": "...", "analysis": "...", "confidence": 0.8}}
            """,
            
            'general': """
            Query: {query}
            Policy Text: {clauses}
            
            Answer briefly with key facts + page reference.
            JSON format: {{"answer": "...", "analysis": "...", "confidence": 0.8}}
            """
        }
        
        print(f"🧠 RTX 3050 Logic Evaluator initialized")
    
    def evaluate_and_generate_response(
        self, 
        parsed_query: Dict[str, Any], 
        matched_clauses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Main evaluation and JSON response generation"""
        
        start_time = time.time()
        
        try:
            if not matched_clauses:
                return self._generate_no_results_response(parsed_query)
            
            print(f"🧠 Evaluating {len(matched_clauses)} clauses...")
            
            # Determine evaluation type
            evaluation_type = self._determine_evaluation_type(parsed_query)
            
            # Prepare clause context
            clause_context = self._prepare_clause_context(matched_clauses)
            
            # Generate logic evaluation
            logic_response = self._generate_logic_evaluation(
                parsed_query, clause_context, evaluation_type
            )
            
            # Structure final response
            final_response = self._structure_final_response(
                parsed_query, matched_clauses, logic_response
            )
            
            # RTX 3050 memory optimization
            if self.rtx_optimizer:
                self.rtx_optimizer.optimize_memory()
            
            evaluation_time = time.time() - start_time
            
            print(f"✅ Logic evaluation completed in {evaluation_time:.3f}s")
            
            return final_response
            
        except Exception as e:
            print(f"❌ Logic evaluation error: {e}")
            return self._generate_error_response(parsed_query, str(e))
    
    def evaluate_fast(
        self, 
        parsed_query: Dict[str, Any], 
        matched_clauses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """🚀 OPTIMIZED evaluation with reduced LLM tokens and processing"""
        
        try:
            if not matched_clauses:
                return {
                    'answer': 'No relevant information found',
                    'confidence': 0.3,
                    'source': 'No matching clauses'
                }
            
            # Take only the best clause for speed
            best_clause = matched_clauses[0]
            clause_text = best_clause['text'][:300]  # Reduced from 400
            
            # Simplified prompt with reduced tokens
            query_text = parsed_query.get('original_query', '')
            
            prompt = f"Q: {query_text}\nPolicy text: {clause_text}\nAnswer briefly (max 50 words):"
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=100,  # Reduced from typical 500+
                timeout=10  # Faster timeout
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                'answer': answer,
                'confidence': 0.8,
                'source': f"Page {best_clause.get('metadata', {}).get('page', 'N/A')}",
                'clause_info': best_clause.get('metadata', {})
            }
            
        except Exception as e:
            return {
                'answer': 'Processing error occurred',
                'confidence': 0.2,
                'source': f'Error: {str(e)}'
            }
    
    def _determine_evaluation_type(self, parsed_query: Dict[str, Any]) -> str:
        """Determine the type of logic evaluation needed"""
        
        intent = parsed_query.get('intent', '').lower()
        answer_type = parsed_query.get('answer_type', '').lower()
        
        if any(word in intent for word in ['cover', 'benefit', 'include']):
            return 'coverage'
        elif any(word in intent for word in ['claim', 'reimburse', 'pay']):
            return 'claim'
        else:
            return 'general'
    
    def _prepare_clause_context(self, matched_clauses: List[Dict[str, Any]]) -> str:
        """Prepare clause context for LLM evaluation"""
        
        context_parts = []
        
        for i, clause in enumerate(matched_clauses[:3]):  # Top 3 clauses only for speed
            confidence = clause.get('confidence', 0)
            text = clause['text'][:400]  # Limit text to 400 chars
            metadata = clause.get('metadata', {})
            
            context_part = f"Clause {i+1} (Page {metadata.get('page', 'N/A')}, Score: {confidence:.2f}):\n{text}..."
            
            context_parts.append(context_part.strip())
        
        return "\n\n".join(context_parts)
    
    def _generate_logic_evaluation(
        self, 
        parsed_query: Dict[str, Any], 
        clause_context: str, 
        evaluation_type: str
    ) -> Dict[str, Any]:
        """Generate LLM-based logic evaluation"""
        
        try:
            query_text = parsed_query.get('original_query', '')
            prompt_template = self.evaluation_prompts.get(evaluation_type, self.evaluation_prompts['general'])
            
            prompt = prompt_template.format(
                query=query_text,
                clauses=clause_context
            )
            
            print(f"🤖 Generating logic evaluation with Groq...")
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert insurance policy analyzer. Provide accurate, concise analysis in JSON format with 'analysis', 'answer', and 'confidence' fields. Maximum 3 sentences."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0,  # More deterministic
                max_tokens=300,   # Reduced from 1500
                timeout=5.0       # Add 5 second timeout
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                logic_result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON if wrapped in text
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    logic_result = json.loads(json_match.group())
                else:
                    # Create structured fallback
                    logic_result = {
                        "analysis": response_text,
                        "confidence": 85,
                        "structured": False
                    }
            
            return logic_result
            
        except Exception as e:
            print(f"⚠️ LLM evaluation error: {e}")
            return {
                "analysis": "Unable to perform detailed analysis due to processing error",
                "confidence": 50,
                "error": str(e)
            }
    
    def _structure_final_response(
        self, 
        parsed_query: Dict[str, Any], 
        matched_clauses: List[Dict[str, Any]], 
        logic_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Structure the final JSON response"""
        
        # Extract key metrics
        top_clause = matched_clauses[0] if matched_clauses else None
        avg_confidence = sum(c.get('confidence', 0) for c in matched_clauses) / len(matched_clauses) if matched_clauses else 0
        
        # Build structured response
        response = {
            "query_analysis": {
                "original_query": parsed_query.get('original_query', ''),
                "intent": parsed_query.get('intent', ''),
                "entities": parsed_query.get('entities', []),
                "answer_type": parsed_query.get('answer_type', ''),
                "processing_time": time.time()
            },
            
            "search_results": {
                "total_clauses_found": len(matched_clauses),
                "average_confidence": round(avg_confidence, 3),
                "search_quality": "high" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.4 else "low"
            },
            
            "logic_evaluation": logic_response,
            
            "evidence": {
                "primary_clause": {
                    "text": top_clause['text'] if top_clause else "",
                    "confidence": top_clause.get('confidence', 0) if top_clause else 0,
                    "page": top_clause.get('metadata', {}).get('page', 'N/A') if top_clause else 'N/A'
                },
                "supporting_clauses": [
                    {
                        "text": clause['text'][:200] + "..." if len(clause['text']) > 200 else clause['text'],
                        "confidence": clause.get('confidence', 0),
                        "page": clause.get('metadata', {}).get('page', 'N/A')
                    }
                    for clause in matched_clauses[1:4]  # Next 3 clauses
                ]
            },
            
            "metadata": {
                "pipeline_version": "RTX3050_Advanced_v1.0",
                "gpu_accelerated": self.rtx_optimizer.gpu_available if self.rtx_optimizer else False,
                "confidence_threshold": 0.3,
                "timestamp": time.time()
            }
        }
        
        # Add evaluation-specific fields
        evaluation_type = self._determine_evaluation_type(parsed_query)
        if evaluation_type == 'coverage':
            response["coverage_analysis"] = {
                "status": logic_response.get("coverage_status", "UNCLEAR"),
                "limitations": logic_response.get("limitations", []),
                "requirements": logic_response.get("requirements", [])
            }
        elif evaluation_type == 'claim':
            response["claim_analysis"] = {
                "eligibility": logic_response.get("claim_eligibility", "NEEDS_REVIEW"),
                "documentation": logic_response.get("required_documentation", []),
                "process": logic_response.get("process_steps", [])
            }
        
        return response
    
    def _generate_no_results_response(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response when no relevant clauses found"""
        
        return {
            "query_analysis": {
                "original_query": parsed_query.get('original_query', ''),
                "intent": parsed_query.get('intent', ''),
                "entities": parsed_query.get('entities', []),
                "answer_type": parsed_query.get('answer_type', '')
            },
            
            "search_results": {
                "total_clauses_found": 0,
                "average_confidence": 0,
                "search_quality": "no_results"
            },
            
            "logic_evaluation": {
                "analysis": "No relevant clauses found in the policy documents for this query.",
                "confidence": 0,
                "recommendation": "Please rephrase your question or contact customer service for clarification."
            },
            
            "evidence": {
                "primary_clause": None,
                "supporting_clauses": []
            },
            
            "metadata": {
                "pipeline_version": "RTX3050_Advanced_v1.0",
                "gpu_accelerated": self.rtx_optimizer.gpu_available if self.rtx_optimizer else False,
                "timestamp": time.time()
            }
        }
    
    def _generate_error_response(self, parsed_query: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Generate error response"""
        
        return {
            "query_analysis": {
                "original_query": parsed_query.get('original_query', ''),
                "error": True
            },
            
            "search_results": {
                "total_clauses_found": 0,
                "average_confidence": 0,
                "search_quality": "error"
            },
            
            "logic_evaluation": {
                "analysis": f"Processing error occurred: {error_msg}",
                "confidence": 0,
                "error": True
            },
            
            "evidence": {
                "primary_clause": None,
                "supporting_clauses": []
            },
            
            "metadata": {
                "pipeline_version": "RTX3050_Advanced_v1.0",
                "gpu_accelerated": False,
                "error": error_msg,
                "timestamp": time.time()
            }
        }

# Export class
__all__ = ['RTX3050LogicEvaluator']
