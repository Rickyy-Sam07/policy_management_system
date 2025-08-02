#!/usr/bin/env python3
"""
Advanced RTX 3050 Optimized Pipeline
Stage 4: Semantic Clause Matching
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import re
from rtx3050_optimizer import rtx_optimizer

class RTX3050ClauseMatcher:
    """
    Stage 4: Semantic Clause Matching
    Insurance-specific clause identification and scoring
    """
    
    def __init__(self):
        self.rtx_optimizer = rtx_optimizer if rtx_optimizer.gpu_available else None
        
        # Insurance domain-specific patterns
        self.insurance_patterns = {
            'coverage': [
                r'cover(?:age|ed|s)?',
                r'benefit(?:s)?',
                r'policy',
                r'insur(?:ance|ed)?',
                r'protect(?:ion|ed)?'
            ],
            'exclusion': [
                r'exclud(?:e|ed|es|ing)',
                r'not cover(?:ed)?',
                r'except(?:ion)?',
                r'limit(?:ation|ed)?',
                r'restriction'
            ],
            'claim': [
                r'claim(?:s)?',
                r'reimburse(?:ment)?',
                r'pay(?:ment|able)?',
                r'compensat(?:e|ion)',
                r'settle(?:ment)?'
            ],
            'premium': [
                r'premium(?:s)?',
                r'cost(?:s)?',
                r'price(?:s)?',
                r'fee(?:s)?',
                r'rate(?:s)?'
            ],
            'deductible': [
                r'deductible(?:s)?',
                r'co-?pay(?:ment)?',
                r'out.of.pocket',
                r'self.insur(?:ance|ed)'
            ],
            'term': [
                r'term(?:s)?',
                r'period',
                r'duration',
                r'expir(?:y|ation)',
                r'valid(?:ity)?'
            ]
        }
        
        # Confidence scoring weights
        self.scoring_weights = {
            'semantic_similarity': 0.4,
            'pattern_match': 0.3,
            'context_relevance': 0.2,
            'clause_type_match': 0.1
        }
        
        print(f"🎯 RTX 3050 Clause Matcher initialized")
    
    def match_clauses(
        self, 
        vector_results: List[Dict[str, Any]], 
        parsed_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Main clause matching with semantic scoring"""
        
        start_time = time.time()
        
        try:
            if not vector_results:
                print(f"❌ No vector results to match")
                return []
            
            print(f"🎯 Matching {len(vector_results)} clauses...")
            
            matched_clauses = []
            
            for result in vector_results:
                clause_score = self._score_clause(result, parsed_query)
                
                if clause_score['total_score'] > 0.3:  # Relevance threshold
                    matched_clause = {
                        'text': result['text'],
                        'similarity': result['similarity'],
                        'metadata': result['metadata'],
                        'clause_analysis': clause_score,
                        'rank': result['rank'],
                        'confidence': clause_score['total_score']
                    }
                    matched_clauses.append(matched_clause)
            
            # Sort by confidence score
            matched_clauses.sort(key=lambda x: x['confidence'], reverse=True)
            
            # RTX 3050 memory optimization
            if self.rtx_optimizer:
                self.rtx_optimizer.optimize_memory()
            
            matching_time = time.time() - start_time
            
            print(f"✅ Clause matching: {len(matched_clauses)} relevant clauses in {matching_time:.3f}s")
            
            return matched_clauses
            
        except Exception as e:
            print(f"❌ Clause matching error: {e}")
            return []
    
    def match_clauses_fast(
        self, 
        vector_results: List[Dict[str, Any]], 
        parsed_query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """🚀 OPTIMIZED clause matching with reduced processing"""
        
        try:
            if not vector_results:
                return []
            
            # Fast processing: take top 4 results and do minimal scoring
            top_results = vector_results[:4]  # Reduced from all results
            
            matched_clauses = []
            
            for result in top_results:
                # Simplified scoring (skip complex analysis)
                basic_score = result.get('similarity', 0.7)
                
                matched_clause = {
                    'text': result['text'],
                    'similarity': result['similarity'],
                    'metadata': result['metadata'],
                    'rank': result['rank'],
                    'confidence': basic_score
                }
                matched_clauses.append(matched_clause)
            
            # Sort by similarity score (simpler than complex scoring)
            matched_clauses.sort(key=lambda x: x['similarity'], reverse=True)
            
            return matched_clauses[:3]  # Return top 3 for speed
            
        except Exception as e:
            return []
    
    def _score_clause(self, result: Dict[str, Any], parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Score individual clause for relevance"""
        
        text = result['text'].lower()
        similarity = result['similarity']
        
        # Extract query components
        intent = parsed_query.get('intent', '').lower()
        entities = [e.lower() for e in parsed_query.get('entities', [])]
        context = parsed_query.get('context', '').lower()
        answer_type = parsed_query.get('answer_type', '').lower()
        
        # 1. Semantic similarity score (from vector search)
        semantic_score = min(similarity, 1.0)  # Ensure max 1.0
        
        # 2. Pattern matching score
        pattern_score = self._calculate_pattern_score(text, intent, entities)
        
        # 3. Context relevance score
        context_score = self._calculate_context_score(text, context)
        
        # 4. Clause type matching score
        clause_type_score = self._calculate_clause_type_score(text, answer_type)
        
        # Calculate weighted total
        total_score = (
            semantic_score * self.scoring_weights['semantic_similarity'] +
            pattern_score * self.scoring_weights['pattern_match'] +
            context_score * self.scoring_weights['context_relevance'] +
            clause_type_score * self.scoring_weights['clause_type_match']
        )
        
        return {
            'semantic_score': semantic_score,
            'pattern_score': pattern_score,
            'context_score': context_score,
            'clause_type_score': clause_type_score,
            'total_score': total_score,
            'matched_patterns': self._get_matched_patterns(text),
            'clause_type': self._identify_clause_type(text)
        }
    
    def _calculate_pattern_score(self, text: str, intent: str, entities: List[str]) -> float:
        """Calculate pattern matching score"""
        
        score = 0.0
        matches = 0
        total_patterns = 0
        
        # Check insurance domain patterns
        for category, patterns in self.insurance_patterns.items():
            for pattern in patterns:
                total_patterns += 1
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1.0
                    matches += 1
                    
                    # Bonus for intent matching
                    if category in intent:
                        score += 0.5
        
        # Check entity patterns
        for entity in entities:
            if entity in text:
                score += 0.5
                matches += 1
        
        # Normalize score
        if total_patterns > 0:
            normalized_score = min(score / (total_patterns * 0.5), 1.0)
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    def _calculate_context_score(self, text: str, context: str) -> float:
        """Calculate contextual relevance score"""
        
        if not context:
            return 0.5  # Neutral if no context
        
        context_words = context.split()
        matches = 0
        
        for word in context_words:
            if len(word) > 3 and word.lower() in text:
                matches += 1
        
        if len(context_words) > 0:
            score = matches / len(context_words)
        else:
            score = 0.0
        
        return min(score, 1.0)
    
    def _calculate_clause_type_score(self, text: str, answer_type: str) -> float:
        """Calculate clause type matching score"""
        
        if not answer_type:
            return 0.5  # Neutral if no specific type
        
        type_patterns = {
            'coverage': ['cover', 'benefit', 'include', 'provide'],
            'exclusion': ['exclude', 'not cover', 'except', 'limit'],
            'amount': ['amount', 'dollar', '$', 'cost', 'price'],
            'condition': ['if', 'when', 'condition', 'require', 'must'],
            'process': ['process', 'procedure', 'step', 'how', 'submit']
        }
        
        if answer_type.lower() in type_patterns:
            patterns = type_patterns[answer_type.lower()]
            matches = sum(1 for pattern in patterns if pattern in text)
            score = min(matches / len(patterns), 1.0)
        else:
            score = 0.5
        
        return score
    
    def _get_matched_patterns(self, text: str) -> List[str]:
        """Get list of matched insurance patterns"""
        
        matched = []
        
        for category, patterns in self.insurance_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    matched.append(f"{category}:{pattern}")
        
        return matched
    
    def _identify_clause_type(self, text: str) -> str:
        """Identify the primary clause type"""
        
        type_scores = {}
        
        for category, patterns in self.insurance_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            type_scores[category] = score
        
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return 'general'
    
    def enhance_clause_context(self, matched_clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance clauses with additional context and relationships"""
        
        try:
            for clause in matched_clauses:
                text = clause['text']
                
                # Add insurance-specific enhancements
                clause['enhancements'] = {
                    'key_terms': self._extract_key_terms(text),
                    'monetary_values': self._extract_monetary_values(text),
                    'conditions': self._extract_conditions(text),
                    'exceptions': self._extract_exceptions(text),
                    'references': self._extract_references(text)
                }
            
            return matched_clauses
            
        except Exception as e:
            print(f"⚠️ Enhancement error: {e}")
            return matched_clauses
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important insurance terms"""
        
        key_terms = []
        
        # Insurance-specific terms
        insurance_terms = [
            'policy', 'coverage', 'benefit', 'premium', 'deductible',
            'claim', 'exclusion', 'limitation', 'copayment', 'coinsurance'
        ]
        
        for term in insurance_terms:
            if re.search(rf'\b{term}\b', text, re.IGNORECASE):
                key_terms.append(term)
        
        return key_terms
    
    def _extract_monetary_values(self, text: str) -> List[str]:
        """Extract monetary amounts"""
        
        money_patterns = [
            r'\$[\d,]+(?:\.\d{2})?',
            r'[\d,]+\s*dollar',
            r'[\d,]+\s*USD'
        ]
        
        values = []
        for pattern in money_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            values.extend(matches)
        
        return values
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditional statements"""
        
        condition_patterns = [
            r'if\s+[^.]+',
            r'when\s+[^.]+',
            r'provided\s+that\s+[^.]+',
            r'subject\s+to\s+[^.]+'
        ]
        
        conditions = []
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend(matches)
        
        return conditions[:3]  # Limit to top 3
    
    def _extract_exceptions(self, text: str) -> List[str]:
        """Extract exception clauses"""
        
        exception_patterns = [
            r'except\s+[^.]+',
            r'excluding\s+[^.]+',
            r'does\s+not\s+cover\s+[^.]+',
            r'not\s+applicable\s+[^.]+'
        ]
        
        exceptions = []
        for pattern in exception_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            exceptions.extend(matches)
        
        return exceptions[:3]  # Limit to top 3
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract section/clause references"""
        
        reference_patterns = [
            r'section\s+\d+',
            r'clause\s+\d+',
            r'paragraph\s+\d+',
            r'article\s+\d+'
        ]
        
        references = []
        for pattern in reference_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return references

# Export class
__all__ = ['RTX3050ClauseMatcher']
