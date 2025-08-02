#!/usr/bin/env python3
"""
Advanced RTX 3050 Optimized Pipeline
Stage 1: Input Documents & Stage 2: LLM Parser
"""

import os
import re
import json
import time
from typing import Dict, List, Any, Optional
from groq import Groq
import fitz  # PyMuPDF
import requests
from rtx3050_optimizer import rtx_optimizer
from multi_format_processor import MultiFormatProcessor

class AdvancedDocumentProcessor:
    """
    Stage 1: Input Documents Processing + Stage 2: LLM Parser
    Optimized PDF processing with RTX 3050 acceleration
    """
    
    def __init__(self, groq_api_key: str, use_gpu=True):
        # Initialize both document processing and LLM parsing
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.rtx_optimizer = rtx_optimizer if use_gpu else None
        self.use_gpu = use_gpu and rtx_optimizer.gpu_available
        self.multi_format_processor = MultiFormatProcessor()
        
        # Insurance-specific entity mapping for query parsing
        self.insurance_entities = {
            'grace_period': [
                'grace', 'payment deadline', 'premium due', 'late payment', 
                'overdue', 'extended payment', 'delayed payment', 'penalty period'
            ],
            'waiting_period': [
                'waiting', 'pre-existing', 'coverage starts', 'initial wait',
                'exclusion period', 'qualifying period', 'cooling period'
            ],
            'maternity': [
                'pregnancy', 'maternity', 'childbirth', 'delivery', 'prenatal',
                'postnatal', 'conception', 'newborn', 'infant care'
            ],
            'coverage_limits': [
                'sum insured', 'coverage amount', 'benefit limit', 'maximum coverage',
                'policy limit', 'annual limit', 'lifetime limit', 'sub-limit'
            ],
            'exclusions': [
                'excluded', 'not covered', 'exceptions', 'limitations',
                'restrictions', 'prohibited', 'barred', 'excepted'
            ]
        }
        
        print(f"🚀 Advanced Document Processor initialized")
        print(f"🎮 RTX 3050 optimization: {'✅ Active' if self.use_gpu else '❌ CPU mode'}")
    
    def process_document(self, blob_url: str) -> Dict[str, Any]:
        """Process document (PDF/DOCX/Email) with RTX 3050 optimization"""
        
        start_time = time.time()
        
        try:
            print(f"📥 Processing document: {blob_url[:50]}...")
            
            # Use multi-format processor
            result = self.multi_format_processor.process_document_url(blob_url)
            
            if result.get('error'):
                return result
            
            # Fallback to original PDF processing if multi-format fails
            if not result.get('sections'):
                print(f"⚠️ Multi-format failed, trying PDF fallback...")
                response = requests.get(blob_url, timeout=30)
                response.raise_for_status()
                doc = fitz.open(stream=response.content, filetype="pdf")
                
                # Extract text and metadata (PDF fallback)
                full_text = ""
                sections = []
                page_count = len(doc)
                
                for page_num in range(page_count):
                    page = doc[page_num]
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                        for para in paragraphs:
                            if len(para) > 50:
                                sections.append({
                                    'page': page_num + 1,
                                    'text': para,
                                    'length': len(para)
                                })
                
                doc.close()
                
                result = {
                    'full_text': full_text.strip(),
                    'sections': sections,
                    'page_count': page_count,
                    'format': 'PDF_FALLBACK'
                }
            
            # RTX 3050 memory optimization
            if self.rtx_optimizer:
                self.rtx_optimizer.optimize_memory()
            
            processing_time = time.time() - start_time
            
            # Add processing metadata
            result.update({
                'char_count': len(result.get('full_text', '')),
                'processing_time': processing_time,
                'rtx3050_optimized': self.use_gpu
            })
            
            sections_count = len(result.get('sections', []))
            char_count = result.get('char_count', 0)
            doc_format = result.get('format', 'UNKNOWN')
            
            print(f"✅ {doc_format} document processed: {sections_count} sections, {char_count} chars in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"❌ Document processing error: {e}")
            return {'error': str(e)}
    
    def process_documents(self, pdf_paths: List[str]) -> Optional[Dict[str, Any]]:
        """
        Process multiple local PDF files
        """
        
        start_time = time.time()
        
        try:
            all_sections = []
            
            for pdf_path in pdf_paths:
                if not os.path.exists(pdf_path):
                    print(f"⚠️ File not found: {pdf_path}")
                    continue
                
                print(f"📄 Processing local file: {pdf_path}")
                
                # Process local PDF file
                doc = fitz.open(pdf_path)
                
                sections = []
                full_text = ""
                
                # Extract text from each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        
                        # Enhanced section extraction for better coverage
                        # Method 1: Paragraph-based sections
                        paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                        for j, para in enumerate(paragraphs):
                            if len(para) > 30:  # Lowered threshold to capture more content
                                sections.append({
                                    'page': page_num + 1,
                                    'text': para,
                                    'length': len(para),
                                    'section_type': 'paragraph',
                                    'section_id': f"page_{page_num+1}_para_{j+1}"
                                })
                        
                        # Method 2: Sentence-based sections for detailed coverage
                        sentences = [s.strip() for s in page_text.split('.') if s.strip()]
                        current_sentence_group = ""
                        sentence_count = 0
                        
                        for sentence in sentences:
                            if sentence_count < 3 and len(current_sentence_group + sentence) < 300:
                                current_sentence_group += sentence + ". "
                                sentence_count += 1
                            else:
                                if current_sentence_group.strip() and len(current_sentence_group) > 50:
                                    sections.append({
                                        'page': page_num + 1,
                                        'text': current_sentence_group.strip(),
                                        'length': len(current_sentence_group),
                                        'section_type': 'sentence_group',
                                        'section_id': f"page_{page_num+1}_sent_group_{len(sections)}"
                                    })
                                current_sentence_group = sentence + ". "
                                sentence_count = 1
                        
                        # Add final sentence group
                        if current_sentence_group.strip() and len(current_sentence_group) > 50:
                            sections.append({
                                'page': page_num + 1,
                                'text': current_sentence_group.strip(),
                                'length': len(current_sentence_group),
                                'section_type': 'sentence_group',
                                'section_id': f"page_{page_num+1}_sent_group_{len(sections)}"
                            })
                        
                        # Method 3: Full page as section (for comprehensive coverage)
                        if len(page_text.strip()) > 100:
                            sections.append({
                                'page': page_num + 1,
                                'text': page_text.strip(),
                                'length': len(page_text),
                                'section_type': 'full_page',
                                'section_id': f"full_page_{page_num+1}"
                            })
                
                doc.close()
                all_sections.extend(sections)
                
                # RTX 3050 memory optimization
                if self.rtx_optimizer:
                    self.rtx_optimizer.optimize_memory()
            
            if not all_sections:
                print(f"❌ No sections extracted from documents")
                return None
            
            processing_time = time.time() - start_time
            
            result = {
                'documents': pdf_paths,
                'sections': all_sections,
                'total_sections': len(all_sections),
                'processing_time': processing_time
            }
            
            print(f"✅ Documents processed: {len(all_sections)} total sections in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Documents processing error: {e}")
            return None
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Stage 2: Parse and understand query with LLM
        """
        
        start_time = time.time()
        
        try:
            print(f"🧠 Parsing query: '{query[:50]}...'")
            
            # Create a simple parser if no Groq client
            if not self.groq_client:
                return self._simple_query_parsing(query)
            
            # Enhanced prompt for insurance query parsing
            prompt = f"""
            Analyze this insurance-related query and extract structured information:
            
            Query: "{query}"
            
            Please provide a JSON response with:
            1. intent: The main purpose (coverage_check, claim_process, premium_info, policy_details, etc.)
            2. entities: List of important terms, amounts, conditions mentioned
            3. context: Additional context or constraints
            4. answer_type: Expected answer format (yes_no, amount, process_steps, list, description)
            
            Respond with valid JSON only.
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert insurance query analyzer. Extract structured information from user queries."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                parsed_result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback to simple parsing
                parsed_result = self._simple_query_parsing(query)
            
            # Add original query and processing time
            parsed_result['original_query'] = query
            parsed_result['processing_time'] = time.time() - start_time
            
            print(f"✅ Query parsed: {parsed_result.get('intent', 'unknown')} intent")
            
            return parsed_result
            
        except Exception as e:
            print(f"⚠️ Query parsing error: {e}")
            return self._simple_query_parsing(query)
    
    def parse_query_fast(self, query: str) -> Dict[str, Any]:
        """🚀 OPTIMIZED query parsing with reduced LLM tokens"""
        
        try:
            # Use simple parsing for speed (skip LLM call for common cases)
            if not self.groq_client:
                return self._simple_query_parsing(query)
            
            # For optimization, use simple parsing for basic queries
            query_lower = query.lower()
            
            # Quick check for simple yes/no questions
            if any(word in query_lower for word in ['is', 'does', 'can', 'will', 'are']):
                return self._simple_query_parsing(query)
            
            # For complex queries, use reduced LLM call
            prompt = f"Query: {query}\nExtract: intent, key terms. JSON format only."
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=150  # Reduced from 500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            try:
                parsed_result = json.loads(response_text)
            except:
                parsed_result = self._simple_query_parsing(query)
            
            parsed_result['original_query'] = query
            
            return parsed_result
            
        except Exception as e:
            return self._simple_query_parsing(query)
    
    def _simple_query_parsing(self, query: str) -> Dict[str, Any]:
        """Fallback simple query parsing without LLM"""
        
        query_lower = query.lower()
        
        # Simple intent detection
        intent = "general"
        if any(word in query_lower for word in ['cover', 'covered', 'benefit']):
            intent = "coverage_check"
        elif any(word in query_lower for word in ['claim', 'reimburse', 'submit']):
            intent = "claim_process"
        elif any(word in query_lower for word in ['premium', 'cost', 'pay', 'price']):
            intent = "premium_info"
        elif any(word in query_lower for word in ['deductible', 'copay']):
            intent = "deductible_info"
        
        # Simple entity extraction
        entities = []
        insurance_terms = ['premium', 'deductible', 'copay', 'coverage', 'claim', 'policy', 'benefit']
        for term in insurance_terms:
            if term in query_lower:
                entities.append(term)
        
        # Answer type detection
        answer_type = "description"
        if '?' in query and any(word in query_lower for word in ['what', 'how much', 'amount']):
            answer_type = "amount"
        elif any(word in query_lower for word in ['how', 'process', 'steps']):
            answer_type = "process_steps"
        elif any(word in query_lower for word in ['is', 'are', 'covered']):
            answer_type = "yes_no"
        
        return {
            'intent': intent,
            'entities': entities,
            'context': query,
            'answer_type': answer_type,
            'original_query': query,
            'processing_time': 0.001,
            'method': 'simple_fallback'
        }

class AdvancedQueryParser:
    """
    Stage 2: LLM Parser
    Extract structured query understanding with RTX 3050 optimization
    """
    
    def __init__(self, groq_api_key: str, use_gpu=True):
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.rtx_optimizer = rtx_optimizer if use_gpu else None
        self.use_gpu = use_gpu and rtx_optimizer.gpu_available
        
        # Insurance-specific entity mapping
        self.insurance_entities = {
            'grace_period': [
                'grace', 'payment deadline', 'premium due', 'late payment', 
                'overdue', 'extended payment', 'delayed payment', 'penalty period'
            ],
            'waiting_period': [
                'waiting', 'pre-existing', 'exclusion period', 'cooling-off', 
                'initial waiting', 'probationary', 'delay in coverage', 'no coverage period'
            ],
            'coverage': [
                'covered', 'benefit', 'medical expenses', 'treatment', 'included', 
                'protection', 'health services', 'insurance benefits', 'what is covered', 
                'coverage limit', 'scope of cover', 'covered conditions', 'policy benefits'
            ],
            'premium': [
                'premium', 'payment', 'cost', 'fee', 'monthly charge', 'annual cost', 
                'insurance price', 'policy fee', 'billing', 'installment', 'premium amount'
            ],
            'claim': [
                'claim', 'reimbursement', 'settlement', 'claim form', 'file a claim', 
                'claim status', 'payment request', 'claim process', 'submit a claim', 
                'compensation request', 'insurance payout'
            ],
            'policy_term': [
                'term', 'duration', 'validity', 'period', 'policy length', 
                'coverage period', 'contract term', 'expiry date', 'renewal term'
            ],
            'contact': [
                'contact', 'customer service', 'helpline', 'support', 
                'get in touch', 'phone number', 'email us', 'customer care', 
                'assistance', 'talk to agent', 'service center', 'helpdesk'
            ],
            'renewal': [
                'renewal', 'renew', 'policy extension', 'continue coverage', 
                'reapply', 'renewal date', 'policy renewal', 'extend plan'
            ],
            'deductible': [
                'deductible', 'out-of-pocket', 'excess amount', 'minimum payment', 
                'deductible amount', 'before insurance pays', 'personal cost'
            ],
            'network': [
                'network', 'in-network', 'out-of-network', 'approved hospitals', 
                'partner clinics', 'network provider', 'hospital list', 
                'provider directory'
            ],
            'exclusions': [
                'exclusions', 'not covered', 'exceptions', 'limitations', 
                'policy exclusions', 'excluded treatments', 'non-covered'
            ],
            'co_payment': [
                'co-payment', 'copay', 'cost sharing', 'shared expenses', 
                'policyholder pays', 'copayment amount', 'split cost'
            ],
            'beneficiary': [
                'beneficiary', 'nominee', 'claim receiver', 'insured person', 
                'benefit recipient', 'assigned person'
            ]
        }

        
        print(f"🧠 Advanced Query Parser initialized")
        print(f"🎮 RTX 3050 optimization: {'✅ Active' if self.use_gpu else '❌ CPU mode'}")
    
    def parse_query(self, question: str) -> Dict[str, Any]:
        """Parse query into structured format with intent and entities"""
        
        start_time = time.time()
        
        try:
            # Fast local entity extraction first
            parsed_query = self._extract_entities_fast(question)
            
            # Enhance with LLM if available
            if self.groq_client:
                llm_enhancement = self._enhance_with_llm(question, parsed_query)
                parsed_query.update(llm_enhancement)
            
            # RTX 3050 optimization
            if self.rtx_optimizer:
                self.rtx_optimizer.optimize_memory()
            
            parsing_time = time.time() - start_time
            parsed_query['parsing_time'] = parsing_time
            parsed_query['rtx3050_optimized'] = self.use_gpu
            
            print(f">> Query parsed: {parsed_query['intent']} -> {parsed_query['primary_entity']} ({parsing_time:.2f}s)")
            return parsed_query
            
        except Exception as e:
            print(f"❌ Query parsing error: {e}")
            return {
                'intent': 'general_inquiry',
                'primary_entity': 'unknown',
                'keywords': question.lower().split(),
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _extract_entities_fast(self, question: str) -> Dict[str, Any]:
        """Fast local entity extraction optimized for insurance queries"""
        
        question_lower = question.lower()
        
        # Detect primary entity
        primary_entity = 'unknown'
        confidence = 0.5
        intent = 'general_inquiry'
        
        for entity, keywords in self.insurance_entities.items():
            for keyword in keywords:
                if keyword in question_lower:
                    primary_entity = entity
                    confidence = 0.8
                    break
            if primary_entity != 'unknown':
                break
        
        # Determine intent based on question pattern
        if any(word in question_lower for word in ['what is', 'what are', 'define']):
            intent = 'definition_lookup'
        elif any(word in question_lower for word in ['how much', 'amount', 'cost']):
            intent = 'amount_inquiry'
        elif any(word in question_lower for word in ['when', 'time', 'period']):
            intent = 'time_inquiry'
        elif any(word in question_lower for word in ['contact', 'phone', 'email']):
            intent = 'contact_inquiry'
        elif any(word in question_lower for word in ['covered', 'cover', 'include']):
            intent = 'coverage_inquiry'
        
        # Extract key terms
        keywords = [word for word in question_lower.split() if len(word) > 3]
        
        return {
            'intent': intent,
            'primary_entity': primary_entity,
            'keywords': keywords,
            'confidence': confidence,
            'question_type': 'insurance_policy'
        }
    
    def _enhance_with_llm(self, question: str, base_parse: Dict) -> Dict[str, Any]:
        """Enhance parsing with LLM for better understanding"""
        
        try:
            prompt = f"""Analyze this insurance policy question and extract structured information:

Question: "{question}"

Extract:
1. Intent (definition_lookup, amount_inquiry, time_inquiry, contact_inquiry, coverage_inquiry)
2. Entity (grace_period, waiting_period, coverage, premium, claim, contact)
3. Specific focus (what specific aspect they're asking about)
4. Answer type expected (time_duration, monetary_amount, yes_no, contact_info, description)

Respond with only JSON:
{{
    "intent": "...",
    "entity": "...", 
    "specific_focus": "...",
    "answer_type": "...",
    "confidence": 0.9
}}"""

            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            # Parse LLM response
            llm_result = json.loads(response.choices[0].message.content.strip())
            
            return {
                'llm_intent': llm_result.get('intent', base_parse['intent']),
                'llm_entity': llm_result.get('entity', base_parse['primary_entity']),
                'specific_focus': llm_result.get('specific_focus', ''),
                'answer_type': llm_result.get('answer_type', 'description'),
                'llm_confidence': llm_result.get('confidence', 0.7),
                'llm_enhanced': True
            }
            
        except Exception as e:
            print(f"⚠️ LLM enhancement failed: {e}")
            return {'llm_enhanced': False}

# Export classes
__all__ = ['AdvancedDocumentProcessor', 'AdvancedQueryParser']
