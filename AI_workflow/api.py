"""
FastAPI Web API for Document Processing
======================================

RESTful API wrapper for the document processing system.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import shutil
from pathlib import Path
import uuid
import time
from datetime import datetime

# Import our document processing system
from document_processor import DocumentProcessor, DocumentProcessingResult

# API Models
class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    ocr_language: str = "en"
    apply_preprocessing: bool = True
    extract_metadata: bool = True


class ProcessingStatus(BaseModel):
    """Status model for async processing."""
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    message: Optional[str] = None


class ProcessingResponse(BaseModel):
    """Response model for document processing."""
    job_id: str
    success: bool
    document_type: str
    parser_used: str
    confidence_score: float
    processing_time: float
    extracted_text: str
    metadata: Dict[str, Any]
    attachments: List[Dict[str, Any]]
    error_message: Optional[str] = None


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    job_id: str
    total_documents: int
    successful_documents: int
    failed_documents: int
    processing_time: float
    results: List[ProcessingResponse]


# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="Intelligent document type detection and content extraction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
processor = DocumentProcessor()

# In-memory job storage (use Redis or database in production)
job_storage: Dict[str, Dict[str, Any]] = {}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/process",
            "batch": "/batch",
            "status": "/status/{job_id}",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_formats": ["pdf", "docx", "doc", "eml", "msg", "jpg", "jpeg", "png", "tiff", "bmp", "gif"]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "document_processor": "available",
            "pdf_parser": "available",
            "docx_parser": "available", 
            "email_parser": "available",
            "ocr_parser": "available"
        }
    }


@app.post("/process", response_model=ProcessingResponse)
async def process_document_api(
    file: UploadFile = File(...),
    ocr_language: str = "en",
    apply_preprocessing: bool = True
):
    """
    Process a single document.
    
    - **file**: Document file to process
    - **ocr_language**: Language for OCR processing (default: en)
    - **apply_preprocessing**: Apply image preprocessing for OCR (default: true)
    """
    job_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)
        
        try:
            # Process document
            processor_instance = DocumentProcessor(ocr_lang=ocr_language)
            result = processor_instance.process_document(temp_path)
            
            # Create response
            response = ProcessingResponse(
                job_id=job_id,
                success=result.parsing_success,
                document_type=str(result.detection_result.metadata.source_type) if result.detection_result else "unknown",
                parser_used=result.parser_used,
                confidence_score=result.confidence_score,
                processing_time=time.time() - start_time,
                extracted_text=result.extracted_text,
                metadata=result.metadata or {},
                attachments=result.attachments or [],
                error_message=result.error_message
            )
            
            return response
            
        finally:
            # Clean up temporary file
            temp_path.unlink(missing_ok=True)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/batch", response_model=Dict[str, Any])
async def process_batch_api(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    ocr_language: str = "en"
):
    """
    Process multiple documents in batch (async).
    
    - **files**: List of document files to process
    - **ocr_language**: Language for OCR processing (default: en)
    
    Returns job ID for status tracking.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max 50)")
    
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_storage[job_id] = {
        "status": "pending",
        "created_at": datetime.now(),
        "total_files": len(files),
        "processed_files": 0,
        "results": []
    }
    
    # Start background processing
    background_tasks.add_task(
        process_batch_background,
        job_id, files, ocr_language
    )
    
    return {
        "job_id": job_id,
        "status": "pending",
        "total_files": len(files),
        "message": "Batch processing started. Use /status/{job_id} to check progress."
    }


@app.get("/status/{job_id}")
async def get_processing_status(job_id: str):
    """
    Get processing status for a batch job.
    
    - **job_id**: Job ID returned from batch processing
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    progress = job_data["processed_files"] / job_data["total_files"] * 100 if job_data["total_files"] > 0 else 0
    
    return {
        "job_id": job_id,
        "status": job_data["status"],
        "created_at": job_data["created_at"],
        "completed_at": job_data.get("completed_at"),
        "progress": progress,
        "processed_files": job_data["processed_files"],
        "total_files": job_data["total_files"],
        "message": job_data.get("message", ""),
        "results": job_data["results"] if job_data["status"] == "completed" else []
    }


@app.get("/results/{job_id}")
async def get_batch_results(job_id: str):
    """
    Get detailed results for a completed batch job.
    
    - **job_id**: Job ID returned from batch processing
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    return BatchProcessingResponse(
        job_id=job_id,
        total_documents=job_data["total_files"],
        successful_documents=sum(1 for r in job_data["results"] if r.get("success", False)),
        failed_documents=sum(1 for r in job_data["results"] if not r.get("success", False)),
        processing_time=job_data.get("processing_time", 0),
        results=job_data["results"]
    )





async def process_batch_background(job_id: str, files: List[UploadFile], ocr_language: str):
    """Background task for batch processing."""
    start_time = time.time()
    
    try:
        job_storage[job_id]["status"] = "processing"
        job_storage[job_id]["message"] = "Processing documents..."
        
        processor_instance = DocumentProcessor(ocr_lang=ocr_language)
        results = []
        
        for i, file in enumerate(files):
            try:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                    file.file.seek(0)  # Reset file pointer
                    shutil.copyfileobj(file.file, temp_file)
                    temp_path = Path(temp_file.name)
                
                try:
                    # Process document
                    result = processor_instance.process_document(temp_path)
                    
                    # Convert to serializable format
                    result_dict = {
                        "filename": file.filename,
                        "success": result.parsing_success,
                        "document_type": str(result.detection_result.metadata.source_type) if result.detection_result else "unknown",
                        "parser_used": result.parser_used,
                        "confidence_score": result.confidence_score,
                        "processing_time": result.processing_time,
                        "extracted_text": result.extracted_text,
                        "text_length": len(result.extracted_text),
                        "metadata": result.metadata or {},
                        "attachments": result.attachments or [],
                        "error_message": result.error_message
                    }
                    
                    results.append(result_dict)
                    
                finally:
                    # Clean up
                    temp_path.unlink(missing_ok=True)
                
                # Update progress
                job_storage[job_id]["processed_files"] = i + 1
                progress = (i + 1) / len(files) * 100
                job_storage[job_id]["message"] = f"Processed {i + 1}/{len(files)} files ({progress:.1f}%)"
                
            except Exception as e:
                # Add error result
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error_message": str(e)
                })
                job_storage[job_id]["processed_files"] = i + 1
        
        # Mark as completed
        job_storage[job_id]["status"] = "completed"
        job_storage[job_id]["completed_at"] = datetime.now()
        job_storage[job_id]["processing_time"] = time.time() - start_time
        job_storage[job_id]["results"] = results
        job_storage[job_id]["message"] = f"Completed processing {len(files)} files"
        
    except Exception as e:
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["completed_at"] = datetime.now()
        job_storage[job_id]["message"] = f"Batch processing failed: {str(e)}"





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
