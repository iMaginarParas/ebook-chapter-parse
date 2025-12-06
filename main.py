import os
import logging
import tempfile
import json
import asyncio
import gc
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor
import psutil

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Simple PDF processing - like the bot
import PyPDF2
import fitz  # PyMuPDF
import re

# Replicate for AI processing
import replicate

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_json(message)
            except:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)

manager = ConnectionManager()

# Pydantic models for request/response
class PageBatchResponse(BaseModel):
    batch_number: int
    page_range: str  # e.g., "0-10", "10-20"
    cleaned_text: str  # Only cleaned text - matches Flutter model
    word_count: int
    cleaned: bool = False
    pages_in_batch: int

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    total_page_batches: int
    total_words: int
    estimated_reading_time_minutes: float
    page_batches: List[PageBatchResponse]
    processing_time_seconds: float
    memory_usage_mb: Optional[float] = None
    pages_processed: Optional[int] = None
    story_start_page: Optional[int] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="PDF Processor with WebSocket Support",
    description="Extract and clean pages from PDF ebooks with real-time progress updates",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimplePDFProcessor:
    def __init__(self):
        # Initialize Replicate client
        self.replicate_client = None
        self._init_replicate_client()
        
        # Simple configuration
        self.PAGES_PER_BATCH = 10
        self.MAX_MEMORY_MB = 512
        self.MIN_BATCH_WORDS = 100
        
        # For WebSocket progress updates
        self.current_client_id: Optional[str] = None
    
    def _init_replicate_client(self):
        """Initialize Replicate client if API token is available"""
        try:
            api_token = os.environ.get("REPLICATE_API_TOKEN")
            if api_token:
                self.replicate_client = replicate.Client(api_token=api_token)
                logger.info("Replicate client initialized successfully")
            else:
                logger.warning("REPLICATE_API_TOKEN not found. AI cleaning will be disabled.")
                self.replicate_client = None
        except Exception as e:
            logger.error(f"Failed to initialize Replicate client: {e}")
            self.replicate_client = None
    
    async def _send_progress(self, stage: str, message: str, progress: int = 0, data: dict = None):
        """Send progress update via WebSocket"""
        if self.current_client_id:
            update = {
                "type": "progress",
                "stage": stage,
                "message": message,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            }
            if data:
                update["data"] = data
            
            await manager.send_message(self.current_client_id, update)
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.MAX_MEMORY_MB:
                logger.info(f"Memory usage high ({memory_mb:.1f}MB), running garbage collection")
                gc.collect()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory after GC: {memory_mb:.1f}MB")
            
            return memory_mb
        except:
            return 0.0
    
    async def extract_pages_from_pdf_async(self, pdf_path: str) -> Tuple[List[Dict], int, int]:
        """
        Extract pages from PDF in batches with WebSocket progress updates
        """
        await self._send_progress("extraction", "Starting PDF extraction...", 0)
        logger.info(f"Starting simple PDF extraction from: {pdf_path}")
        
        try:
            # Try PyMuPDF first (better text extraction)
            result = await self._extract_with_pymupdf_async(pdf_path)
            await self._send_progress("extraction", "PDF extraction completed", 100)
            return result
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}, trying PyPDF2...")
            await self._send_progress("extraction", "Trying alternative extraction method...", 50)
            try:
                result = await self._extract_with_pypdf2_async(pdf_path)
                await self._send_progress("extraction", "PDF extraction completed", 100)
                return result
            except Exception as e2:
                logger.error(f"Both extraction methods failed: {e2}")
                await self._send_progress("extraction", f"Extraction failed: {str(e2)}", 0)
                return [], 0, 0
    
    async def _extract_with_pymupdf_async(self, pdf_path: str) -> Tuple[List[Dict], int, int]:
        """Extract using PyMuPDF with async progress updates"""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        await self._send_progress("extraction", f"Analyzing {total_pages} pages...", 10)
        
        # Find where story starts
        start_page = self._find_story_start_page(doc)
        logger.info(f"Story starts at page {start_page}")
        await self._send_progress("extraction", f"Story detected at page {start_page}", 20)
        
        page_batches = []
        pages_processed = 0
        
        # Process pages in batches
        current_batch_text = ""
        current_batch_start = start_page
        batch_number = 1
        
        pages_to_process = total_pages - start_page
        
        for page_num in range(start_page, len(doc)):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Basic cleaning only
                page_text = self._clean_page_text(page_text)
                
                if page_text.strip():
                    current_batch_text += page_text + "\n\n"
                    pages_processed += 1
                
                # Update progress
                progress = 20 + int((page_num - start_page) / pages_to_process * 60)
                if page_num % 5 == 0:  # Update every 5 pages
                    await self._send_progress(
                        "extraction", 
                        f"Processing page {page_num + 1}/{total_pages}...",
                        progress
                    )
                
                # Check if we should finalize this batch
                pages_in_current_batch = page_num - current_batch_start + 1
                if pages_in_current_batch >= self.PAGES_PER_BATCH or page_num == len(doc) - 1:
                    if current_batch_text.strip():
                        word_count = len(current_batch_text.split())
                        if word_count >= self.MIN_BATCH_WORDS:
                            page_range = f"{current_batch_start}-{page_num}"
                            page_batches.append({
                                'batch_number': batch_number,
                                'page_range': page_range,
                                'cleaned_text': current_batch_text.strip(),
                                'word_count': word_count,
                                'pages_in_batch': pages_in_current_batch,
                                'cleaned': False
                            })
                            
                            await self._send_progress(
                                "extraction",
                                f"Batch {batch_number} extracted ({page_range})",
                                progress,
                                {"batch": batch_number, "pages": page_range}
                            )
                            
                            batch_number += 1
                    
                    # Reset for next batch
                    current_batch_text = ""
                    current_batch_start = page_num + 1
                
                # Memory check
                if page_num % 20 == 0:
                    self._check_memory_usage()
                
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        doc.close()
        return page_batches, pages_processed, start_page
    
    async def _extract_with_pypdf2_async(self, pdf_path: str) -> Tuple[List[Dict], int, int]:
        """Fallback to PyPDF2 with async updates"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Simple heuristic for start page
            start_page = max(3, total_pages // 20)
            
            page_batches = []
            pages_processed = 0
            
            # Process pages in batches
            current_batch_text = ""
            current_batch_start = start_page
            batch_number = 1
            
            for page_num in range(start_page, total_pages):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    page_text = self._clean_page_text(page_text)
                    
                    if page_text.strip():
                        current_batch_text += page_text + "\n\n"
                        pages_processed += 1
                    
                    # Progress updates
                    if page_num % 5 == 0:
                        progress = 20 + int((page_num - start_page) / (total_pages - start_page) * 60)
                        await self._send_progress(
                            "extraction",
                            f"Processing page {page_num + 1}/{total_pages}...",
                            progress
                        )
                    
                    pages_in_current_batch = page_num - current_batch_start + 1
                    if pages_in_current_batch >= self.PAGES_PER_BATCH or page_num == total_pages - 1:
                        if current_batch_text.strip():
                            word_count = len(current_batch_text.split())
                            if word_count >= self.MIN_BATCH_WORDS:
                                page_range = f"{current_batch_start}-{page_num}"
                                page_batches.append({
                                    'batch_number': batch_number,
                                    'page_range': page_range,
                                    'cleaned_text': current_batch_text.strip(),
                                    'word_count': word_count,
                                    'pages_in_batch': pages_in_current_batch,
                                    'cleaned': False
                                })
                                batch_number += 1
                        
                        current_batch_text = ""
                        current_batch_start = page_num + 1
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
            
            return page_batches, pages_processed, start_page
    
    def _find_story_start_page(self, doc) -> int:
        """Find where the actual story starts in the PDF"""
        try:
            # Keywords that indicate front matter
            front_matter_keywords = [
                'table of contents', 'copyright', 'dedication', 'acknowledgment',
                'preface', 'foreword', 'introduction', 'published by', 'isbn',
                'all rights reserved', 'contents'
            ]
            
            # Look at first 20% of pages
            max_pages_to_check = min(len(doc), max(10, len(doc) // 5))
            
            for page_num in range(max_pages_to_check):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text().lower()
                    
                    # Skip if it's clearly front matter
                    if any(keyword in text for keyword in front_matter_keywords):
                        continue
                    
                    # Check if page has substantial content
                    words = text.split()
                    if len(words) > 100:
                        # This looks like story content
                        return page_num
                
                except Exception as e:
                    logger.warning(f"Error checking page {page_num}: {e}")
                    continue
            
            # Default: skip first few pages
            return min(3, len(doc) - 1)
        
        except Exception as e:
            logger.error(f"Error finding story start: {e}")
            return 0
    
    def _clean_page_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (simple heuristic)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    async def clean_batch_with_ai_async(self, batch: Dict) -> Dict:
        """Clean a single batch with AI and send progress"""
        if not self.replicate_client:
            return batch
        
        try:
            batch_num = batch['batch_number']
            await self._send_progress(
                "ai_cleaning",
                f"AI cleaning batch {batch_num}...",
                0,
                {"batch": batch_num}
            )
            
            prompt = f"""Clean this text from an ebook. Remove:
- Page numbers, headers, footers
- Copyright notices, publisher info
- Navigation elements
- Excessive whitespace

Keep only the story content. Return the cleaned text directly.

Text:
{batch['cleaned_text']}"""

            output = await asyncio.to_thread(
                lambda: self.replicate_client.run(
                    "meta/meta-llama-3-70b-instruct",
                    input={
                        "prompt": prompt,
                        "max_tokens": 4000,
                        "temperature": 0.1
                    }
                )
            )
            
            cleaned = ''.join(output).strip()
            
            if cleaned and len(cleaned) > 50:
                batch['cleaned_text'] = cleaned
                batch['word_count'] = len(cleaned.split())
                batch['cleaned'] = True
                
                await self._send_progress(
                    "ai_cleaning",
                    f"Batch {batch_num} cleaned successfully",
                    100,
                    {"batch": batch_num, "cleaned": True}
                )
            
            return batch
        
        except Exception as e:
            logger.warning(f"AI cleaning failed for batch {batch['batch_number']}: {e}")
            await self._send_progress(
                "ai_cleaning",
                f"Batch {batch['batch_number']} - AI cleaning failed, using basic text",
                100,
                {"batch": batch['batch_number'], "cleaned": False}
            )
            return batch
    
    async def clean_all_page_batches_parallel_async(
        self, 
        page_batches: List[Dict], 
        max_concurrent: int = 5
    ) -> List[Dict]:
        """Clean all batches with AI in parallel with progress updates"""
        if not self.replicate_client:
            logger.info("AI cleaning disabled - no Replicate client")
            return page_batches
        
        await self._send_progress(
            "ai_cleaning",
            f"Starting AI cleaning for {len(page_batches)} batches...",
            0
        )
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def clean_with_semaphore(batch: Dict, index: int) -> Dict:
            async with semaphore:
                progress = int((index / len(page_batches)) * 100)
                await self._send_progress(
                    "ai_cleaning",
                    f"Cleaning batch {batch['batch_number']} ({index + 1}/{len(page_batches)})",
                    progress
                )
                return await self.clean_batch_with_ai_async(batch)
        
        tasks = [
            clean_with_semaphore(batch, i) 
            for i, batch in enumerate(page_batches)
        ]
        
        cleaned_batches = await asyncio.gather(*tasks)
        
        await self._send_progress(
            "ai_cleaning",
            "AI cleaning completed for all batches",
            100,
            {"total_batches": len(cleaned_batches)}
        )
        
        return cleaned_batches

# Initialize processor
try:
    processor = SimplePDFProcessor()
    logger.info("SimplePDFProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SimplePDFProcessor: {e}")
    processor = None

# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            # Echo back or handle client messages if needed
            await websocket.send_json({
                "type": "pong",
                "message": "Connection active",
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

# ============================================================================
# HTTP ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "PDF Processor API with WebSocket Support",
        "version": "3.0.0",
        "status": "Running" if processor else "Limited functionality",
        "ai_enabled": bool(processor and processor.replicate_client),
        "endpoints": {
            "POST /process-pdf": "Extract page batches (basic cleaning)",
            "POST /process-pdf-with-ai": "Extract and clean with AI",
            "WebSocket /ws/{client_id}": "Real-time progress updates",
            "GET /health": "Health check"
        },
        "features": {
            "websocket_updates": "Real-time progress during processing",
            "simple_extraction": "PyMuPDF + PyPDF2 fallback",
            "smart_story_detection": "Finds where story starts",
            "page_batching": "10 pages per batch",
            "ai_cleaning": "OpenAI via Replicate for text cleaning"
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    memory_usage = 0
    try:
        if processor:
            memory_usage = processor._check_memory_usage()
    except:
        pass
    
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "processor_available": bool(processor),
        "ai_enabled": bool(processor and processor.replicate_client),
        "memory_usage_mb": memory_usage,
        "active_connections": len(manager.active_connections),
        "version": "3.0.0"
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...), client_id: str = None):
    """Extract page batches from PDF with WebSocket progress updates"""
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    # Set client ID for progress updates
    if client_id:
        processor.current_client_id = client_id
        await manager.send_message(client_id, {
            "type": "started",
            "message": "Processing started",
            "timestamp": datetime.now().isoformat()
        })
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if len(contents) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
        
        logger.info(f"Processing PDF: {file.filename} ({file_size_mb:.1f}MB)")
        
        if client_id:
            await processor._send_progress(
                "upload",
                f"File uploaded: {file.filename} ({file_size_mb:.1f}MB)",
                100
            )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract page batches with progress updates
            page_batches, pages_processed, story_start_page = await processor.extract_pages_from_pdf_async(tmp_path)
            
            if not page_batches:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable page batches found. Ensure PDF contains extractable text."
                )
            
            # Calculate stats
            total_words = sum(batch['word_count'] for batch in page_batches)
            reading_time = total_words / 200
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            batch_responses = [PageBatchResponse(**batch) for batch in page_batches]
            
            logger.info(f"Processing completed: {len(page_batches)} batches, {total_words} words")
            
            if client_id:
                await manager.send_message(client_id, {
                    "type": "completed",
                    "message": "Processing completed successfully",
                    "data": {
                        "total_batches": len(page_batches),
                        "total_words": total_words,
                        "processing_time": processing_time
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed successfully with simple extraction",
                file_name=file.filename,
                total_page_batches=len(page_batches),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                page_batches=batch_responses,
                processing_time_seconds=processing_time,
                memory_usage_mb=memory_usage,
                pages_processed=pages_processed,
                story_start_page=story_start_page
            )
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            processor.current_client_id = None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        if client_id:
            await manager.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-pdf-with-ai", response_model=ProcessingResponse)
async def process_pdf_with_ai(
    file: UploadFile = File(...), 
    max_concurrent: int = 5,
    client_id: str = None
):
    """Extract and clean page batches with AI and WebSocket progress"""
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    # Set client ID for progress updates
    if client_id:
        processor.current_client_id = client_id
        await manager.send_message(client_id, {
            "type": "started",
            "message": "AI processing started",
            "timestamp": datetime.now().isoformat()
        })
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if len(contents) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
        
        if max_concurrent < 1 or max_concurrent > 20:
            raise HTTPException(status_code=400, detail="max_concurrent must be between 1 and 20")
        
        logger.info(f"Processing PDF with AI: {file.filename} ({file_size_mb:.1f}MB)")
        
        if client_id:
            await processor._send_progress(
                "upload",
                f"File uploaded: {file.filename} ({file_size_mb:.1f}MB)",
                100
            )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract page batches
            page_batches, pages_processed, story_start_page = await processor.extract_pages_from_pdf_async(tmp_path)
            
            if not page_batches:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable page batches found. Ensure PDF contains extractable text."
                )
            
            logger.info(f"Extracted {len(page_batches)} batches. Starting AI cleaning...")
            
            # Clean with AI
            cleaned_batches = await processor.clean_all_page_batches_parallel_async(
                page_batches, 
                max_concurrent
            )
            
            # Calculate stats
            total_words = sum(batch['word_count'] for batch in cleaned_batches)
            reading_time = total_words / 200
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            ai_cleaned_count = sum(1 for batch in cleaned_batches if batch['cleaned'])
            
            batch_responses = [PageBatchResponse(**batch) for batch in cleaned_batches]
            
            ai_status = "with AI cleaning" if processor.replicate_client else "basic cleaning only (AI unavailable)"
            
            if client_id:
                await manager.send_message(client_id, {
                    "type": "completed",
                    "message": f"Processing completed {ai_status}",
                    "data": {
                        "total_batches": len(cleaned_batches),
                        "ai_cleaned": ai_cleaned_count,
                        "total_words": total_words,
                        "processing_time": processing_time
                    },
                    "timestamp": datetime.now().isoformat()
                })
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed {ai_status}. {ai_cleaned_count}/{len(cleaned_batches)} batches cleaned with AI",
                file_name=file.filename,
                total_page_batches=len(cleaned_batches),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                page_batches=batch_responses,
                processing_time_seconds=processing_time,
                memory_usage_mb=memory_usage,
                pages_processed=pages_processed,
                story_start_page=story_start_page
            )
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            processor.current_client_id = None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF with AI: {e}")
        if client_id:
            await manager.send_message(client_id, {
                "type": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting PDF Processor API with WebSocket Support...")
    print("📡 WebSocket endpoint: /ws/{client_id}")
    print("📚 Real-time progress updates enabled")
    print("🔍 Story detection + batch extraction")
    print("🧹 AI cleaning available")
    
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("✅ AI cleaning enabled (Replicate + OpenAI)")
    else:
        print("⚠️ AI cleaning disabled (no Replicate token)")
    
    uvicorn.run(app, host="0.0.0.0", port=port)