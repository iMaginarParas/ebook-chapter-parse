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
from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Pydantic models for request/response
class PageBatchResponse(BaseModel):
    batch_number: int
    page_range: str  # e.g., "0-10", "10-20"
    text: str
    cleaned_text: Optional[str] = None
    word_count: int
    original_word_count: Optional[int] = None
    cleaned: bool = False
    improvement_ratio: Optional[float] = None
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
    title="Simple PDF Page Processor",
    description="Extract and clean pages from PDF ebooks in batches - Simple & Reliable",
    version="2.1.0"
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
    
    def extract_pages_from_pdf(self, pdf_path: str) -> Tuple[List[Dict], int, int]:
        """
        Extract pages from PDF in batches - Simple approach like the bot
        """
        logger.info(f"Starting simple PDF extraction from: {pdf_path}")
        
        try:
            # Try PyMuPDF first (better text extraction)
            return self._extract_with_pymupdf(pdf_path)
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}, trying PyPDF2...")
            try:
                return self._extract_with_pypdf2(pdf_path)
            except Exception as e2:
                logger.error(f"Both extraction methods failed: {e2}")
                return [], 0, 0
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Tuple[List[Dict], int, int]:
        """Extract using PyMuPDF - simple and reliable"""
        doc = fitz.open(pdf_path)
        
        # Find where story starts
        start_page = self._find_story_start_page(doc)
        logger.info(f"Story starts at page {start_page}")
        
        page_batches = []
        pages_processed = 0
        
        # Process pages in batches
        current_batch_text = ""
        current_batch_start = start_page
        batch_number = 1
        
        for page_num in range(start_page, len(doc)):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Basic cleaning only
                page_text = self._clean_page_text(page_text)
                
                if page_text.strip():
                    current_batch_text += page_text + "\n\n"
                    pages_processed += 1
                
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
                                'text': current_batch_text.strip(),
                                'word_count': word_count,
                                'pages_in_batch': pages_in_current_batch,
                                'cleaned': False
                            })
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
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Tuple[List[Dict], int, int]:
        """Fallback to PyPDF2"""
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
                    
                    # Basic cleaning
                    page_text = self._clean_page_text(page_text)
                    
                    if page_text.strip():
                        current_batch_text += page_text + "\n\n"
                        pages_processed += 1
                    
                    # Check if we should finalize this batch
                    pages_in_current_batch = page_num - current_batch_start + 1
                    if pages_in_current_batch >= self.PAGES_PER_BATCH or page_num == total_pages - 1:
                        if current_batch_text.strip():
                            word_count = len(current_batch_text.split())
                            if word_count >= self.MIN_BATCH_WORDS:
                                page_range = f"{current_batch_start}-{page_num}"
                                page_batches.append({
                                    'batch_number': batch_number,
                                    'page_range': page_range,
                                    'text': current_batch_text.strip(),
                                    'word_count': word_count,
                                    'pages_in_batch': pages_in_current_batch,
                                    'cleaned': False
                                })
                                batch_number += 1
                        
                        # Reset for next batch
                        current_batch_text = ""
                        current_batch_start = page_num + 1
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
            
            return page_batches, pages_processed, start_page
    
    def _find_story_start_page(self, doc) -> int:
        """Find where the actual story starts - simple approach like the bot"""
        story_indicators = [
            r'chapter\s+1',
            r'chapter\s+one', 
            r'prologue',
            r'part\s+one',
            r'part\s+1',
            r'once upon a time',
            r'it was',
            r'the story',
            r'in the beginning'
        ]
        
        for page_num in range(min(20, len(doc))):  # Check first 20 pages
            try:
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                # Look for story indicators
                for indicator in story_indicators:
                    if re.search(indicator, text):
                        logger.info(f"Found story indicator '{indicator}' on page {page_num}")
                        return page_num
                
                # If page has substantial narrative content
                if len(text) > 500 and self._is_narrative_text(text):
                    logger.info(f"Found narrative content on page {page_num}")
                    return page_num
                        
            except Exception as e:
                logger.warning(f"Error checking page {page_num}: {e}")
                continue
        
        # Default: skip first 5 pages
        default_start = min(5, len(doc) // 4)
        logger.info(f"Using default start page: {default_start}")
        return default_start
    
    def _is_narrative_text(self, text: str) -> bool:
        """Check if text appears to be narrative content - simple approach"""
        # Count sentences and narrative indicators
        sentences = len(re.findall(r'[.!?]+', text))
        narrative_words = len(re.findall(r'\b(said|asked|replied|thought|felt|saw|heard|walked|ran|looked)\b', text.lower()))
        
        # Check for common non-narrative patterns
        non_narrative_patterns = [
            r'table of contents',
            r'copyright',
            r'published by',
            r'isbn',
            r'all rights reserved',
            r'acknowledgments',
            r'dedication',
            r'about the author'
        ]
        
        for pattern in non_narrative_patterns:
            if re.search(pattern, text.lower()):
                return False
        
        # If it has many sentences and some narrative words, likely story content
        return sentences > 10 and narrative_words > 5
    
    def _clean_page_text(self, text: str) -> str:
        """Clean page text - simple and safe like the bot"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers - simple patterns only
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page\s+\d+.*$', '', text, flags=re.MULTILINE)
        
        # Remove common OCR artifacts - safe character filtering
        text = re.sub(r'[^\w\s.,!?;:"\'-]', '', text)
        
        # Fix common spacing issues - safe patterns only
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    async def clean_page_batch_text_with_ai(self, batch_text: str, page_range: str) -> str:
        """Clean page batch text using GPT-4o-mini via Replicate"""
        if not self.replicate_client:
            logger.warning(f"Replicate client not available for pages {page_range}, skipping AI cleaning")
            return batch_text
        
        try:
            logger.info(f"Cleaning text for pages {page_range} using AI...")
            
            # Truncate if too long
            if len(batch_text) > 12000:
                logger.info(f"Page batch too long ({len(batch_text)} chars), truncating")
                batch_text = batch_text[:12000] + "..."
            
            # Simple cleaning prompt - like the bot
            cleaning_prompt = f"""You are a professional text editor. Clean up this extracted PDF text by:

1. Remove OCR artifacts and page numbers
2. Fix broken words and sentences  
3. Remove header/footer text and page references
4. Correct spacing and punctuation
5. Merge broken paragraphs properly
6. Keep dialogue formatting intact
7. Remove any non-story content (page numbers, chapter markers that don't belong)
8. Maintain the original story structure and flow

IMPORTANT: Only return the cleaned story text. No explanations, no additional comments, just the clean readable text.

Original Text:
{batch_text}

Clean Text:"""

            # Call GPT-4o-mini
            cleaned_text = ""
            try:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    def sync_replicate_call():
                        result = ""
                        for event in self.replicate_client.stream(
                            "openai/gpt-4o-mini",
                            input={
                                "prompt": cleaning_prompt,
                                "max_tokens": 4000,
                                "temperature": 0.1,
                                "top_p": 0.9
                            }
                        ):
                            result += str(event)
                        return result
                    
                    cleaned_text = await loop.run_in_executor(executor, sync_replicate_call)
                
                # Simple post-processing
                cleaned_text = self._post_process_cleaned_text(cleaned_text)
                
                logger.info(f"Successfully cleaned pages {page_range}")
                return cleaned_text
                
            except Exception as ai_error:
                logger.warning(f"AI cleaning failed for pages {page_range}: {ai_error}")
                return batch_text
                
        except Exception as e:
            logger.error(f"Error in AI text cleaning: {e}")
            return batch_text
    
    def _post_process_cleaned_text(self, text: str) -> str:
        """Simple post-processing like the bot"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove AI meta-commentary
        text = re.sub(r'^(Here is the cleaned text:|Clean Text:|Cleaned version:).*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Fix paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix quotation marks
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Fix sentence spacing
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    async def clean_all_page_batches_parallel(self, page_batches: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """Clean all page batches in parallel"""
        logger.info(f"Starting AI cleaning for {len(page_batches)} page batches")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def clean_single_batch(batch_index: int, batch: Dict) -> tuple[int, Dict]:
            async with semaphore:
                try:
                    cleaned_text = await self.clean_page_batch_text_with_ai(
                        batch['text'], 
                        batch['page_range']
                    )
                    
                    cleaned_batch = {
                        'batch_number': batch['batch_number'],
                        'page_range': batch['page_range'],
                        'text': batch['text'],
                        'cleaned_text': cleaned_text,
                        'word_count': len(cleaned_text.split()),
                        'original_word_count': batch['word_count'],
                        'cleaned': bool(self.replicate_client),
                        'improvement_ratio': len(cleaned_text.split()) / batch['word_count'] if batch['word_count'] > 0 else 1.0,
                        'pages_in_batch': batch['pages_in_batch']
                    }
                    
                    return (batch_index, cleaned_batch)
                    
                except Exception as e:
                    logger.error(f"Error cleaning batch {batch_index}: {e}")
                    # Return with basic cleaning on error
                    fallback_batch = {
                        'batch_number': batch['batch_number'],
                        'page_range': batch['page_range'],
                        'text': batch['text'],
                        'cleaned_text': batch['text'],  # No cleaning
                        'word_count': batch['word_count'],
                        'original_word_count': batch['word_count'],
                        'cleaned': False,
                        'improvement_ratio': 1.0,
                        'pages_in_batch': batch['pages_in_batch']
                    }
                    return (batch_index, fallback_batch)
        
        # Create tasks
        tasks = [clean_single_batch(i, batch) for i, batch in enumerate(page_batches)]
        
        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by index
        result_batches = [None] * len(page_batches)
        exceptions_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in task {i}: {result}")
                exceptions_count += 1
                # Add fallback
                batch = page_batches[i]
                result_batches[i] = {
                    'batch_number': batch['batch_number'],
                    'page_range': batch['page_range'],
                    'text': batch['text'],
                    'cleaned_text': batch['text'],
                    'word_count': batch['word_count'],
                    'original_word_count': batch['word_count'],
                    'cleaned': False,
                    'improvement_ratio': 1.0,
                    'pages_in_batch': batch['pages_in_batch']
                }
            else:
                index, cleaned_batch = result
                result_batches[index] = cleaned_batch
        
        final_batches = [batch for batch in result_batches if batch is not None]
        
        logger.info(f"Cleaning completed: {len(final_batches)} batches, {exceptions_count} exceptions")
        return final_batches

# Initialize processor
try:
    processor = SimplePDFProcessor()
    logger.info("SimplePDFProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SimplePDFProcessor: {e}")
    processor = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Simple PDF Page Processor API - Reliable & Fast",
        "version": "2.1.0",
        "status": "Running" if processor else "Limited functionality",
        "ai_enabled": bool(processor and processor.replicate_client),
        "approach": "Simple & reliable like Telegram bot",
        "endpoints": {
            "POST /process-pdf": "Extract page batches (10 pages each)",
            "POST /process-pdf-with-ai": "Extract and clean with AI",
            "GET /health": "Health check"
        },
        "features": {
            "simple_extraction": "Basic PyMuPDF + PyPDF2 fallback",
            "smart_story_detection": "Finds where story starts",
            "page_batching": "10 pages per batch (0-10, 10-20, etc.)",
            "ai_cleaning": "OpenAI via Replicate for text cleaning",
            "reliable": "No complex regex that breaks"
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
        "version": "2.1.0"
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...)):
    """Extract page batches from PDF (basic cleaning only)"""
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if len(contents) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
        
        logger.info(f"Processing PDF: {file.filename} ({file_size_mb:.1f}MB)")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract page batches
            page_batches, pages_processed, story_start_page = processor.extract_pages_from_pdf(tmp_path)
            
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-pdf-with-ai", response_model=ProcessingResponse)
async def process_pdf_with_ai(file: UploadFile = File(...), max_concurrent: int = 5):
    """Extract and clean page batches with AI"""
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
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
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract page batches
            page_batches, pages_processed, story_start_page = processor.extract_pages_from_pdf(tmp_path)
            
            if not page_batches:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable page batches found. Ensure PDF contains extractable text."
                )
            
            logger.info(f"Extracted {len(page_batches)} batches. Starting AI cleaning...")
            
            # Clean with AI
            cleaned_batches = await processor.clean_all_page_batches_parallel(page_batches, max_concurrent)
            
            # Calculate stats
            total_words = sum(batch['word_count'] for batch in cleaned_batches)
            reading_time = total_words / 200
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            ai_cleaned_count = sum(1 for batch in cleaned_batches if batch['cleaned'])
            
            batch_responses = [PageBatchResponse(**batch) for batch in cleaned_batches]
            
            ai_status = "with AI cleaning" if processor.replicate_client else "basic cleaning only (AI unavailable)"
            
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF with AI: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting Simple PDF Page Processor API...")
    print("📚 Simple & reliable approach like Telegram bot")
    print("🔍 Focus: Story detection + basic extraction")
    print("🧹 AI cleaning: Let OpenAI handle the complex stuff")
    print("✅ No complex regex patterns that break")
    
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("✅ AI cleaning enabled (Replicate + OpenAI)")
    else:
        print("⚠️ AI cleaning disabled (no Replicate token)")
    
    uvicorn.run(app, host="0.0.0.0", port=port)