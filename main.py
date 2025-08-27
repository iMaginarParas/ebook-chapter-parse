import os
import logging
import tempfile
import json
import re
import asyncio
import gc
import traceback
from datetime import datetime
from typing import List, Dict, Optional, Callable, Tuple
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import psutil  # For memory monitoring

# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF processing
import PyPDF2
import fitz  # PyMuPDF for better text extraction
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine, LTChar
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams

# Replicate for AI processing
import replicate

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response - Modified for page-based processing
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
    page_batches: List[PageBatchResponse]  # Changed from chapters to page_batches
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
    title="PDF Ebook Page Processor",
    description="Extract and clean pages from PDF ebooks in batches using AI - Optimized for large files",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImprovedEbookPageProcessor:
    def __init__(self):
        # Initialize Replicate client only when needed for AI processing
        self.replicate_client = None
        self._init_replicate_client()
        
        # Processing configuration - Modified for page-based processing
        self.PAGES_PER_BATCH = 10  # Process 10 pages per batch
        self.PROCESSING_BATCH_SIZE = 5  # Process 5 page batches at once for memory management
        self.MAX_MEMORY_MB = 512  # Maximum memory usage before garbage collection
        self.MIN_BATCH_WORDS = 100  # Minimum words for a valid page batch
        self.MAX_PAGES_TO_SCAN = 50  # Maximum pages to scan for story start
    
    def _init_replicate_client(self):
        """Initialize Replicate client if API token is available"""
        try:
            api_token = os.environ.get("REPLICATE_API_TOKEN")
            if api_token:
                self.replicate_client = replicate.Client(api_token=api_token)
                logger.info("✅ Replicate client initialized successfully")
            else:
                logger.warning("⚠️ REPLICATE_API_TOKEN not found. AI cleaning will be disabled.")
                self.replicate_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Replicate client: {e}")
            self.replicate_client = None
    
    def _check_memory_usage(self) -> float:
        """Check current memory usage and trigger GC if needed"""
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
    
    async def clean_page_batch_text_with_ai(self, batch_text: str, page_range: str) -> str:
        """Clean page batch text using GPT-4o-mini via Replicate"""
        # Check if Replicate client is available
        if not self.replicate_client:
            logger.warning(f"Replicate client not available for pages {page_range}, using basic cleaning")
            return self._basic_text_cleaning(batch_text)
        
        try:
            logger.info(f"Cleaning text for pages {page_range} using AI...")
            
            # Truncate very long batches to avoid token limits
            if len(batch_text) > 12000:  # About 3000 tokens
                logger.info(f"Page batch too long ({len(batch_text)} chars), truncating for AI processing")
                batch_text = batch_text[:12000] + "..."
            
            # Create a detailed prompt for text cleaning
            cleaning_prompt = f"""You are a professional text editor. Clean up this extracted PDF text from pages {page_range} by:

1. Remove OCR artifacts, page numbers, headers/footers, and page markers
2. Fix broken words and sentences from PDF extraction
3. Correct spacing and punctuation errors
4. Merge broken paragraphs properly while preserving paragraph breaks
5. Keep dialogue formatting intact with proper quotes
6. Remove any non-story content (page markers, copyright text, footnotes)
7. Maintain the original story structure and narrative flow
8. Fix hyphenated words that broke across lines
9. Remove redundant content that appears across pages

IMPORTANT: Return ONLY the cleaned readable story text. No explanations or meta-commentary.

Text to clean:
{batch_text}

Cleaned text:"""

            # Call GPT-4o-mini for text cleaning using asyncio to run in thread pool
            cleaned_text = ""
            try:
                # Run the synchronous replicate call in a thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # Create a synchronous function for the replicate call
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
                    
                    # Run in thread pool to avoid blocking
                    cleaned_text = await loop.run_in_executor(executor, sync_replicate_call)
                
                # Post-process the AI output
                cleaned_text = self._post_process_cleaned_text(cleaned_text)
                
                logger.info(f"Successfully cleaned pages {page_range}")
                return cleaned_text
                
            except Exception as ai_error:
                logger.warning(f"AI cleaning failed for pages {page_range}: {ai_error}")
                # Fallback to basic cleaning
                return self._basic_text_cleaning(batch_text)
                
        except Exception as e:
            logger.error(f"Error in AI text cleaning: {e}")
            return self._basic_text_cleaning(batch_text)

    async def clean_all_page_batches_parallel(self, page_batches: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """Clean all page batches in parallel with concurrency limit"""
        logger.info(f"Starting parallel AI cleaning for {len(page_batches)} page batches with max {max_concurrent} concurrent tasks")
        
        # Log input page batches for debugging
        batch_ranges = [batch.get('page_range', f'Batch {batch.get("batch_number", "?")}') for batch in page_batches]
        logger.info(f"Input page batches: {batch_ranges}")
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def clean_single_batch(batch_index: int, batch: Dict) -> tuple[int, Dict]:
            async with semaphore:
                try:
                    logger.info(f"Starting AI cleaning for batch index {batch_index}: {batch['page_range']}")
                    
                    # Clean the batch text with AI
                    cleaned_text = await self.clean_page_batch_text_with_ai(
                        batch['text'], 
                        batch['page_range']
                    )
                    
                    # Create cleaned batch object
                    cleaned_batch = {
                        'batch_number': batch['batch_number'],
                        'page_range': batch['page_range'],
                        'text': batch['text'],  # Keep original
                        'cleaned_text': cleaned_text,  # Add cleaned version
                        'word_count': len(cleaned_text.split()),
                        'original_word_count': batch['word_count'],
                        'cleaned': bool(self.replicate_client),  # True if AI was used
                        'improvement_ratio': len(cleaned_text.split()) / batch['word_count'] if batch['word_count'] > 0 else 1.0,
                        'pages_in_batch': batch['pages_in_batch']
                    }
                    
                    logger.info(f"Completed cleaning batch index {batch_index}: {batch['page_range']} (original: {batch['word_count']} words -> cleaned: {cleaned_batch['word_count']} words)")
                    return (batch_index, cleaned_batch)
                    
                except Exception as e:
                    logger.error(f"Error cleaning page batch index {batch_index} '{batch['page_range']}': {e}")
                    # Return batch with basic cleaning on error
                    fallback_batch = {
                        'batch_number': batch['batch_number'],
                        'page_range': batch['page_range'],
                        'text': batch['text'],
                        'cleaned_text': self._basic_text_cleaning(batch['text']),
                        'word_count': batch['word_count'],
                        'original_word_count': batch['word_count'],
                        'cleaned': False,  # Mark as failed AI cleaning
                        'improvement_ratio': 1.0,
                        'pages_in_batch': batch['pages_in_batch']
                    }
                    return (batch_index, fallback_batch)
        
        # Create tasks for all page batches with their indices
        tasks = [clean_single_batch(i, batch) for i, batch in enumerate(page_batches)]
        
        # Run all tasks concurrently and wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by original index to maintain order and handle exceptions
        result_batches = [None] * len(page_batches)
        exceptions_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in task {i}: {result}")
                exceptions_count += 1
                # Add batch with basic cleaning as fallback
                batch = page_batches[i]
                result_batches[i] = {
                    'batch_number': batch['batch_number'],
                    'page_range': batch['page_range'],
                    'text': batch['text'],
                    'cleaned_text': self._basic_text_cleaning(batch['text']),
                    'word_count': batch['word_count'],
                    'original_word_count': batch['word_count'],
                    'cleaned': False,
                    'improvement_ratio': 1.0,
                    'pages_in_batch': batch['pages_in_batch']
                }
            else:
                # result is a tuple of (index, cleaned_batch)
                index, cleaned_batch = result
                result_batches[index] = cleaned_batch
        
        # Filter out any None values (shouldn't happen, but safety check)
        final_batches = [batch for batch in result_batches if batch is not None]
        
        # Log output batches for debugging
        output_ranges = [batch.get('page_range', f'Batch {batch.get("batch_number", "?")}') for batch in final_batches]
        
        logger.info(f"Parallel cleaning completed:")
        logger.info(f"  - Input page batches: {len(page_batches)}")
        logger.info(f"  - Output page batches: {len(final_batches)}")
        logger.info(f"  - Exceptions handled: {exceptions_count}")
        logger.info(f"  - Successfully AI cleaned: {sum(1 for batch in final_batches if batch.get('cleaned', False))}")
        logger.info(f"Output page batches: {output_ranges}")
        
        return final_batches
    
    def _post_process_cleaned_text(self, text: str) -> str:
        """Post-process AI cleaned text to ensure quality"""
        if not text:
            return ""
        
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Fix common AI output issues
        # Remove any meta-commentary that might slip through
        text = re.sub(r'^(Here is the cleaned text:|Clean Text:|Cleaned version:|Cleaned text:).*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Ensure proper paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix quotation marks spacing
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        text = re.sub(r"'\s+", "'", text)
        text = re.sub(r"\s+'", "'", text)
        
        # Ensure sentences end properly
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        # Fix common OCR errors that AI might miss
        text = re.sub(r'\b(\w+)-\s*\n\s*(\w+)\b', r'\1\2', text)  # Fix hyphenated words
        
        return text.strip()
    
    def _basic_text_cleaning(self, text: str) -> str:
        """Enhanced fallback text cleaning if AI fails"""
        if not text:
            return ""
        
        # Step 1: Fix broken words and hyphenation
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Step 2: Remove page numbers and headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip standalone page numbers and common page artifacts
            skip_patterns = [
                r'^\d+$',                        # Standalone page numbers
                r'^Page\s+\d+',                  # "Page X" headers
                r'^Chapter\s+\d+\s*$',           # Standalone "Chapter X"
                r'^\d+\s*$',                     # Numbers only
                r'^[^\w]+$',                     # Only punctuation/symbols
                r'.*copyright.*',                # Copyright lines
                r'.*all rights reserved.*',      # Rights lines
                r'^www\.',                       # Website URLs
                r'.*\.com.*',                    # More URLs
                r'^\s*[-=_]{3,}\s*$',            # Separator lines
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                continue
            
            # Skip lines with mostly non-alphabetic characters
            alpha_ratio = sum(c.isalpha() for c in line) / len(line) if len(line) > 0 else 0
            if alpha_ratio < 0.3:  # Less than 30% alphabetic
                continue
            
            cleaned_lines.append(line)
        
        # Step 3: Reconstruct paragraphs intelligently
        paragraphs = []
        current_paragraph = ""
        
        for line in cleaned_lines:
            # Check if this line starts a new paragraph
            starts_new_paragraph = (
                # Line starts with capital and current paragraph is substantial
                (line[0].isupper() and len(current_paragraph) > 100) or
                # Line starts with quote
                line.startswith(('"', "'", '"', "'")) or
                # Previous line ended with period and this starts with capital
                (current_paragraph.rstrip().endswith('.') and line[0].isupper())
            )
            
            if starts_new_paragraph and current_paragraph:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = line
            else:
                # Continue current paragraph
                if current_paragraph:
                    # Smart spacing - add space if needed
                    if current_paragraph.rstrip().endswith(('.', '!', '?', ':', ';')):
                        current_paragraph += " " + line
                    else:
                        current_paragraph += " " + line
                else:
                    current_paragraph = line
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
        
        # Step 4: Final text assembly and cleanup
        result = "\n\n".join(paragraphs)
        
        # Final cleanup passes
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Fix paragraph breaks
        result = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result)  # Fix sentence spacing
        
        # Fix common punctuation issues
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)
        result = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', result)
        
        return result.strip()
    
    def extract_pages_from_pdf(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Extract pages from PDF in batches (main story content only)
        """
        logger.info(f"Starting PDF page extraction from: {pdf_path}")
        
        # Get file size for logging
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        logger.info(f"PDF file size: {file_size_mb:.1f}MB")
        
        # Try extraction methods in order of quality
        extraction_methods = [
            ("PDFMiner with layout analysis", self._extract_pages_with_pdfminer_advanced),
            ("PDFMiner simple extraction", self._extract_pages_with_pdfminer_simple),
            ("PyMuPDF with batching", self._extract_pages_with_pymupdf_batched),
            ("PyMuPDF basic", self._extract_pages_with_pymupdf_basic),
        ]
        
        for method_name, extraction_func in extraction_methods:
            try:
                logger.info(f"Trying page extraction with: {method_name}")
                
                # Check memory before extraction
                memory_before = self._check_memory_usage()
                logger.info(f"Memory before extraction: {memory_before:.1f}MB")
                
                page_batches, pages_processed, story_start_page = extraction_func(pdf_path, progress_callback)
                
                # Check memory after extraction
                memory_after = self._check_memory_usage()
                logger.info(f"Memory after extraction: {memory_after:.1f}MB")
                
                if page_batches:
                    logger.info(f"Successfully created {len(page_batches)} page batches using {method_name}")
                    logger.info(f"Processed {pages_processed} pages starting from page {story_start_page}")
                    
                    return page_batches, pages_processed, story_start_page
                else:
                    logger.warning(f"No page batches created using {method_name}")
                    
            except Exception as e:
                logger.warning(f"{method_name} failed: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                # Force garbage collection on error
                gc.collect()
                continue
        
        logger.error("All extraction methods failed")
        return [], 0, 0
    
    def _extract_pages_with_pdfminer_advanced(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[List[Dict], int, int]:
        """Advanced PDFMiner extraction with page batching"""
        page_batches = []
        pages_processed = 0
        
        try:
            # Configure LAParams for better text extraction
            laparams = LAParams(
                line_margin=0.5,
                word_margin=0.1,
                char_margin=2.0,
                box_flow=0.5,
                detect_vertical=False,
            )
            
            # Get total pages and find story start
            with open(pdf_path, 'rb') as file:
                total_pages = len(list(PDFPage.get_pages(file)))
            
            story_start_page = self._find_story_start_page_pdfminer(pdf_path)
            logger.info(f"Processing {total_pages} pages starting from page {story_start_page}")
            
            # Extract text in page batches
            with open(pdf_path, 'rb') as file:
                resource_manager = PDFResourceManager()
                device = PDFPageAggregator(resource_manager, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, device)
                
                pages = list(PDFPage.get_pages(file))
                
                # Process pages in batches
                current_batch_text = ""
                current_batch_start = story_start_page
                batch_number = 1
                
                for page_num in range(story_start_page, total_pages):
                    try:
                        interpreter.process_page(pages[page_num])
                        layout = device.get_result()
                        page_text = self._extract_text_from_layout(layout)
                        
                        if page_text and len(page_text.strip()) > 20:
                            cleaned_page_text = self._advanced_page_cleaning(page_text)
                            if cleaned_page_text.strip():
                                current_batch_text += cleaned_page_text + "\n\n"
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
                        
                        # Report progress
                        if progress_callback and page_num % 5 == 0:
                            progress = (page_num - story_start_page) / (total_pages - story_start_page)
                            progress_callback(progress)
                        
                        # Memory management
                        if page_num % self.PROCESSING_BATCH_SIZE == 0:
                            self._check_memory_usage()
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error in PDFMiner advanced page extraction: {e}")
            raise
        
        logger.info(f"PDFMiner advanced extraction completed: {len(page_batches)} page batches, {pages_processed} pages processed")
        return page_batches, pages_processed, story_start_page
    
    def _extract_pages_with_pdfminer_simple(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[List[Dict], int, int]:
        """Simple PDFMiner extraction with page batching"""
        try:
            logger.info("Using PDFMiner simple extraction for page batching")
            
            # Extract all text at once
            full_text = extract_text(pdf_path)
            
            if not full_text or len(full_text.strip()) < 500:
                raise Exception("Insufficient text extracted with simple method")
            
            # Get page count and find story start
            with open(pdf_path, 'rb') as file:
                total_pages = len(list(PDFPage.get_pages(file)))
            
            story_start_page = max(5, total_pages // 10)  # Simple heuristic
            
            # Clean extracted text
            cleaned_text = self._clean_extracted_text_pdfminer(full_text)
            
            # Split text into page-like batches (approximate)
            page_batches = self._split_text_into_page_batches(cleaned_text, total_pages)
            
            logger.info(f"PDFMiner simple extraction completed: {len(page_batches)} page batches")
            return page_batches, total_pages - story_start_page, story_start_page
            
        except Exception as e:
            logger.error(f"Error in PDFMiner simple page extraction: {e}")
            raise
    
    def _extract_pages_with_pymupdf_batched(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[List[Dict], int, int]:
        """PyMuPDF extraction with page batching"""
        page_batches = []
        pages_processed = 0
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            story_start_page = self._find_story_start_page_enhanced(doc)
            
            logger.info(f"Processing {total_pages} pages starting from page {story_start_page}")
            
            # Process pages in batches
            current_batch_text = ""
            current_batch_start = story_start_page
            batch_number = 1
            
            for page_num in range(story_start_page, total_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = self._extract_page_text_enhanced(page)
                    
                    if page_text and len(page_text.strip()) > 20:
                        cleaned_page_text = self._advanced_page_cleaning(page_text)
                        if cleaned_page_text.strip():
                            current_batch_text += cleaned_page_text + "\n\n"
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
                    
                    # Free page memory
                    page = None
                    
                    # Report progress
                    if progress_callback and page_num % 10 == 0:
                        progress = (page_num - story_start_page) / (total_pages - story_start_page)
                        progress_callback(progress)
                    
                    # Memory management
                    if page_num % self.PROCESSING_BATCH_SIZE == 0:
                        self._check_memory_usage()
                    
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error in batched page extraction: {e}")
            raise
        
        logger.info(f"Batched page extraction completed: {len(page_batches)} page batches, {pages_processed} pages processed")
        return page_batches, pages_processed, story_start_page
    
    def _extract_pages_with_pymupdf_basic(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[List[Dict], int, int]:
        """Basic PyMuPDF extraction with page batching"""
        page_batches = []
        pages_processed = 0
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            story_start_page = self._find_story_start_page_enhanced(doc)
            
            # Process pages in batches
            current_batch_text = ""
            current_batch_start = story_start_page
            batch_number = 1
            
            for page_num in range(story_start_page, total_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    page_text = self._clean_page_text_basic(page_text)
                    
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
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error in basic page extraction: {e}")
            raise
        
        return page_batches, pages_processed, story_start_page
    
    def _find_story_start_page_pdfminer(self, pdf_path: str) -> int:
        """Find story start using PDFMiner with content analysis"""
        story_indicators = [
            r'chapter\s+(?:1|one|i)\b',
            r'prologue\b',
            r'part\s+(?:1|one|i)\b',
            r'once upon a time',
            r'it was\s+(?:a|the)',
            r'the story\s+(?:of|begins)',
            r'in the beginning',
            r'long ago',
            r'many years\s+(?:ago|before)'
        ]
        
        narrative_words = [
            'said', 'asked', 'replied', 'thought', 'felt', 'saw', 'heard', 
            'walked', 'ran', 'looked', 'smiled', 'laughed', 'cried', 'whispered',
            'suddenly', 'quietly', 'moment', 'time', 'day', 'night', 'morning',
            'character', 'person', 'man', 'woman', 'child', 'friend'
        ]
        
        candidate_pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pages = list(PDFPage.get_pages(file))
                max_pages_to_check = min(self.MAX_PAGES_TO_SCAN, len(pages))
                
                for page_num in range(max_pages_to_check):
                    try:
                        # Extract text from single page
                        page_text = ""
                        with open(pdf_path, 'rb') as file2:
                            resource_manager = PDFResourceManager()
                            device = PDFPageAggregator(resource_manager, laparams=LAParams())
                            interpreter = PDFPageInterpreter(resource_manager, device)
                            
                            pages2 = list(PDFPage.get_pages(file2))
                            if page_num < len(pages2):
                                interpreter.process_page(pages2[page_num])
                                layout = device.get_result()
                                page_text = self._extract_text_from_layout(layout).lower()
                        
                        if len(page_text) < 100:
                            continue
                        
                        page_score = 0
                        
                        # Check for story indicators
                        for indicator in story_indicators:
                            if re.search(indicator, page_text):
                                page_score += 50
                        
                        # Check narrative content quality
                        if len(page_text) > 1000:
                            narrative_count = sum(1 for word in narrative_words if f' {word} ' in page_text)
                            sentence_count = len(re.findall(r'[.!?]+', page_text))
                            dialog_count = page_text.count('"') + page_text.count("'")
                            
                            page_score += min(narrative_count * 2, 30)
                            page_score += min(sentence_count // 5, 20)
                            page_score += min(dialog_count // 4, 15)
                        
                        # Penalty for non-narrative content
                        non_narrative_patterns = [
                            r'table of contents', r'copyright', r'published by', r'isbn',
                            r'all rights reserved', r'acknowledgments', r'dedication',
                            r'about the author', r'index', r'bibliography', r'preface'
                        ]
                        
                        penalty = sum(25 for pattern in non_narrative_patterns if re.search(pattern, page_text))
                        page_score -= penalty
                        
                        if page_score > 0:
                            candidate_pages.append((page_num, page_score))
                            
                    except Exception as e:
                        logger.warning(f"Error analyzing page {page_num}: {e}")
                        continue
        
        except Exception as e:
            logger.warning(f"Error in story start detection: {e}")
        
        # Select best candidate
        if candidate_pages:
            candidate_pages.sort(key=lambda x: x[1], reverse=True)
            best_page = candidate_pages[0][0]
            logger.info(f"Selected page {best_page} as story start")
            return best_page
        
        # Fallback
        with open(pdf_path, 'rb') as file:
            total_pages = len(list(PDFPage.get_pages(file)))
            default_start = max(5, total_pages // 10)
            logger.info(f"Using default start page: {default_start}")
            return default_start
    
    def _find_story_start_page_enhanced(self, doc) -> int:
        """Enhanced story start detection with better heuristics"""
        story_indicators = [
            r'chapter\s+(?:1|one|i)\b',
            r'prologue\b',
            r'part\s+(?:1|one|i)\b',
            r'once upon a time',
            r'it was\s+(?:a|the)',
            r'the story\s+(?:of|begins)',
            r'in the beginning',
            r'long ago',
            r'many years\s+(?:ago|before)'
        ]
        
        narrative_words = [
            'said', 'asked', 'replied', 'thought', 'felt', 'saw', 'heard', 
            'walked', 'ran', 'looked', 'smiled', 'laughed', 'cried', 'whispered',
            'suddenly', 'quietly', 'moment', 'time', 'day', 'night', 'morning',
            'character', 'person', 'man', 'woman', 'child', 'friend'
        ]
        
        candidate_pages = []
        max_pages_to_check = min(self.MAX_PAGES_TO_SCAN, len(doc))
        
        for page_num in range(max_pages_to_check):
            try:
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                if len(text) < 100:
                    continue
                
                page_score = 0
                
                # Check for explicit story indicators
                for indicator in story_indicators:
                    if re.search(indicator, text):
                        page_score += 50
                
                # Check narrative content quality
                if len(text) > 1000:
                    narrative_count = sum(1 for word in narrative_words if f' {word} ' in text)
                    sentence_count = len(re.findall(r'[.!?]+', text))
                    dialog_count = text.count('"') + text.count("'")
                    
                    page_score += min(narrative_count * 2, 30)
                    page_score += min(sentence_count // 5, 20)
                    page_score += min(dialog_count // 4, 15)
                
                # Penalty for non-narrative content
                non_narrative_patterns = [
                    r'table of contents', r'copyright', r'published by', r'isbn',
                    r'all rights reserved', r'acknowledgments', r'dedication',
                    r'about the author', r'index', r'bibliography', r'preface'
                ]
                
                penalty = sum(25 for pattern in non_narrative_patterns if re.search(pattern, text))
                page_score -= penalty
                
                if page_score > 0:
                    candidate_pages.append((page_num, page_score))
                        
            except Exception as e:
                logger.warning(f"Error analyzing page {page_num}: {e}")
                continue
        
        # Select the best candidate page
        if candidate_pages:
            candidate_pages.sort(key=lambda x: x[1], reverse=True)
            best_page = candidate_pages[0][0]
            logger.info(f"Selected page {best_page} as story start")
            return best_page
        
        # Fallback
        default_start = max(5, len(doc) // 10)
        logger.info(f"No clear story start found, using default: page {default_start}")
        return default_start
    
    def _extract_text_from_layout(self, layout) -> str:
        """Extract text from PDFMiner layout object with structure preservation"""
        text_elements = []
        
        def extract_text_recursive(obj):
            if hasattr(obj, '_objs'):
                for child in obj._objs:
                    extract_text_recursive(child)
            elif hasattr(obj, 'get_text'):
                text = obj.get_text().strip()
                if text and len(text) > 1:
                    y_pos = getattr(obj, 'y1', 0) if hasattr(obj, 'y1') else 0
                    x_pos = getattr(obj, 'x0', 0) if hasattr(obj, 'x0') else 0
                    text_elements.append((y_pos, x_pos, text))
        
        extract_text_recursive(layout)
        
        # Sort by position (top to bottom, left to right)
        text_elements.sort(key=lambda x: (-x[0], x[1]))
        
        # Join text elements
        page_text = '\n'.join(element[2] for element in text_elements)
        
        return page_text
    
    def _extract_page_text_enhanced(self, page) -> str:
        """Enhanced page text extraction with multiple fallback methods"""
        extraction_methods = [
            lambda p: self._extract_text_blocks_enhanced(p),
            lambda p: self._extract_from_dict_enhanced(p),
            lambda p: p.get_text("text")
        ]
        
        for method in extraction_methods:
            try:
                text = method(page)
                if text and len(text.strip()) > 50:
                    return text
            except Exception as e:
                logger.debug(f"Page extraction method failed: {e}")
                continue
        
        return ""
    
    def _extract_text_blocks_enhanced(self, page) -> str:
        """Enhanced text blocks extraction with better ordering"""
        try:
            blocks = page.get_text("blocks")
            if not blocks:
                return ""
            
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            text_parts = []
            for block in sorted_blocks:
                if len(block) >= 5 and isinstance(block[4], str) and block[4].strip():
                    block_text = block[4].strip()
                    if len(block_text) > 3:
                        text_parts.append(block_text)
            
            return "\n".join(text_parts)
        
        except Exception as e:
            logger.debug(f"Enhanced text blocks extraction failed: {e}")
            return ""
    
    def _extract_from_dict_enhanced(self, page) -> str:
        """Enhanced dictionary-based extraction with better structure preservation"""
        try:
            text_dict = page.get_text("dict")
            text_parts = []
            
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                
                block_lines = []
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        if "text" in span and span["text"].strip():
                            line_text += span["text"]
                    
                    if line_text.strip():
                        block_lines.append(line_text.strip())
                
                if block_lines:
                    block_text = " ".join(block_lines)
                    if len(block_text.strip()) > 3:
                        text_parts.append(block_text)
            
            return "\n".join(text_parts)
        
        except Exception as e:
            logger.debug(f"Enhanced dictionary extraction failed: {e}")
            return ""
    
    def _clean_extracted_text_pdfminer(self, text: str) -> str:
        """Clean text extracted by PDFMiner simple method"""
        if not text:
            return ""
        
        lines = text.split('\n')
        
        # Look for story start indicators
        story_start_patterns = [
            r'(?i)chapter\s+(?:1|one|i)\b',
            r'(?i)prologue\b',
            r'(?i)part\s+(?:1|one|i)\b'
        ]
        
        story_start_line = 0
        for i, line in enumerate(lines):
            for pattern in story_start_patterns:
                if re.search(pattern, line):
                    story_start_line = max(0, i - 2)
                    logger.info(f"Found story start at line {i}: {line[:50]}...")
                    break
            if story_start_line > 0:
                break
        
        if story_start_line == 0:
            for i, line in enumerate(lines[:len(lines)//5]):
                if len(line) > 100 and line.count(' ') > 15:
                    story_start_line = i
                    break
            
            if story_start_line == 0:
                story_start_line = len(lines) // 10
        
        story_text = '\n'.join(lines[story_start_line:])
        return self._advanced_page_cleaning(story_text)
    
    def _split_text_into_page_batches(self, text: str, total_pages: int) -> List[Dict]:
        """Split text into page-like batches"""
        words = text.split()
        total_words = len(words)
        
        # Estimate words per page batch
        estimated_words_per_batch = total_words // max(1, (total_pages // self.PAGES_PER_BATCH))
        
        page_batches = []
        current_batch = []
        batch_number = 1
        
        for word in words:
            current_batch.append(word)
            
            if len(current_batch) >= estimated_words_per_batch:
                batch_text = ' '.join(current_batch)
                word_count = len(current_batch)
                
                if word_count >= self.MIN_BATCH_WORDS:
                    start_page = (batch_number - 1) * self.PAGES_PER_BATCH
                    end_page = min(start_page + self.PAGES_PER_BATCH - 1, total_pages - 1)
                    
                    page_batches.append({
                        'batch_number': batch_number,
                        'page_range': f"{start_page}-{end_page}",
                        'text': batch_text,
                        'word_count': word_count,
                        'pages_in_batch': self.PAGES_PER_BATCH,
                        'cleaned': False
                    })
                    batch_number += 1
                
                current_batch = []
        
        # Handle remaining words
        if current_batch:
            batch_text = ' '.join(current_batch)
            word_count = len(current_batch)
            
            if word_count >= self.MIN_BATCH_WORDS:
                start_page = (batch_number - 1) * self.PAGES_PER_BATCH
                end_page = total_pages - 1
                
                page_batches.append({
                    'batch_number': batch_number,
                    'page_range': f"{start_page}-{end_page}",
                    'text': batch_text,
                    'word_count': word_count,
                    'pages_in_batch': end_page - start_page + 1,
                    'cleaned': False
                })
        
        return page_batches
    
    def _advanced_page_cleaning(self, text: str) -> str:
        """Enhanced page text cleaning with better OCR error handling"""
        if not text:
            return ""
        
        # Fix OCR-specific issues first
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:"])', r'\1', text)
        text = re.sub(r'([.,!?;:"])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Fix quote marks
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'['']', "'", text)
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Remove headers, footers, page numbers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Skip page artifacts
            skip_patterns = [
                r'^\d+
            ,
                r'^Page\s+\d+',
                r'^Chapter\s+\d+\s*
            ,
                r'.*copyright.*',
                r'.*all rights reserved.*',
                r'^www\.',
                r'.*\.com.*',
                r'^\s*[-=_]{3,}\s*
            ,
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            if len(line) < 4 and not line.startswith(('"', "'")):
                continue
            
            alpha_ratio = sum(c.isalpha() for c in line) / len(line) if len(line) > 0 else 0
            if alpha_ratio < 0.3:
                continue
            
            cleaned_lines.append(line)
        
        # Smart paragraph reconstruction
        paragraphs = []
        current_paragraph = ""
        
        for line in cleaned_lines:
            starts_new_paragraph = False
            
            if current_paragraph:
                prev_line_ended_sentence = current_paragraph.rstrip().endswith(('.', '!', '?'))
                current_line_starts_capital = line[0].isupper()
                current_line_starts_quote = line.startswith(('"', "'", '"', "'"))
                paragraph_is_long = len(current_paragraph) > 200
                
                starts_new_paragraph = (
                    (prev_line_ended_sentence and current_line_starts_capital and paragraph_is_long) or
                    current_line_starts_quote or
                    re.match(r'^(Chapter|CHAPTER|Section|\d+\.)', line)
                )
            
            if starts_new_paragraph and current_paragraph:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = line
            else:
                if current_paragraph:
                    if current_paragraph.rstrip().endswith(('.', '!', '?', ':', ';')):
                        current_paragraph += " " + line
                    else:
                        current_paragraph += " " + line
                else:
                    current_paragraph = line
        
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
        
        # Final text assembly and cleanup
        result = "\n\n".join(paragraphs)
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        result = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result)
        
        return result.strip()
    
    def _clean_page_text_basic(self, text: str) -> str:
        """Basic page text cleaning (fallback)"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+\s*, ?', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page\s+\d+.*', '', text, flags=re.MULTILINE)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s.,!?;:"\'-]', '', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()

# Initialize processor
try:
    processor = ImprovedEbookPageProcessor()
    logger.info("✅ ImprovedEbookPageProcessor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ImprovedEbookPageProcessor: {e}")
    processor = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Ebook Page Processor API - Enhanced Version",
        "version": "2.0.0",
        "status": "✅ Running" if processor else "⚠️ Limited functionality",
        "ai_enabled": bool(processor and processor.replicate_client),
        "endpoints": {
            "POST /process-pdf": "Upload PDF and extract page batches (10 pages each)",
            "POST /process-pdf-with-ai": "Upload PDF, extract and clean page batches with AI (parallel processing)",
            "GET /health": "Health check"
        },
        "page_processing": {
            "batch_size": "10 pages per batch",
            "processing": "Main story content only, skips front/back matter",
            "formats": "0-10, 10-20, 20-30, etc.",
            "memory_optimized": "Batch processing for large PDFs",
            "ai_cleaning": "Parallel AI cleaning with GPT-4o-mini"
        },
        "improvements": {
            "memory_management": "Optimized for large PDFs with batch processing",
            "text_extraction": "Enhanced multi-method extraction with better OCR handling",
            "page_detection": "Smart story start detection to skip non-content pages",
            "parallel_processing": "AI page cleaning runs in parallel for faster processing",
            "error_resilience": "Better handling of problematic pages and PDFs",
            "max_concurrent": "Control concurrent AI tasks (1-20, default: 5)",
            "fallback_cleaning": "Enhanced basic text cleaning if AI fails"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
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
        "version": "2.0.0",
        "page_batch_size": 10
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...)):
    """
    Extract page batches from PDF ebook (without AI cleaning) - Enhanced Version
    
    - **file**: PDF file to process (max 100MB)
    - Returns: Extracted page batches (10 pages each) with enhanced cleaning
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if len(contents) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
        
        logger.info(f"Processing PDF: {file.filename} ({file_size_mb:.1f}MB)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract page batches with progress tracking
            def progress_callback(progress):
                pass
            
            page_batches, pages_processed, story_start_page = processor.extract_pages_from_pdf(tmp_path, progress_callback)
            
            if not page_batches:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable page batches found in the PDF. The file may be image-based, corrupted, or contain no extractable text content."
                )
            
            # Calculate stats
            total_words = sum(batch['word_count'] for batch in page_batches)
            reading_time = total_words / 200  # 200 words per minute
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            # Convert to response format
            batch_responses = [
                PageBatchResponse(**batch) for batch in page_batches
            ]
            
            logger.info(f"Processing completed: {len(page_batches)} page batches, {total_words} words, {processing_time:.2f}s")
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed successfully with enhanced page batch extraction",
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
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        logger.error(f"Exception details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-pdf-with-ai", response_model=ProcessingResponse)
async def process_pdf_with_ai(file: UploadFile = File(...), max_concurrent: int = 5):
    """
    Extract and clean page batches from PDF ebook using AI (Enhanced Parallel Processing)
    
    - **file**: PDF file to process (max 100MB)
    - **max_concurrent**: Maximum number of page batches to process simultaneously (default: 5)
    - Returns: Extracted page batches with AI-powered text cleaning
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size
        contents = await file.read()
        file_size_mb = len(contents) / (1024 * 1024)
        
        if len(contents) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 100MB")
        
        # Validate max_concurrent parameter
        if max_concurrent < 1 or max_concurrent > 20:
            raise HTTPException(status_code=400, detail="max_concurrent must be between 1 and 20")
        
        logger.info(f"Processing PDF with AI: {file.filename} ({file_size_mb:.1f}MB)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract page batches
            logger.info("Extracting page batches from PDF using enhanced methods...")
            
            def progress_callback(progress):
                pass
            
            page_batches, pages_processed, story_start_page = processor.extract_pages_from_pdf(tmp_path, progress_callback)
            
            if not page_batches:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable page batches found in the PDF. The file may be image-based, corrupted, or contain no extractable text content."
                )
            
            logger.info(f"Extracted {len(page_batches)} page batches. Starting parallel AI cleaning with {max_concurrent} concurrent tasks...")
            
            # Clean page batches with AI in parallel
            cleaned_batches = await processor.clean_all_page_batches_parallel(page_batches, max_concurrent)
            
            # Calculate stats
            total_words = sum(batch['word_count'] for batch in cleaned_batches)
            reading_time = total_words / 200  # 200 words per minute
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            # Count successful AI cleanings
            ai_cleaned_count = sum(1 for batch in cleaned_batches if batch['cleaned'])
            
            logger.info(f"Processing completed in {processing_time:.2f}s. AI cleaned: {ai_cleaned_count}/{len(cleaned_batches)} page batches")
            
            # Convert to response format
            batch_responses = [
                PageBatchResponse(**batch) for batch in cleaned_batches
            ]
            
            ai_status = "with enhanced AI cleaning" if processor.replicate_client else "with enhanced basic cleaning (AI unavailable)"
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed {ai_status}. {ai_cleaned_count}/{len(cleaned_batches)} page batches successfully cleaned with AI",
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
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF with AI: {e}")
        logger.error(f"Exception details: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/download-json")
async def download_page_batches_json(page_batches_data: Dict):
    """
    Download processed page batches as JSON file
    
    - **page_batches_data**: The page batches data to download
    - Returns: JSON file download
    """
    try:
        # Create JSON file with metadata
        json_data = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'processor': 'PDF Ebook Page Processor API v2.0',
                'total_page_batches': len(page_batches_data.get('page_batches', [])),
                'batch_size': '10 pages per batch',
                'version': '2.0.0'
            },
            'data': page_batches_data
        }
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(json_data, tmp_file, indent=2)
            json_path = tmp_file.name
        
        return FileResponse(
            json_path,
            media_type='application/json',
            filename='processed_page_batches.json'
        )
        
    except Exception as e:
        logger.error(f"Error creating JSON download: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway provides this)
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting Enhanced PDF Ebook Page Processor API v2.0...")
    print("🔑 Checking Replicate configuration...")
    
    # Check Replicate token
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("✅ Replicate API token found - AI cleaning enabled")
    else:
        print("⚠️ Replicate API token not found - AI cleaning disabled, enhanced basic processing available")
    
    print("📄 Page-Based Processing Features:")
    print("  • Page batch extraction (10 pages per batch: 0-10, 10-20, etc.)")
    print("  • Smart story start detection to skip front matter")
    print("  • Memory-optimized processing for large PDFs (up to 100MB)")
    print("  • Advanced OCR error correction and text cleaning")
    print("  • Batch processing for better memory management") 
    print("  • Enhanced duplicate content removal")
    print("  • Better error handling and recovery")
    print("🧹 AI-powered text cleaning with GPT-4o-mini (when available)")
    print("📚 PDF ebook support: Enhanced page extraction + AI cleaning + batch formatting")
    print(f"🌐 FastAPI server starting on port {port}...")
    
    uvicorn.run(app, host="0.0.0.0", port=port)