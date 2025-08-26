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

# Replicate for AI processing
import replicate

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChapterResponse(BaseModel):
    chapter_number: int
    title: str
    text: str
    cleaned_text: Optional[str] = None
    word_count: int
    original_word_count: Optional[int] = None
    cleaned: bool = False
    improvement_ratio: Optional[float] = None

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    file_name: str
    total_chapters: int
    total_words: int
    estimated_reading_time_minutes: float
    chapters: List[ChapterResponse]
    processing_time_seconds: float
    memory_usage_mb: Optional[float] = None
    pages_processed: Optional[int] = None

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="PDF Ebook Chapter Processor",
    description="Extract and clean chapters from PDF ebooks using AI - Optimized for large files",
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

class ImprovedEbookProcessor:
    def __init__(self):
        # Initialize Replicate client only when needed for AI processing
        self.replicate_client = None
        self._init_replicate_client()
        
        # Processing configuration
        self.BATCH_SIZE = 10  # Process pages in batches to manage memory
        self.MAX_MEMORY_MB = 512  # Maximum memory usage before garbage collection
        self.MIN_CHAPTER_WORDS = 100  # Minimum words for a valid chapter
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
    
    async def clean_chapter_text_with_ai(self, chapter_text: str, chapter_title: str) -> str:
        """Clean chapter text using GPT-4o-mini via Replicate"""
        # Check if Replicate client is available
        if not self.replicate_client:
            logger.warning(f"Replicate client not available for {chapter_title}, using basic cleaning")
            return self._basic_text_cleaning(chapter_text)
        
        try:
            logger.info(f"Cleaning text for {chapter_title} using AI...")
            
            # Truncate very long chapters to avoid token limits
            if len(chapter_text) > 12000:  # About 3000 tokens
                logger.info(f"Chapter too long ({len(chapter_text)} chars), truncating for AI processing")
                chapter_text = chapter_text[:12000] + "..."
            
            # Create a detailed prompt for text cleaning
            cleaning_prompt = f"""You are a professional text editor. Clean up this extracted PDF text by:

1. Remove OCR artifacts, page numbers, and headers/footers
2. Fix broken words and sentences from PDF extraction
3. Correct spacing and punctuation errors
4. Merge broken paragraphs properly while preserving paragraph breaks
5. Keep dialogue formatting intact with proper quotes
6. Remove any non-story content (page markers, copyright text)
7. Maintain the original story structure and narrative flow
8. Fix hyphenated words that broke across lines

IMPORTANT: Return ONLY the cleaned readable text. No explanations or meta-commentary.

Text to clean:
{chapter_text}

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
                
                logger.info(f"Successfully cleaned {chapter_title}")
                return cleaned_text
                
            except Exception as ai_error:
                logger.warning(f"AI cleaning failed for {chapter_title}: {ai_error}")
                # Fallback to basic cleaning
                return self._basic_text_cleaning(chapter_text)
                
        except Exception as e:
            logger.error(f"Error in AI text cleaning: {e}")
            return self._basic_text_cleaning(chapter_text)

    async def clean_all_chapters_parallel(self, chapters: List[Dict], max_concurrent: int = 5) -> List[Dict]:
        """Clean all chapters in parallel with concurrency limit"""
        logger.info(f"Starting parallel AI cleaning for {len(chapters)} chapters with max {max_concurrent} concurrent tasks")
        
        # Log input chapters for debugging
        chapter_titles = [ch.get('title', f'Chapter {ch.get("chapter_number", "?")}') for ch in chapters]
        logger.info(f"Input chapters: {chapter_titles}")
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def clean_single_chapter(chapter_index: int, chapter: Dict) -> tuple[int, Dict]:
            async with semaphore:
                try:
                    logger.info(f"Starting AI cleaning for index {chapter_index}: {chapter['title']}")
                    
                    # Clean the chapter text with AI
                    cleaned_text = await self.clean_chapter_text_with_ai(
                        chapter['text'], 
                        chapter['title']
                    )
                    
                    # Create cleaned chapter object
                    cleaned_chapter = {
                        'chapter_number': chapter['chapter_number'],
                        'title': chapter['title'],
                        'text': chapter['text'],  # Keep original
                        'cleaned_text': cleaned_text,  # Add cleaned version
                        'word_count': len(cleaned_text.split()),
                        'original_word_count': chapter['word_count'],
                        'cleaned': bool(self.replicate_client),  # True if AI was used
                        'improvement_ratio': len(cleaned_text.split()) / chapter['word_count'] if chapter['word_count'] > 0 else 1.0
                    }
                    
                    logger.info(f"Completed cleaning index {chapter_index}: {chapter['title']} (original: {chapter['word_count']} words -> cleaned: {cleaned_chapter['word_count']} words)")
                    return (chapter_index, cleaned_chapter)
                    
                except Exception as e:
                    logger.error(f"Error cleaning chapter index {chapter_index} '{chapter['title']}': {e}")
                    # Return chapter with basic cleaning on error
                    fallback_chapter = {
                        'chapter_number': chapter['chapter_number'],
                        'title': chapter['title'],
                        'text': chapter['text'],
                        'cleaned_text': self._basic_text_cleaning(chapter['text']),
                        'word_count': chapter['word_count'],
                        'original_word_count': chapter['word_count'],
                        'cleaned': False,  # Mark as failed AI cleaning
                        'improvement_ratio': 1.0
                    }
                    return (chapter_index, fallback_chapter)
        
        # Create tasks for all chapters with their indices
        tasks = [clean_single_chapter(i, chapter) for i, chapter in enumerate(chapters)]
        
        # Run all tasks concurrently and wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Sort results by original index to maintain order and handle exceptions
        result_chapters = [None] * len(chapters)
        exceptions_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in task {i}: {result}")
                exceptions_count += 1
                # Add chapter with basic cleaning as fallback
                chapter = chapters[i]
                result_chapters[i] = {
                    'chapter_number': chapter['chapter_number'],
                    'title': chapter['title'],
                    'text': chapter['text'],
                    'cleaned_text': self._basic_text_cleaning(chapter['text']),
                    'word_count': chapter['word_count'],
                    'original_word_count': chapter['word_count'],
                    'cleaned': False,
                    'improvement_ratio': 1.0
                }
            else:
                # result is a tuple of (index, cleaned_chapter)
                index, cleaned_chapter = result
                result_chapters[index] = cleaned_chapter
        
        # Filter out any None values (shouldn't happen, but safety check)
        final_chapters = [ch for ch in result_chapters if ch is not None]
        
        # Log output chapters for debugging
        output_titles = [ch.get('title', f'Chapter {ch.get("chapter_number", "?")}') for ch in final_chapters]
        
        logger.info(f"Parallel cleaning completed:")
        logger.info(f"  - Input chapters: {len(chapters)}")
        logger.info(f"  - Output chapters: {len(final_chapters)}")
        logger.info(f"  - Exceptions handled: {exceptions_count}")
        logger.info(f"  - Successfully AI cleaned: {sum(1 for ch in final_chapters if ch.get('cleaned', False))}")
        logger.info(f"Output chapters: {output_titles}")
        
        return final_chapters
    
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
            
            # Skip standalone page numbers
            if re.match(r'^\d+$', line):
                continue
            
            # Skip common headers/footers patterns
            # Skip obvious page artifacts
            skip_patterns = [
                r'^\d+$',                          # Standalone page numbers
                r'^Page\s+\d+',                    # "Page X" headers
                r'^Chapter\s+\d+\s*$',             # Standalone "Chapter X"
                r'^\d+\s*$',                       # Numbers only
                r'^[^\w]*$',                       # Only punctuation/symbols
                r'.*copyright.*',                   # Copyright lines
                r'.*all rights reserved.*',        # Rights lines
                r'^www\.',                         # Website URLs
                r'.*\.com.*',                      # More URLs
                r'^\s*[-=_]{3,}\s*$'              # Separator lines
            ] = [
                r'^Page\s+\d+',
                r'^Chapter\s+\d+\s*$',
                r'^\d+\s*$',
                r'^.*copyright.*$',
                r'^.*all rights reserved.*$',
                r'^.*\d{4}.*$'  # Years in headers
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
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
                # Line starts with chapter marker
                re.match(r'^(Chapter|CHAPTER|\d+\.)', line) or
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
    
    def extract_chapters_from_pdf(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Extract chapters from PDF with improved memory management and quality
        """
        logger.info(f"Starting PDF extraction from: {pdf_path}")
        
        # Get file size for logging
        file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        logger.info(f"PDF file size: {file_size_mb:.1f}MB")
        
        # Try multiple extraction methods in order of quality
        extraction_methods = [
            ("Advanced PyMuPDF with batching", self._extract_with_pymupdf_batched),
            ("PyMuPDF with layout preservation", self._extract_with_pymupdf_advanced),
            ("PyMuPDF basic", self._extract_with_pymupdf_basic),
            ("PyPDF2 fallback", self._extract_with_pypdf2)
        ]
        
        for method_name, extraction_func in extraction_methods:
            try:
                logger.info(f"Trying text extraction with: {method_name}")
                
                # Check memory before extraction
                memory_before = self._check_memory_usage()
                logger.info(f"Memory before extraction: {memory_before:.1f}MB")
                
                full_text, pages_processed = extraction_func(pdf_path, progress_callback)
                
                # Check memory after extraction
                memory_after = self._check_memory_usage()
                logger.info(f"Memory after extraction: {memory_after:.1f}MB")
                
                if full_text and len(full_text.strip()) > 500:
                    logger.info(f"✅ Successfully extracted {len(full_text)} characters using {method_name}")
                    logger.info(f"✅ Processed {pages_processed} pages")
                    
                    # Split into chapters
                    chapters = self._split_into_chapters_enhanced(full_text)
                    
                    if chapters:
                        logger.info(f"✅ Created {len(chapters)} chapters")
                        return chapters
                    else:
                        logger.warning(f"⚠️ No chapters found using {method_name}")
                else:
                    logger.warning(f"⚠️ Insufficient text extracted using {method_name}")
                    
            except Exception as e:
                logger.warning(f"❌ {method_name} failed: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")
                # Force garbage collection on error
                gc.collect()
                continue
        
        logger.error("❌ All extraction methods failed")
        return []
    
    def _extract_with_pymupdf_batched(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[str, int]:
        """
        Memory-efficient PyMuPDF extraction processing pages in batches
        """
        full_text = ""
        pages_processed = 0
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            start_page = self._find_story_start_page_enhanced(doc)
            
            logger.info(f"Processing {total_pages} pages starting from page {start_page} in batches of {self.BATCH_SIZE}")
            
            # Process pages in batches to manage memory
            for batch_start in range(start_page, total_pages, self.BATCH_SIZE):
                batch_end = min(batch_start + self.BATCH_SIZE, total_pages)
                batch_text = ""
                
                logger.info(f"Processing batch: pages {batch_start}-{batch_end}")
                
                for page_num in range(batch_start, batch_end):
                    try:
                        page = doc.load_page(page_num)
                        
                        # Try multiple extraction methods for this page
                        page_text = self._extract_page_text_enhanced(page)
                        
                        if page_text and len(page_text.strip()) > 20:
                            # Advanced cleaning for this page
                            cleaned_page_text = self._advanced_page_cleaning(page_text)
                            
                            if cleaned_page_text.strip():
                                batch_text += cleaned_page_text + "\n\n"
                                pages_processed += 1
                        
                        # Free page memory
                        page = None
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
                
                # Add batch text to full text
                full_text += batch_text
                
                # Report progress
                if progress_callback:
                    progress = (batch_end - start_page) / (total_pages - start_page)
                    progress_callback(progress)
                
                # Check memory usage after each batch
                memory_usage = self._check_memory_usage()
                logger.debug(f"Memory usage after batch {batch_start}-{batch_end}: {memory_usage:.1f}MB")
                
                # Force garbage collection between batches for large files
                if memory_usage > self.MAX_MEMORY_MB * 0.8:
                    gc.collect()
            
            doc.close()
            doc = None  # Explicitly delete reference
            
        except Exception as e:
            logger.error(f"Error in batched extraction: {e}")
            raise
        
        logger.info(f"Batched extraction completed: {pages_processed} pages processed")
        return full_text, pages_processed
    
    def _extract_page_text_enhanced(self, page) -> str:
        """
        Enhanced page text extraction with multiple fallback methods
        """
        extraction_methods = [
            # Method 1: Text blocks (usually best for layout)
            lambda p: self._extract_text_blocks_enhanced(p),
            # Method 2: Dictionary method (most detailed)
            lambda p: self._extract_from_dict_enhanced(p),
            # Method 3: Simple text extraction (fallback)
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
            
            # Sort blocks by y-coordinate (top to bottom), then x-coordinate (left to right)
            sorted_blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            
            text_parts = []
            for block in sorted_blocks:
                if len(block) >= 5 and isinstance(block[4], str) and block[4].strip():
                    # Clean the block text
                    block_text = block[4].strip()
                    # Skip very short blocks that are likely artifacts
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
                    # Join lines in this block
                    block_text = " ".join(block_lines)
                    if len(block_text.strip()) > 3:
                        text_parts.append(block_text)
            
            return "\n".join(text_parts)
        
        except Exception as e:
            logger.debug(f"Enhanced dictionary extraction failed: {e}")
            return ""
    
    def _extract_with_pymupdf_advanced(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[str, int]:
        """Advanced PyMuPDF extraction with layout preservation (fallback method)"""
        doc = fitz.open(pdf_path)
        start_page = self._find_story_start_page_enhanced(doc)
        full_text = ""
        pages_processed = 0
        
        total_pages = len(doc)
        
        for page_num in range(start_page, total_pages):
            try:
                page = doc.load_page(page_num)
                page_text = self._extract_page_text_enhanced(page)
                
                if page_text and len(page_text.strip()) > 20:
                    # Advanced cleaning for this page
                    page_text = self._advanced_page_cleaning(page_text)
                    
                    if page_text.strip():
                        full_text += page_text + "\n\n"
                        pages_processed += 1
                
                # Report progress
                if progress_callback and page_num % 10 == 0:
                    progress = (page_num - start_page) / (total_pages - start_page)
                    progress_callback(progress)
                
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        doc.close()
        return full_text, pages_processed
    
    def _extract_with_pymupdf_basic(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[str, int]:
        """Basic PyMuPDF extraction (simple fallback)"""
        doc = fitz.open(pdf_path)
        start_page = self._find_story_start_page_enhanced(doc)
        full_text = ""
        pages_processed = 0
        
        for page_num in range(start_page, len(doc)):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_text = self._clean_page_text_basic(page_text)
                
                if page_text.strip():
                    full_text += page_text + "\n\n"
                    pages_processed += 1
                    
            except Exception as e:
                logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        doc.close()
        return full_text, pages_processed
    
    def _extract_with_pypdf2(self, pdf_path: str, progress_callback: Optional[Callable] = None) -> Tuple[str, int]:
        """PyPDF2 extraction (last resort)"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            pages_processed = 0
            
            # Skip first few pages
            start_page = max(3, len(pdf_reader.pages) // 20)
            
            for page_num in range(start_page, len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    page_text = self._clean_page_text_basic(page_text)
                    
                    if page_text.strip():
                        full_text += page_text + "\n\n"
                        pages_processed += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue
        
        return full_text, pages_processed
    
    def _find_story_start_page_enhanced(self, doc) -> int:
        """Enhanced story start detection with better heuristics"""
        
        # First pass: Look for explicit story indicators
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
        
        # Content quality indicators
        narrative_words = [
            'said', 'asked', 'replied', 'thought', 'felt', 'saw', 'heard', 
            'walked', 'ran', 'looked', 'smiled', 'laughed', 'cried', 'whispered',
            'suddenly', 'quietly', 'moment', 'time', 'day', 'night', 'morning',
            'character', 'person', 'man', 'woman', 'child', 'friend'
        ]
        
        # Track potential start pages with scores
        candidate_pages = []
        max_pages_to_check = min(self.MAX_PAGES_TO_SCAN, len(doc))
        
        for page_num in range(max_pages_to_check):
            try:
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                if len(text) < 100:  # Skip pages with very little text
                    continue
                
                page_score = 0
                
                # Check for explicit story indicators (high weight)
                for indicator in story_indicators:
                    if re.search(indicator, text):
                        page_score += 50
                        logger.debug(f"Found story indicator '{indicator}' on page {page_num}")
                
                # Check narrative content quality
                if len(text) > 1000:  # Substantial content
                    # Count narrative words
                    narrative_count = sum(1 for word in narrative_words if f' {word} ' in text)
                    sentence_count = len(re.findall(r'[.!?]+', text))
                    dialog_count = text.count('"') + text.count("'")
                    
                    # Add points for narrative elements
                    page_score += min(narrative_count * 2, 30)  # Cap at 30 points
                    page_score += min(sentence_count // 5, 20)   # Points for sentences
                    page_score += min(dialog_count // 4, 15)     # Points for dialogue
                    
                    # Bonus for good sentence structure
                    if sentence_count > 20 and len(text) / sentence_count > 50:
                        page_score += 10
                
                # Penalty for non-narrative content
                non_narrative_patterns = [
                    r'table of contents', r'copyright', r'published by', r'isbn',
                    r'all rights reserved', r'acknowledgments', r'dedication',
                    r'about the author', r'index', r'bibliography', r'preface',
                    r'introduction', r'foreword', r'contents'
                ]
                
                penalty = 0
                for pattern in non_narrative_patterns:
                    if re.search(pattern, text):
                        penalty += 25
                        logger.debug(f"Found non-narrative pattern '{pattern}' on page {page_num}")
                
                page_score -= penalty
                
                # Penalty for very short paragraphs (likely TOC or similar)
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                if paragraphs:
                    avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs)
                    if avg_paragraph_length < 100:  # Very short paragraphs
                        page_score -= 15
                
                if page_score > 0:
                    candidate_pages.append((page_num, page_score))
                    logger.debug(f"Page {page_num} scored {page_score} points")
                        
            except Exception as e:
                logger.warning(f"Error analyzing page {page_num}: {e}")
                continue
        
        # Select the best candidate page
        if candidate_pages:
            candidate_pages.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            best_page = candidate_pages[0][0]
            best_score = candidate_pages[0][1]
            logger.info(f"Selected page {best_page} as story start (score: {best_score})")
            return best_page
        
        # Fallback: skip first 10% of pages or minimum 5 pages
        default_start = max(5, len(doc) // 10)
        logger.info(f"No clear story start found, using default: page {default_start}")
        return default_start
    
    def _advanced_page_cleaning(self, text: str) -> str:
        """Enhanced page text cleaning with better OCR error handling"""
        if not text:
            return ""
        
        # Step 1: Fix OCR-specific issues first
        # Fix broken words across lines (most common OCR issue)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Fix spacing around punctuation (OCR often gets this wrong)
        text = re.sub(r'\s+([.,!?;:"])', r'\1', text)
        text = re.sub(r'([.,!?;:"])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Fix quote marks (OCR often mangles these)
        text = re.sub(r'["""]', '"', text)  # Normalize fancy quotes
        text = re.sub(r'['']', "'", text)   # Normalize fancy apostrophes
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Fix common OCR character substitutions
        ocr_fixes = {
            r'\bl\b': 'I',         # lowercase l mistaken for I
            r'\bO\b': '0',         # O mistaken for zero in numbers
            r'rn\b': 'm',          # rn mistaken for m
            r'\bvv': 'w',          # vv mistaken for w
            r'fi': 'fi',           # ligature fixes
            r'fl': 'fl'
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Step 2: Remove headers, footers, page numbers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip obvious page artifacts
            skip_patterns = [
                r'^\d+$',                          # Standalone page numbers
                r'^Page\s+\d+',                    # "Page X" headers
                r'^Chapter\s+\d+\s*$',             # Standalone "Chapter X"
                r'^\d+\s*$',                       # Numbers only
                r'^[^\w]*$',                       # Only punctuation/symbols
                r'.*copyright.*',                   # Copyright lines
                r'.*all rights reserved.*',        # Rights lines
                r'^www\.',                         # Website URLs
                r'.*\.com.*',                      # More URLs
                r'^\s*[-=_]{3,}\s*$'              # Separator lines
            ]
            
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            # Skip very short lines that are likely artifacts (unless they're dialogue)
            if len(line) < 4 and not line.startswith(('"', "'")):
                continue
            
            # Skip lines with mostly non-alphabetic characters
            alpha_ratio = sum(c.isalpha() for c in line) / len(line)
            if alpha_ratio < 0.3:  # Less than 30% alphabetic
                continue
            
            cleaned_lines.append(line)
        
        # Step 3: Smart paragraph reconstruction
        paragraphs = []
        current_paragraph = ""
        
        for i, line in enumerate(cleaned_lines):
            # Determine if this line should start a new paragraph
            starts_new_paragraph = False
            
            if current_paragraph:
                # Check various indicators for paragraph breaks
                prev_line_ended_sentence = current_paragraph.rstrip().endswith(('.', '!', '?'))
                current_line_starts_capital = line[0].isupper()
                current_line_starts_quote = line.startswith(('"', "'", '"', "'"))
                paragraph_is_long = len(current_paragraph) > 200
                
                # New paragraph if:
                starts_new_paragraph = (
                    # Previous paragraph ended sentence and this starts with capital
                    (prev_line_ended_sentence and current_line_starts_capital and paragraph_is_long) or
                    # Line starts with dialogue
                    current_line_starts_quote or
                    # Line starts with chapter/section marker
                    re.match(r'^(Chapter|CHAPTER|Section|\d+\.)', line) or
                    # Line starts with paragraph marker
                    re.match(r'^\s*¶', line) or
                    # Significant indentation change (new paragraph)
                    (len(line) - len(line.lstrip()) > 4)
                )
            
            if starts_new_paragraph and current_paragraph:
                paragraphs.append(current_paragraph.strip())
                current_paragraph = line
            else:
                # Continue current paragraph
                if current_paragraph:
                    # Smart spacing - check if we need a space
                    last_char = current_paragraph.rstrip()[-1] if current_paragraph.rstrip() else ''
                    first_char = line[0] if line else ''
                    
                    # Add space unless there are specific punctuation rules
                    if last_char in '.!?:;':
                        current_paragraph += " " + line
                    elif last_char in '-—':
                        current_paragraph += line  # No space after dash
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
        result = re.sub(r'\s+', ' ', result)  # Normalize whitespace within lines
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Fix excessive line breaks
        result = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result)  # Fix sentence spacing
        
        # Fix dialogue formatting
        result = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', result)
        
        # Remove any remaining artifacts
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        return result.strip()
    
    def _clean_page_text_basic(self, text: str) -> str:
        """Basic page text cleaning (fallback)"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+\s*, '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page\s+\d+.*, '', text, flags=re.MULTILINE)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s.,!?;:"\'-]', '', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def _split_into_chapters_enhanced(self, text: str) -> List[Dict]:
        """Enhanced chapter splitting with better detection and validation"""
        chapters = []
        
        # Enhanced chapter detection patterns
        chapter_patterns = [
            r'(?i)(?:^|\n)\s*(chapter\s+\d+)',
            r'(?i)(?:^|\n)\s*(chapter\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))',
            r'(?i)(?:^|\n)\s*(chapter\s+[ivxlcdm]+)',  # Roman numerals
            r'(?i)(?:^|\n)\s*(part\s+\d+)',
            r'(?i)(?:^|\n)\s*(part\s+(?:one|two|three|four|five))',
            r'(?i)(?:^|\n)\s*(prologue)',
            r'(?i)(?:^|\n)\s*(epilogue)',
            r'(?i)(?:^|\n)\s*(\d+\.\s*[A-Z])',  # "1. Title" format
        ]
        
        # Find all potential chapter breaks
        chapter_breaks = []
        for pattern in chapter_patterns:
            for match in re.finditer(pattern, text):
                start_pos = match.start()
                title = match.group(1).strip()
                
                # Look for a title after the chapter marker
                remaining_text = text[match.end():match.end()+200]
                title_match = re.search(r'^[^\n]*', remaining_text)
                if title_match and len(title_match.group().strip()) > len(title):
                    extended_title = title + " " + title_match.group().strip()
                    if len(extended_title) < 100:  # Reasonable title length
                        title = extended_title
                
                chapter_breaks.append((start_pos, title, match.end()))
        
        # Sort by position and remove duplicates/overlaps
        chapter_breaks.sort(key=lambda x: x[0])
        filtered_breaks = []
        
        for i, (pos, title, end_pos) in enumerate(chapter_breaks):
            # Skip if too close to previous break (likely duplicate detection)
            if filtered_breaks and pos - filtered_breaks[-1][0] < 100:
                # Keep the one with the better title
                if len(title) > len(filtered_breaks[-1][1]):
                    filtered_breaks[-1] = (pos, title, end_pos)
                continue
            filtered_breaks.append((pos, title, end_pos))
        
        chapter_breaks = filtered_breaks
        logger.info(f"Found {len(chapter_breaks)} chapter markers after filtering")
        
        if not chapter_breaks:
            # No chapters found, try alternative splitting methods
            return self._split_by_content_analysis(text)
        
        # Extract chapters based on markers
        for i, (start_pos, title, title_end_pos) in enumerate(chapter_breaks):
            # Determine end position
            if i + 1 < len(chapter_breaks):
                end_pos = chapter_breaks[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract chapter text (starting after the title)
            chapter_text = text[title_end_pos:end_pos].strip()
            
            # Validate chapter quality
            word_count = len(chapter_text.split())
            
            # Only include substantial chapters
            if word_count >= self.MIN_CHAPTER_WORDS:
                # Clean up the title
                clean_title = re.sub(r'\s+', ' ', title).strip()
                clean_title = re.sub(r'^(chapter|part)\s+', '', clean_title, flags=re.IGNORECASE)
                
                chapters.append({
                    'chapter_number': i + 1,
                    'title': clean_title,
                    'text': chapter_text,
                    'word_count': word_count,
                    'cleaned': False  # Mark as not cleaned yet
                })
                
                logger.debug(f"Added chapter {i+1}: '{clean_title}' ({word_count} words)")
            else:
                logger.debug(f"Skipped short chapter: '{title}' ({word_count} words)")
        
        # Final validation - remove chapters that are too similar (duplicates)
        chapters = self._remove_duplicate_chapters(chapters)
        
        logger.info(f"Successfully created {len(chapters)} valid chapters")
        return chapters
    
    def _split_by_content_analysis(self, text: str) -> List[Dict]:
        """Fallback chapter splitting based on content analysis"""
        logger.info("No clear chapter markers found, analyzing content for natural breaks")
        
        # Split by large gaps or obvious breaks
        potential_breaks = []
        
        # Look for natural break patterns
        break_patterns = [
            r'\n\s*\*\s*\*\s*\*\s*\n',  # *** breaks
            r'\n\s*---+\s*\n',          # --- breaks  
            r'\n\s*===+\s*\n',          # === breaks
            r'\n\n\n+',                 # Multiple line breaks
        ]
        
        for pattern in break_patterns:
            for match in re.finditer(pattern, text):
                potential_breaks.append(match.start())
        
        # If no clear breaks, split by content length
        if not potential_breaks:
            # Split into roughly equal sections
            words = text.split()
            total_words = len(words)
            target_chapter_size = max(2000, total_words // 10)  # Aim for ~10 chapters or min 2000 words
            
            sections = []
            current_section = []
            
            for word in words:
                current_section.append(word)
                if len(current_section) >= target_chapter_size:
                    # Look for a sentence boundary to break
                    section_text = ' '.join(current_section)
                    last_sentence_end = max(
                        section_text.rfind('. '),
                        section_text.rfind('! '),
                        section_text.rfind('? ')
                    )
                    
                    if last_sentence_end > len(section_text) * 0.7:  # Break is in last 30%
                        break_point = last_sentence_end + 2
                        sections.append(section_text[:break_point])
                        remaining = section_text[break_point:]
                        current_section = remaining.split() if remaining.strip() else []
                    else:
                        # No good break point found, break at word boundary
                        sections.append(' '.join(current_section))
                        current_section = []
            
            # Add remaining content
            if current_section:
                sections.append(' '.join(current_section))
        
        else:
            # Use the breaks we found
            potential_breaks.sort()
            sections = []
            last_pos = 0
            
            for break_pos in potential_breaks:
                if break_pos - last_pos > 1000:  # Minimum section size
                    section = text[last_pos:break_pos].strip()
                    if section:
                        sections.append(section)
                    last_pos = break_pos
            
            # Add final section
            if last_pos < len(text):
                final_section = text[last_pos:].strip()
                if len(final_section) > 500:
                    sections.append(final_section)
        
        # Convert sections to chapters
        chapters = []
        for i, section in enumerate(sections):
            word_count = len(section.split())
            if word_count >= self.MIN_CHAPTER_WORDS:
                chapters.append({
                    'chapter_number': i + 1,
                    'title': f"Section {i + 1}",
                    'text': section,
                    'word_count': word_count,
                    'cleaned': False
                })
        
        logger.info(f"Created {len(chapters)} sections from content analysis")
        return chapters
    
    def _remove_duplicate_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """Remove duplicate chapters based on content similarity"""
        if len(chapters) <= 1:
            return chapters
        
        filtered_chapters = []
        
        for chapter in chapters:
            is_duplicate = False
            chapter_words = set(chapter['text'].lower().split()[:100])  # First 100 words
            
            for existing in filtered_chapters:
                existing_words = set(existing['text'].lower().split()[:100])
                
                # Calculate similarity (Jaccard similarity)
                intersection = len(chapter_words & existing_words)
                union = len(chapter_words | existing_words)
                similarity = intersection / union if union > 0 else 0
                
                if similarity > 0.8:  # 80% similar
                    logger.debug(f"Removing duplicate chapter: {chapter['title']} (similar to {existing['title']})")
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_chapters.append(chapter)
        
        return filtered_chapters

# Initialize processor - this should work even without Replicate token
try:
    processor = ImprovedEbookProcessor()
    logger.info("✅ ImprovedEbookProcessor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize ImprovedEbookProcessor: {e}")
    # Create a fallback processor that only does basic processing
    processor = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Ebook Chapter Processor API - Enhanced Version",
        "version": "2.0.0",
        "status": "✅ Running" if processor else "⚠️ Limited functionality",
        "ai_enabled": bool(processor and processor.replicate_client),
        "endpoints": {
            "POST /process-pdf": "Upload PDF and extract chapters",
            "POST /process-pdf-with-ai": "Upload PDF, extract and clean chapters with AI (parallel processing)",
            "GET /health": "Health check"
        },
        "improvements": {
            "memory_management": "Optimized for large PDFs with batch processing",
            "text_extraction": "Enhanced multi-method extraction with better OCR handling",
            "chapter_detection": "Improved chapter boundary detection and validation",
            "parallel_processing": "AI chapter cleaning runs in parallel for faster processing",
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
        "version": "2.0.0"
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...)):
    """
    Extract chapters from PDF ebook (without AI cleaning) - Enhanced Version
    
    - **file**: PDF file to process (max 100MB for improved version)
    - Returns: Extracted chapters with enhanced cleaning
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size (increased to 100MB for improved version)
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
            # Extract chapters with progress tracking
            pages_processed = 0
            
            def progress_callback(progress):
                nonlocal pages_processed
                # This could be extended to provide real-time updates
                pass
            
            chapters = processor.extract_chapters_from_pdf(tmp_path, progress_callback)
            
            if not chapters:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable chapters found in the PDF. The file may be image-based, corrupted, or contain no extractable text content."
                )
            
            # Calculate stats
            total_words = sum(chapter['word_count'] for chapter in chapters)
            reading_time = total_words / 200  # 200 words per minute
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            # Convert to response format
            chapter_responses = [
                ChapterResponse(**chapter) for chapter in chapters
            ]
            
            logger.info(f"Processing completed: {len(chapters)} chapters, {total_words} words, {processing_time:.2f}s")
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed successfully with enhanced extraction methods",
                file_name=file.filename,
                total_chapters=len(chapters),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                chapters=chapter_responses,
                processing_time_seconds=processing_time,
                memory_usage_mb=memory_usage,
                pages_processed=pages_processed
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
    Extract and clean chapters from PDF ebook using AI (Enhanced Parallel Processing)
    
    - **file**: PDF file to process (max 100MB)
    - **max_concurrent**: Maximum number of chapters to process simultaneously (default: 5)
    - Returns: Extracted chapters with AI-powered text cleaning
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size (100MB limit for enhanced version)
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
            # Extract chapters with enhanced methods
            logger.info("Extracting chapters from PDF using enhanced methods...")
            
            pages_processed = 0
            def progress_callback(progress):
                nonlocal pages_processed
                # Could be extended for WebSocket progress updates
                pass
            
            chapters = processor.extract_chapters_from_pdf(tmp_path, progress_callback)
            
            if not chapters:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable chapters found in the PDF. The file may be image-based, corrupted, or contain no extractable text content."
                )
            
            logger.info(f"Extracted {len(chapters)} chapters. Starting parallel AI cleaning with {max_concurrent} concurrent tasks...")
            
            # Clean chapters with AI in parallel
            cleaned_chapters = await processor.clean_all_chapters_parallel(chapters, max_concurrent)
            
            # Calculate stats
            total_words = sum(ch['word_count'] for ch in cleaned_chapters)
            reading_time = total_words / 200  # 200 words per minute
            processing_time = (datetime.now() - start_time).total_seconds()
            memory_usage = processor._check_memory_usage()
            
            # Count successful AI cleanings
            ai_cleaned_count = sum(1 for ch in cleaned_chapters if ch['cleaned'])
            
            logger.info(f"Processing completed in {processing_time:.2f}s. AI cleaned: {ai_cleaned_count}/{len(cleaned_chapters)} chapters")
            
            # Convert to response format
            chapter_responses = [
                ChapterResponse(**chapter) for chapter in cleaned_chapters
            ]
            
            ai_status = "with enhanced AI cleaning" if processor.replicate_client else "with enhanced basic cleaning (AI unavailable)"
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed {ai_status}. {ai_cleaned_count}/{len(cleaned_chapters)} chapters successfully cleaned with AI",
                file_name=file.filename,
                total_chapters=len(cleaned_chapters),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                chapters=chapter_responses,
                processing_time_seconds=processing_time,
                memory_usage_mb=memory_usage,
                pages_processed=pages_processed
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
async def download_chapters_json(chapters_data: Dict):
    """
    Download processed chapters as JSON file
    
    - **chapters_data**: The chapters data to download
    - Returns: JSON file download
    """
    try:
        # Create JSON file with metadata
        json_data = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'processor': 'PDF Ebook Chapter Processor API v2.0',
                'total_chapters': len(chapters_data.get('chapters', [])),
                'version': '2.0.0'
            },
            'data': chapters_data
        }
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(json_data, tmp_file, indent=2)
            json_path = tmp_file.name
        
        return FileResponse(
            json_path,
            media_type='application/json',
            filename='processed_chapters.json'
        )
        
    except Exception as e:
        logger.error(f"Error creating JSON download: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway provides this)
    port = int(os.environ.get("PORT", 8000))
    
    print("🚀 Starting Enhanced PDF Ebook Chapter Processor API v2.0...")
    print("🔑 Checking Replicate configuration...")
    
    # Check Replicate token
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("✅ Replicate API token found - AI cleaning enabled")
    else:
        print("⚠️ Replicate API token not found - AI cleaning disabled, enhanced basic processing available")
    
    print("📈 Enhanced Features:")
    print("  • Memory-optimized processing for large PDFs (up to 100MB)")
    print("  • Advanced OCR error correction and text cleaning")
    print("  • Improved chapter detection with content analysis")
    print("  • Batch processing for better memory management") 
    print("  • Enhanced duplicate chapter removal")
    print("  • Better error handling and recovery")
    print("🧹 AI-powered text cleaning with GPT-4o-mini (when available)")
    print("📚 PDF ebook support: Enhanced extraction + AI cleaning + chapter formatting")
    print(f"🌐 FastAPI server starting on port {port}...")
    
    uvicorn.run(app, host="0.0.0.0", port=port)