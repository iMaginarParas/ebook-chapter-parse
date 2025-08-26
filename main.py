import os
import logging
import tempfile
import json
import re
import asyncio
import gc
from datetime import datetime
from typing import List, Dict, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import time

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

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None

# Initialize FastAPI app
app = FastAPI(
    title="PDF Ebook Chapter Processor",
    description="Extract and clean chapters from PDF ebooks using AI - Railway Optimized",
    version="1.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EbookProcessor:
    def __init__(self):
        self.replicate_client = None
        self._init_replicate_client()
        self.processing_lock = asyncio.Semaphore(2)  # Reduced for Railway
        
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

    async def clean_chapter_text_with_ai(self, chapter_text: str, chapter_title: str, retries: int = 1) -> str:
        """Clean chapter text using GPT-4o-mini via Replicate with Railway optimization"""
        if not self.replicate_client:
            logger.warning(f"Replicate client not available for {chapter_title}, using basic cleaning")
            return self._basic_text_cleaning(chapter_text)
        
        for attempt in range(retries + 1):
            try:
                logger.info(f"Cleaning text for {chapter_title} using AI (attempt {attempt + 1})...")
                
                # More aggressive truncation for Railway
                max_chars = 6000  # Reduced for Railway
                if len(chapter_text) > max_chars:
                    chapter_text = chapter_text[:max_chars] + "..."
                    logger.warning(f"Truncated chapter {chapter_title} to {max_chars} characters")
                
                cleaning_prompt = f"""Clean up this PDF text by removing OCR errors, fixing spacing, and improving readability. Keep the story content intact. Only return the cleaned text, no explanations.

Text to clean:
{chapter_text}

Cleaned text:"""

                # Reduced timeout for Railway
                try:
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        def sync_replicate_call():
                            result = ""
                            timeout_start = time.time()
                            for event in self.replicate_client.stream(
                                "openai/gpt-4o-mini",
                                input={
                                    "prompt": cleaning_prompt,
                                    "max_tokens": 2000,  # Reduced
                                    "temperature": 0.1,
                                    "top_p": 0.9
                                }
                            ):
                                # Shorter timeout for Railway
                                if time.time() - timeout_start > 30:  # 30 seconds
                                    logger.warning(f"AI cleaning timeout for {chapter_title}")
                                    break
                                result += str(event)
                            return result
                        
                        # Shorter timeout
                        cleaned_text = await asyncio.wait_for(
                            loop.run_in_executor(executor, sync_replicate_call),
                            timeout=35  # 35 second timeout
                        )
                    
                    cleaned_text = self._post_process_cleaned_text(cleaned_text)
                    
                    if len(cleaned_text.strip()) > 50:  # Validate output
                        logger.info(f"Successfully cleaned {chapter_title}")
                        return cleaned_text
                    else:
                        logger.warning(f"AI output too short for {chapter_title}, using basic cleaning")
                        return self._basic_text_cleaning(chapter_text)
                        
                except asyncio.TimeoutError:
                    logger.warning(f"AI cleaning timeout for {chapter_title} on attempt {attempt + 1}")
                    return self._basic_text_cleaning(chapter_text)
                    
                except Exception as ai_error:
                    logger.warning(f"AI cleaning failed for {chapter_title} on attempt {attempt + 1}: {ai_error}")
                    return self._basic_text_cleaning(chapter_text)
                    
            except Exception as e:
                logger.error(f"Error in AI text cleaning attempt {attempt + 1}: {e}")
                return self._basic_text_cleaning(chapter_text)

        return self._basic_text_cleaning(chapter_text)

    async def clean_all_chapters_parallel(self, chapters: List[Dict], max_concurrent: int = 2) -> List[Dict]:
        """Clean all chapters in parallel with Railway optimization"""
        if not chapters:
            return []
            
        # Strict limits for Railway
        max_concurrent = min(max_concurrent, 2)  
        logger.info(f"Starting parallel AI cleaning for {len(chapters)} chapters with max {max_concurrent} concurrent tasks")
        
        # Create semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def clean_single_chapter(chapter_index: int, chapter: Dict) -> tuple[int, Dict]:
            async with semaphore:
                try:
                    # Add memory management
                    gc.collect()
                    
                    logger.info(f"Starting AI cleaning for index {chapter_index}: {chapter['title']}")
                    
                    cleaned_text = await self.clean_chapter_text_with_ai(
                        chapter['text'], 
                        chapter['title']
                    )
                    
                    cleaned_chapter = {
                        'chapter_number': chapter['chapter_number'],
                        'title': chapter['title'],
                        'text': chapter['text'][:1000] + "..." if len(chapter['text']) > 1000 else chapter['text'],
                        'cleaned_text': cleaned_text,
                        'word_count': len(cleaned_text.split()),
                        'original_word_count': chapter['word_count'],
                        'cleaned': bool(self.replicate_client and len(cleaned_text) != len(self._basic_text_cleaning(chapter['text']))),
                        'improvement_ratio': len(cleaned_text.split()) / chapter['word_count'] if chapter['word_count'] > 0 else 1.0
                    }
                    
                    logger.info(f"Completed cleaning index {chapter_index}: {chapter['title']}")
                    return (chapter_index, cleaned_chapter)
                    
                except Exception as e:
                    logger.error(f"Error cleaning chapter index {chapter_index} '{chapter.get('title', 'Unknown')}': {e}")
                    # Return chapter with basic cleaning on error
                    fallback_chapter = {
                        'chapter_number': chapter.get('chapter_number', chapter_index + 1),
                        'title': chapter.get('title', f"Chapter {chapter_index + 1}"),
                        'text': chapter.get('text', '')[:1000] + "..." if len(chapter.get('text', '')) > 1000 else chapter.get('text', ''),
                        'cleaned_text': self._basic_text_cleaning(chapter.get('text', '')),
                        'word_count': len(chapter.get('text', '').split()),
                        'original_word_count': chapter.get('word_count', 0),
                        'cleaned': False,
                        'improvement_ratio': 1.0
                    }
                    return (chapter_index, fallback_chapter)
        
        # Create tasks with better error handling
        tasks = []
        for i, chapter in enumerate(chapters):
            if chapter and 'text' in chapter:  # Validate chapter data
                tasks.append(clean_single_chapter(i, chapter))
        
        if not tasks:
            logger.error("No valid chapters to process")
            return []
        
        # Run all tasks concurrently with shorter timeout for Railway
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120  # 2 minute total timeout for Railway
            )
        except asyncio.TimeoutError:
            logger.error("Parallel processing timeout")
            return []
        
        # Process results
        result_chapters = [None] * len(chapters)
        exceptions_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in task {i}: {result}")
                exceptions_count += 1
                # Add fallback chapter
                if i < len(chapters):
                    chapter = chapters[i]
                    result_chapters[i] = {
                        'chapter_number': chapter.get('chapter_number', i + 1),
                        'title': chapter.get('title', f"Chapter {i + 1}"),
                        'text': chapter.get('text', '')[:1000],
                        'cleaned_text': self._basic_text_cleaning(chapter.get('text', '')),
                        'word_count': len(chapter.get('text', '').split()),
                        'original_word_count': chapter.get('word_count', 0),
                        'cleaned': False,
                        'improvement_ratio': 1.0
                    }
            else:
                index, cleaned_chapter = result
                if 0 <= index < len(result_chapters):
                    result_chapters[index] = cleaned_chapter
        
        # Filter out None values
        final_chapters = [ch for ch in result_chapters if ch is not None]
        
        logger.info(f"Parallel cleaning completed: {len(final_chapters)} chapters, {exceptions_count} exceptions")
        
        # Force garbage collection
        gc.collect()
        
        return final_chapters

    def _post_process_cleaned_text(self, text: str) -> str:
        """Post-process AI cleaned text"""
        if not text:
            return ""
        
        text = text.strip()
        
        # Remove AI commentary
        text = re.sub(r'^(Here is the cleaned text:|Clean Text:|Cleaned version:|Here\'s the cleaned text:).*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^(Here is|Here\'s).*?cleaned.*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
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

    def _basic_text_cleaning(self, text: str) -> str:
        """Improved basic text cleaning"""
        if not text:
            return ""
            
        # Remove page numbers and artifacts
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page\s+\d+.*$', '', text, flags=re.MULTILINE)
        
        # Fix hyphenated words
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Fix quotes
        text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
        
        return text.strip()

    def extract_chapters_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Railway-optimized chapter extraction"""
        try:
            doc = fitz.open(pdf_path)
            start_page = self._find_story_start_page(doc)
            
            # Process pages in smaller chunks to avoid memory issues
            full_text = ""
            max_pages = min(80, len(doc) - start_page)  # Reduced for Railway
            
            logger.info(f"Extracting text from pages {start_page} to {start_page + max_pages}")
            
            for page_num in range(start_page, start_page + max_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Immediate cleanup to save memory
                    if page_text and len(page_text.strip()) > 20:
                        cleaned_page = self._clean_page_text(page_text)
                        if cleaned_page and len(full_text) < 300000:  # 300KB text limit for Railway
                            full_text += cleaned_page + "\n\n"
                    
                    # Free page memory immediately
                    page = None
                    
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue
            
            doc.close()
            doc = None  # Explicit cleanup
            
            if not full_text or len(full_text.strip()) < 1000:
                logger.error("Insufficient text extracted from PDF")
                return []
            
            # Split into chapters
            chapters = self._split_into_chapters_improved(full_text)
            
            # Cleanup
            full_text = None
            gc.collect()
            
            # Limit chapter count and size for Railway
            if len(chapters) > 12:
                chapters = chapters[:12]
                logger.warning(f"Limited to first 12 chapters for Railway deployment")
            
            # Limit individual chapter size
            for chapter in chapters:
                if len(chapter['text']) > 30000:  # 30KB per chapter max
                    chapter['text'] = chapter['text'][:30000] + "..."
                    chapter['word_count'] = len(chapter['text'].split())
            
            return chapters
            
        except Exception as e:
            logger.error(f"Error in PDF extraction: {e}")
            return []

    def _split_into_chapters_improved(self, text: str) -> List[Dict]:
        """Improved chapter splitting with better patterns"""
        chapters = []
        
        # Enhanced chapter patterns - more comprehensive
        chapter_patterns = [
            r'(?i)(?:^|\n)\s*(chapter\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|i+|ii|iii|iv|v|vi|vii|viii|ix|x)+)(?:\s*[:.]?\s*([^\n]{0,100}))?',
            r'(?i)(?:^|\n)\s*((?:part|book)\s+(?:\d+|one|two|three|four|five|i+|ii|iii|iv|v))(?:\s*[:.]?\s*([^\n]{0,100}))?',
            r'(?i)(?:^|\n)\s*(prologue|epilogue|preface|introduction|conclusion)(?:\s*[:.]?\s*([^\n]{0,100}))?',
            r'(?i)(?:^|\n)\s*(\d+)\s*[:.]?\s*([^\n]{10,100})(?=\n)',  # Numbered chapters with titles
        ]
        
        # Find all potential chapter markers
        chapter_breaks = []
        for pattern in chapter_patterns:
            for match in re.finditer(pattern, text):
                start_pos = match.start()
                chapter_indicator = match.group(1).strip()
                chapter_title = match.group(2).strip() if match.lastindex >= 2 and match.group(2) else ""
                
                # Create full title
                if chapter_title:
                    full_title = f"{chapter_indicator}: {chapter_title}"
                else:
                    full_title = chapter_indicator
                
                chapter_breaks.append((start_pos, full_title, chapter_indicator))
        
        # Remove duplicates and sort
        seen_positions = set()
        unique_breaks = []
        for pos, title, indicator in chapter_breaks:
            # Allow some tolerance for nearby positions
            nearby = any(abs(pos - seen_pos) < 50 for seen_pos in seen_positions)
            if not nearby:
                unique_breaks.append((pos, title, indicator))
                seen_positions.add(pos)
        
        unique_breaks.sort(key=lambda x: x[0])
        
        logger.info(f"Found {len(unique_breaks)} potential chapter markers")
        
        if len(unique_breaks) < 2:
            logger.warning("Insufficient chapter markers found, trying content-based splitting")
            return self._split_by_content(text)
        
        # Extract chapters
        for i, (start_pos, title, indicator) in enumerate(unique_breaks):
            # Find end position
            if i + 1 < len(unique_breaks):
                end_pos = unique_breaks[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract chapter text
            chapter_text = text[start_pos:end_pos].strip()
            
            # Validate chapter length
            if len(chapter_text) > 500:  # Minimum meaningful chapter length
                # Clean the chapter text
                chapter_text = self._clean_chapter_content(chapter_text)
                
                if len(chapter_text) > 200:  # After cleaning, still substantial
                    chapters.append({
                        'chapter_number': i + 1,
                        'title': title[:100],  # Limit title length
                        'text': chapter_text,
                        'word_count': len(chapter_text.split()),
                        'cleaned': False
                    })
        
        logger.info(f"Successfully extracted {len(chapters)} chapters")
        return chapters

    def _split_by_content(self, text: str) -> List[Dict]:
        """Fallback method to split text by content patterns"""
        # Split by large paragraph breaks
        sections = re.split(r'\n\s*\n\s*\n+', text)
        sections = [s.strip() for s in sections if len(s.strip()) > 1000]  # Increased minimum
        
        chapters = []
        for i, section in enumerate(sections[:15]):  # Limit to 15 sections for Railway
            # Try to find a good title from the beginning
            lines = section.split('\n')
            title = f"Section {i + 1}"
            
            # Look for potential titles in first few lines
            for line in lines[:3]:
                line = line.strip()
                if 20 <= len(line) <= 100 and not line.endswith('.'):
                    title = f"Section {i + 1}: {line}"
                    break
            
            chapters.append({
                'chapter_number': i + 1,
                'title': title,
                'text': section,
                'word_count': len(section.split()),
                'cleaned': False
            })
        
        logger.info(f"Created {len(chapters)} content sections")
        return chapters

    def _clean_chapter_content(self, text: str) -> str:
        """Clean individual chapter content"""
        if not text:
            return ""
        
        # Remove the chapter header line if it's repeated in content
        lines = text.split('\n')
        if len(lines) > 1:
            # Remove first line if it looks like a chapter header
            first_line = lines[0].strip()
            if re.match(r'(?i)(chapter|part|prologue|epilogue)', first_line):
                text = '\n'.join(lines[1:])
        
        # Basic cleaning
        text = self._basic_text_cleaning(text)
        
        return text

    def _find_story_start_page(self, doc) -> int:
        """Find where the actual story content begins"""
        story_indicators = [
            r'(?i)chapter\s+(?:1|one|i)\b',
            r'(?i)prologue',
            r'(?i)part\s+(?:1|one|i)\b',
        ]
        
        content_indicators = ['said', 'asked', 'looked', 'walked', 'thought', 'felt']
        
        for page_num in range(min(25, len(doc))):  # Reduced for Railway
            try:
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                # Look for story start indicators
                for indicator in story_indicators:
                    if re.search(indicator, text):
                        logger.info(f"Found story indicator on page {page_num}")
                        return page_num
                
                # Check for narrative content
                if len(text) > 800:  # Substantial content
                    content_score = sum(1 for word in content_indicators if word in text)
                    
                    # Avoid table of contents, copyright pages, etc.
                    avoid_patterns = [
                        r'table of contents', r'copyright', r'isbn', r'published',
                        r'all rights reserved', r'acknowledgments', r'contents'
                    ]
                    
                    has_avoid_pattern = any(re.search(pattern, text) for pattern in avoid_patterns)
                    
                    if content_score >= 2 and not has_avoid_pattern:
                        logger.info(f"Found narrative content on page {page_num}")
                        return page_num
                        
            except Exception as e:
                logger.warning(f"Error checking page {page_num}: {e}")
                continue
        
        # Default: skip first few pages
        default_start = min(5, len(doc) // 15)
        logger.info(f"Using default start page: {default_start}")
        return default_start

    def _clean_page_text(self, text: str) -> str:
        """Enhanced page text cleaning"""
        if not text:
            return ""
        
        # Remove obvious page artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, page numbers, headers
            if (not line or 
                re.match(r'^\d+$', line) or
                re.match(r'^page\s+\d+', line, re.I) or
                len(line) < 3):
                continue
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Fix common issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\s+', ' ', text)  # Fix spacing
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Fix paragraph breaks
        
        return text.strip()

# Initialize processor with better error handling
try:
    processor = EbookProcessor()
    logger.info("✅ EbookProcessor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize EbookProcessor: {e}")
    processor = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Ebook Chapter Processor API - Railway Optimized",
        "version": "1.2.0",
        "status": "✅ Running" if processor else "⚠️ Limited functionality",
        "ai_enabled": bool(processor and processor.replicate_client),
        "optimizations": [
            "Railway memory optimization",
            "Reduced timeout handling", 
            "Limited concurrent processing",
            "Enhanced error handling",
            "Memory cleanup and garbage collection",
            "Smaller chunk processing"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check for Railway"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "processor_available": bool(processor),
        "ai_enabled": bool(processor and processor.replicate_client),
        "environment": os.environ.get('RAILWAY_ENVIRONMENT', 'development'),
        "memory_info": {
            "garbage_collected": True
        }
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...)):
    """Process PDF with Railway optimization"""
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    async with processor.processing_lock:
        start_time = datetime.now()
        tmp_path = None
        
        try:
            # Validate file
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
            # Check file size - Railway has memory limits
            contents = await file.read()
            if len(contents) > 12 * 1024 * 1024:  # 12MB limit for Railway
                raise HTTPException(status_code=413, detail="File too large. Maximum size is 12MB")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            # Extract chapters with timeout
            logger.info(f"Extracting chapters from {file.filename}")
            chapters = processor.extract_chapters_from_pdf(tmp_path)
            
            if not chapters:
                raise HTTPException(
                    status_code=422, 
                    detail="Could not extract readable chapters. The PDF may be image-based or have poor text quality."
                )
            
            # Limit chapters to prevent timeouts on Railway
            max_chapters = 8  # Reduced for Railway
            if len(chapters) > max_chapters:
                logger.warning(f"Limiting to first {max_chapters} chapters (found {len(chapters)})")
                chapters = chapters[:max_chapters]
            
            # Process chapters with timeout protection
            processing_timeout = 30  # Reduced timeout for Railway
            
            try:
                cleaned_chapters = await asyncio.wait_for(
                    processor.clean_all_chapters_parallel(chapters, max_concurrent=2),
                    timeout=processing_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("AI processing timeout, using basic cleaning")
                # Fallback to basic cleaning
                cleaned_chapters = []
                for i, chapter in enumerate(chapters):
                    cleaned_text = processor._basic_text_cleaning(chapter['text'])
                    cleaned_chapters.append({
                        'chapter_number': chapter['chapter_number'],
                        'title': chapter['title'],
                        'text': chapter['text'][:1000] + "..." if len(chapter['text']) > 1000 else chapter['text'],
                        'cleaned_text': cleaned_text,
                        'word_count': len(cleaned_text.split()),
                        'original_word_count': chapter['word_count'],
                        'cleaned': False,
                        'improvement_ratio': 1.0
                    })
            
            if not cleaned_chapters:
                raise HTTPException(status_code=500, detail="Failed to process any chapters")
            
            # Calculate stats
            total_words = sum(chapter.get('word_count', 0) for chapter in cleaned_chapters)
            reading_time = total_words / 200 if total_words > 0 else 0
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Count AI cleaned chapters
            ai_cleaned_count = sum(1 for chapter in cleaned_chapters if chapter.get('cleaned', False))
            
            # Convert to response format
            chapter_responses = [ChapterResponse(**chapter) for chapter in cleaned_chapters]
            
            ai_status = "with AI enhancement" if processor.replicate_client else "with basic cleaning only"
            
            return ProcessingResponse(
                success=True,
                message=f"Successfully processed {len(cleaned_chapters)} chapters {ai_status}. AI enhanced: {ai_cleaned_count}/{len(cleaned_chapters)}",
                file_name=file.filename,
                total_chapters=len(cleaned_chapters),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                chapters=chapter_responses,
                processing_time_seconds=processing_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:200]}")
        finally:
            # Clean up
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            # Force garbage collection
            gc.collect()

@app.post("/download-json")
async def download_chapters_json(chapters_data: Dict):
    """Download processed chapters as JSON file"""
    try:
        # Create JSON file with metadata
        json_data = {
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'processor': 'PDF Ebook Chapter Processor API v1.2 - Railway Optimized',
                'total_chapters': len(chapters_data.get('chapters', [])),
            },
            'data': chapters_data
        }
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(json_data, tmp_file, indent=2, ensure_ascii=False)
            json_path = tmp_file.name
        
        return FileResponse(
            json_path,
            media_type='application/json',
            filename='processed_chapters.json'
        )
        
    except Exception as e:
        logger.error(f"Error creating JSON download: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating download: {str(e)}")

@app.post("/process-basic")
async def process_pdf_basic_only(file: UploadFile = File(...)):
    """Process PDF with basic cleaning only (no AI) - Faster for Railway"""
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    tmp_path = None
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size
        contents = await file.read()
        if len(contents) > 15 * 1024 * 1024:  # 15MB limit
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Extract chapters
        logger.info(f"Extracting chapters from {file.filename} (basic mode)")
        chapters = processor.extract_chapters_from_pdf(tmp_path)
        
        if not chapters:
            raise HTTPException(
                status_code=422, 
                detail="Could not extract readable chapters. The PDF may be image-based or have poor text quality."
            )
        
        # Apply basic cleaning only
        cleaned_chapters = []
        for chapter in chapters:
            cleaned_text = processor._basic_text_cleaning(chapter['text'])
            cleaned_chapters.append({
                'chapter_number': chapter['chapter_number'],
                'title': chapter['title'],
                'text': chapter['text'][:1000] + "..." if len(chapter['text']) > 1000 else chapter['text'],
                'cleaned_text': cleaned_text,
                'word_count': len(cleaned_text.split()),
                'original_word_count': chapter['word_count'],
                'cleaned': False,  # No AI cleaning
                'improvement_ratio': len(cleaned_text.split()) / chapter['word_count'] if chapter['word_count'] > 0 else 1.0
            })
        
        # Calculate stats
        total_words = sum(chapter['word_count'] for chapter in cleaned_chapters)
        reading_time = total_words / 200 if total_words > 0 else 0
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to response format
        chapter_responses = [ChapterResponse(**chapter) for chapter in cleaned_chapters]
        
        return ProcessingResponse(
            success=True,
            message=f"Successfully processed {len(cleaned_chapters)} chapters with basic cleaning (no AI)",
            file_name=file.filename,
            total_chapters=len(cleaned_chapters),
            total_words=total_words,
            estimated_reading_time_minutes=reading_time,
            chapters=chapter_responses,
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in basic processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)[:200]}")
    finally:
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        gc.collect()

# Error handlers
@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    return JSONResponse(
        status_code=413,
        content={
            "success": False,
            "error": "File too large",
            "details": "Maximum file size is 12MB for Railway deployment"
        }
    )

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "details": "Please try again with a smaller PDF or contact support"
        }
    )

# Replace the startup section at the end of your main.py with this:
if __name__ == "__main__":
    import uvicorn
    
    # Railway port detection - try multiple environment variables
    port = int(
        os.environ.get("PORT") or 
        os.environ.get("PORT0") or 
        os.environ.get("HTTP_PORT") or 
        "8000"
    )
    
    print(f"🚀 Starting Railway-Optimized PDF Ebook Chapter Processor API...")
    print(f"🌐 Server starting on 0.0.0.0:{port}...")
    print(f"🔧 Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'development')}")
    print(f"💾 Memory optimizations: enabled")
    print(f"⏱️  Timeouts: reduced for Railway")
    print(f"🔍 Available env vars: PORT={os.environ.get('PORT')}, PORT0={os.environ.get('PORT0')}")
    
    # Check Replicate token
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("✅ Replicate API token found - AI cleaning enabled")
    else:
        print("⚠️ Replicate API token not found - basic processing only")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0", 
            port=port,
            workers=1,
            timeout_keep_alive=120,  # Increased for Railway
            timeout_graceful_shutdown=30,
            log_level="info",
            access_log=True,
            # Remove Railway-specific settings that might cause issues
        )
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        raise