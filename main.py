import os
import logging
import tempfile
import json
import re
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

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
    description="Extract and clean chapters from PDF ebooks using AI",
    version="1.0.0"
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
        # Initialize Replicate client only when needed for AI processing
        self.replicate_client = None
        self._init_replicate_client()
    
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
    
    async def clean_chapter_text_with_ai(self, chapter_text: str, chapter_title: str) -> str:
        """Clean chapter text using GPT-4o-mini via Replicate"""
        # Check if Replicate client is available
        if not self.replicate_client:
            logger.warning(f"Replicate client not available for {chapter_title}, using basic cleaning")
            return self._basic_text_cleaning(chapter_text)
        
        try:
            logger.info(f"Cleaning text for {chapter_title} using AI...")
            
            # Create a detailed prompt for text cleaning
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
{chapter_text}

Clean Text:"""

            # Call GPT-4o-mini for text cleaning using asyncio to run in thread pool
            cleaned_text = ""
            try:
                # Run the synchronous replicate call in a thread pool
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
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
        text = re.sub(r'^(Here is the cleaned text:|Clean Text:|Cleaned version:).*?\n', '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Ensure proper paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix quotation marks
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        
        # Ensure sentences end properly
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove excessive spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _basic_text_cleaning(self, text: str) -> str:
        """Fallback basic text cleaning if AI fails"""
        # Remove page numbers and headers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page\s+\d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'RABBIT-HOLE\.\s*\d*', '', text)
        text = re.sub(r'DOWN THE\s*', '', text)
        text = re.sub(r'CHAPTER [IVX]+\.?\s*', '', text)
        
        # Fix broken words
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        
        # Fix quotes
        text = re.sub(r'"\s*([^"]*?)\s*"', r'"\1"', text)
        
        return text.strip()
    
    def extract_chapters_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract chapters from PDF with improved text extraction"""
        # Try multiple extraction methods in order of quality
        extraction_methods = [
            ("PyMuPDF with layout preservation", self._extract_with_pymupdf_advanced),
            ("PyMuPDF basic", self._extract_with_pymupdf_basic),
            ("PyPDF2", self._extract_with_pypdf2)
        ]
        
        for method_name, extraction_func in extraction_methods:
            try:
                logger.info(f"Trying text extraction with: {method_name}")
                full_text = extraction_func(pdf_path)
                
                if full_text and len(full_text.strip()) > 500:
                    logger.info(f"✅ Successfully extracted {len(full_text)} characters using {method_name}")
                    
                    # Split into chapters
                    chapters = self._split_into_chapters(full_text)
                    
                    if chapters:
                        logger.info(f"✅ Created {len(chapters)} chapters")
                        return chapters
                    else:
                        logger.warning(f"⚠️ No chapters found using {method_name}")
                else:
                    logger.warning(f"⚠️ Insufficient text extracted using {method_name}")
                    
            except Exception as e:
                logger.warning(f"❌ {method_name} failed: {e}")
                continue
        
        logger.error("❌ All extraction methods failed")
        return []
    
    def _extract_with_pymupdf_advanced(self, pdf_path: str) -> str:
        """Advanced PyMuPDF extraction with layout preservation"""
        doc = fitz.open(pdf_path)
        start_page = self._find_story_start_page(doc)
        full_text = ""
        
        for page_num in range(start_page, len(doc)):
            page = doc.load_page(page_num)
            
            # Try different extraction methods for better quality
            methods_to_try = [
                # Method 1: Text with layout info
                lambda p: p.get_text("text"),
                # Method 2: Text blocks (preserves reading order better)
                lambda p: self._extract_text_blocks(p),
                # Method 3: Dictionary method (most detailed)
                lambda p: self._extract_from_dict(p)
            ]
            
            page_text = ""
            for method in methods_to_try:
                try:
                    page_text = method(page)
                    if page_text and len(page_text.strip()) > 50:
                        break
                except:
                    continue
            
            if not page_text:
                logger.warning(f"No text extracted from page {page_num}")
                continue
            
            # Advanced cleaning for this page
            page_text = self._advanced_page_cleaning(page_text)
            
            if page_text.strip():
                full_text += page_text + "\n\n"
        
        doc.close()
        return full_text
    
    def _extract_text_blocks(self, page) -> str:
        """Extract text using text blocks method"""
        blocks = page.get_text("blocks")
        text_parts = []
        
        for block in blocks:
            if len(block) >= 5 and block[4].strip():  # block[4] contains the text
                text_parts.append(block[4].strip())
        
        return "\n".join(text_parts)
    
    def _extract_from_dict(self, page) -> str:
        """Extract text using dictionary method for better structure"""
        text_dict = page.get_text("dict")
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line.get("spans", []):
                        if "text" in span:
                            line_text += span["text"]
                    if line_text.strip():
                        text_parts.append(line_text.strip())
        
        return "\n".join(text_parts)
    
    def _extract_with_pymupdf_basic(self, pdf_path: str) -> str:
        """Basic PyMuPDF extraction (fallback)"""
        doc = fitz.open(pdf_path)
        start_page = self._find_story_start_page(doc)
        full_text = ""
        
        for page_num in range(start_page, len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            page_text = self._clean_page_text(page_text)
            
            if page_text.strip():
                full_text += page_text + "\n\n"
        
        doc.close()
        return full_text
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """PyPDF2 extraction (last resort)"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            
            # Skip first few pages
            start_page = max(3, len(pdf_reader.pages) // 20)
            
            for page_num in range(start_page, len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                page_text = self._clean_page_text(page_text)
                
                if page_text.strip():
                    full_text += page_text + "\n\n"
        
        return full_text
    
    def _advanced_page_cleaning(self, text: str) -> str:
        """Advanced page text cleaning for better quality"""
        if not text:
            return ""
        
        # Step 1: Fix common OCR issues
        # Fix broken words across lines (hyphenation)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*([a-zA-Z])', r'\1 \2', text)
        
        # Fix quote marks
        text = re.sub(r'"\s+', '"', text)
        text = re.sub(r'\s+"', '"', text)
        text = re.sub(r"'\s+", "'", text)
        text = re.sub(r"\s+'", "'", text)
        
        # Step 2: Remove headers, footers, page numbers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip page numbers (standalone numbers)
            if re.match(r'^\d+$', line):
                continue
            
            # Skip common headers/footers
            if re.match(r'^(Page\s+\d+|Chapter\s+\d+\s*$|\d+\s*$)', line, re.IGNORECASE):
                continue
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                continue
            
            # Skip lines with mostly special characters
            if len(re.sub(r'[a-zA-Z0-9\s]', '', line)) > len(line) * 0.5:
                continue
            
            cleaned_lines.append(line)
        
        # Step 3: Reconstruct paragraphs
        reconstructed = []
        current_paragraph = ""
        
        for line in cleaned_lines:
            # Check if this line starts a new paragraph
            if (line[0].isupper() and len(current_paragraph) > 50) or \
               re.match(r'^(Chapter|CHAPTER|\d+\.)', line) or \
               (current_paragraph and line[0] in '"\''):
                
                if current_paragraph:
                    reconstructed.append(current_paragraph.strip())
                current_paragraph = line
            else:
                # Continue current paragraph
                if current_paragraph:
                    # Add space if the previous line doesn't end with punctuation
                    if not current_paragraph.rstrip().endswith(('.', '!', '?', ';', ':')):
                        current_paragraph += " " + line
                    else:
                        current_paragraph += " " + line
                else:
                    current_paragraph = line
        
        # Add the last paragraph
        if current_paragraph:
            reconstructed.append(current_paragraph.strip())
        
        # Step 4: Final cleaning
        result = "\n\n".join(reconstructed)
        
        # Remove excessive whitespace
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        # Fix sentence spacing
        result = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', result)
        
        return result.strip()
    
    def _find_story_start_page(self, doc) -> int:
        """Improved story start detection"""
        story_indicators = [
            r'chapter\s+(?:1|one|i)\b',
            r'prologue',
            r'part\s+(?:1|one|i)\b',
            r'once upon a time',
            r'it was a',
            r'the story',
            r'in the beginning',
            r'long ago',
            r'many years'
        ]
        
        # Content quality indicators
        narrative_words = [
            'said', 'asked', 'replied', 'thought', 'felt', 'saw', 'heard', 
            'walked', 'ran', 'looked', 'smiled', 'laughed', 'cried', 'whispered'
        ]
        
        for page_num in range(min(25, len(doc))):  # Check first 25 pages
            try:
                page = doc.load_page(page_num)
                text = page.get_text().lower()
                
                # Look for story indicators
                for indicator in story_indicators:
                    if re.search(indicator, text):
                        logger.info(f"Found story start indicator '{indicator}' on page {page_num}")
                        return page_num
                
                # Check content quality
                if len(text) > 1000:  # Substantial content
                    # Count narrative indicators
                    narrative_count = sum(1 for word in narrative_words if word in text)
                    sentence_count = len(re.findall(r'[.!?]+', text))
                    
                    # Check for non-narrative patterns
                    non_narrative_patterns = [
                        r'table of contents', r'copyright', r'published by', r'isbn',
                        r'all rights reserved', r'acknowledgments', r'dedication',
                        r'about the author', r'index', r'bibliography'
                    ]
                    
                    is_non_narrative = any(re.search(pattern, text) for pattern in non_narrative_patterns)
                    
                    # If it has good narrative content and isn't clearly non-narrative
                    if narrative_count >= 3 and sentence_count > 20 and not is_non_narrative:
                        logger.info(f"Found quality narrative content on page {page_num}")
                        return page_num
                        
            except Exception as e:
                logger.warning(f"Error checking page {page_num}: {e}")
                continue
        
        # Default: skip first 10% of pages or 5 pages, whichever is larger
        default_start = max(5, len(doc) // 10)
        logger.info(f"Using default start page: {default_start}")
        return default_start
    
    def _clean_page_text(self, text: str) -> str:
        """Clean and normalize page text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Page\s+\d+.*$', '', text, flags=re.MULTILINE)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s.,!?;:"\'-]', '', text)
        
        # Fix common spacing issues
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def _split_into_chapters(self, text: str) -> List[Dict]:
        """Split text into chapters with clean formatting"""
        chapters = []
        
        # Look for chapter markers
        chapter_patterns = [
            r'(?i)(chapter\s+\d+)',
            r'(?i)(chapter\s+\w+)',
            r'(?i)(part\s+\d+)',
            r'(?i)(part\s+\w+)',
            r'(?i)(prologue)',
            r'(?i)(epilogue)'
        ]
        
        # Find all chapter markers
        chapter_breaks = []
        for pattern in chapter_patterns:
            for match in re.finditer(pattern, text):
                chapter_breaks.append((match.start(), match.group().strip()))
        
        # Sort by position
        chapter_breaks.sort(key=lambda x: x[0])
        
        logger.info(f"Found {len(chapter_breaks)} chapter markers")
        
        if not chapter_breaks:
            # No chapters found, split by large gaps or treat as single chapter
            sections = re.split(r'\n\n\n+', text)
            sections = [section.strip() for section in sections if len(section.strip()) > 500]
            
            logger.info(f"No chapter markers found, splitting into {len(sections)} sections")
            
            for i, section in enumerate(sections):
                chapters.append({
                    'chapter_number': i + 1,
                    'title': f"Section {i + 1}",
                    'text': section,
                    'word_count': len(section.split()),
                    'cleaned': False  # Mark as not cleaned yet
                })
        else:
            # Extract chapters based on markers
            for i, (start_pos, title) in enumerate(chapter_breaks):
                # Determine end position
                if i + 1 < len(chapter_breaks):
                    end_pos = chapter_breaks[i + 1][0]
                else:
                    end_pos = len(text)
                
                # Extract chapter text
                chapter_text = text[start_pos:end_pos].strip()
                
                # Clean and format the chapter
                if len(chapter_text) > 200:  # Only include substantial chapters
                    chapters.append({
                        'chapter_number': i + 1,
                        'title': title,
                        'text': chapter_text,
                        'word_count': len(chapter_text.split()),
                        'cleaned': False  # Mark as not cleaned yet
                    })
        
        logger.info(f"Successfully created {len(chapters)} chapters")
        return chapters

# Initialize processor - this should work even without Replicate token
try:
    processor = EbookProcessor()
    logger.info("✅ EbookProcessor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize EbookProcessor: {e}")
    # Create a fallback processor that only does basic processing
    processor = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Ebook Chapter Processor API",
        "version": "1.0.0",
        "status": "✅ Running" if processor else "⚠️ Limited functionality",
        "ai_enabled": bool(processor and processor.replicate_client),
        "endpoints": {
            "POST /process-pdf": "Upload PDF and extract chapters",
            "POST /process-pdf-with-ai": "Upload PDF, extract and clean chapters with AI (parallel processing)",
            "GET /health": "Health check"
        },
        "features": {
            "parallel_processing": "AI chapter cleaning runs in parallel for faster processing",
            "max_concurrent": "Control concurrent AI tasks (1-20, default: 5)",
            "fallback_cleaning": "Basic text cleaning if AI fails",
            "async_processing": "Non-blocking async operations"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "processor_available": bool(processor),
        "ai_enabled": bool(processor and processor.replicate_client)
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...)):
    """
    Extract chapters from PDF ebook (without AI cleaning)
    
    - **file**: PDF file to process (max 20MB)
    - Returns: Extracted chapters with basic cleaning
    """
    if not processor:
        raise HTTPException(status_code=500, detail="Processor not available")
    
    start_time = datetime.now()
    
    try:
        # Validate file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size (20MB limit)
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract chapters
            chapters = processor.extract_chapters_from_pdf(tmp_path)
            
            if not chapters:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable chapters found in the PDF. Please ensure the PDF contains text content."
                )
            
            # Calculate stats
            total_words = sum(chapter['word_count'] for chapter in chapters)
            reading_time = total_words / 200  # 200 words per minute
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to response format
            chapter_responses = [
                ChapterResponse(**chapter) for chapter in chapters
            ]
            
            return ProcessingResponse(
                success=True,
                message="PDF processed successfully",
                file_name=file.filename,
                total_chapters=len(chapters),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                chapters=chapter_responses,
                processing_time_seconds=processing_time
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-pdf-with-ai", response_model=ProcessingResponse)
async def process_pdf_with_ai(file: UploadFile = File(...), max_concurrent: int = 5):
    """
    Extract and clean chapters from PDF ebook using AI (Parallel Processing)
    
    - **file**: PDF file to process (max 20MB)
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
        
        # Check file size (20MB limit)
        contents = await file.read()
        if len(contents) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 20MB")
        
        # Validate max_concurrent parameter
        if max_concurrent < 1 or max_concurrent > 20:
            raise HTTPException(status_code=400, detail="max_concurrent must be between 1 and 20")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Extract chapters
            logger.info("Extracting chapters from PDF...")
            chapters = processor.extract_chapters_from_pdf(tmp_path)
            
            if not chapters:
                raise HTTPException(
                    status_code=422, 
                    detail="No readable chapters found in the PDF. Please ensure the PDF contains text content."
                )
            
            logger.info(f"Extracted {len(chapters)} chapters. Starting parallel AI cleaning with {max_concurrent} concurrent tasks...")
            
            # Clean chapters with AI in parallel
            cleaned_chapters = await processor.clean_all_chapters_parallel(chapters, max_concurrent)
            
            # Calculate stats
            total_words = sum(ch['word_count'] for ch in cleaned_chapters)
            reading_time = total_words / 200  # 200 words per minute
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Count successful AI cleanings
            ai_cleaned_count = sum(1 for ch in cleaned_chapters if ch['cleaned'])
            
            logger.info(f"Processing completed in {processing_time:.2f}s. AI cleaned: {ai_cleaned_count}/{len(cleaned_chapters)} chapters")
            
            # Convert to response format
            chapter_responses = [
                ChapterResponse(**chapter) for chapter in cleaned_chapters
            ]
            
            ai_status = "with AI" if processor.replicate_client else "with basic cleaning (AI unavailable)"
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed {ai_status}. {ai_cleaned_count}/{len(cleaned_chapters)} chapters successfully cleaned with AI",
                file_name=file.filename,
                total_chapters=len(cleaned_chapters),
                total_words=total_words,
                estimated_reading_time_minutes=reading_time,
                chapters=chapter_responses,
                processing_time_seconds=processing_time
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF with AI: {e}")
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
                'processor': 'PDF Ebook Chapter Processor API',
                'total_chapters': len(chapters_data.get('chapters', [])),
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
    
    print("🚀 Starting PDF Ebook Chapter Processor API...")
    print("🔑 Checking Replicate configuration...")
    
    # Check Replicate token
    if os.environ.get("REPLICATE_API_TOKEN"):
        print("✅ Replicate API token found - AI cleaning enabled")
    else:
        print("⚠️ Replicate API token not found - AI cleaning disabled, basic processing only")
    
    print("🧹 AI-powered text cleaning with GPT-4o-mini (when available)")
    print("📚 PDF ebook support: Text extraction + AI cleaning + chapter formatting")
    print(f"🌐 FastAPI server starting on port {port}...")
    
    uvicorn.run(app, host="0.0.0.0", port=port)