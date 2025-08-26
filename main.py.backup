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
        # Initialize Replicate client with environment variable
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN environment variable is required")
        self.replicate_client = replicate.Client(api_token=api_token)
    
    async def clean_chapter_text_with_ai(self, chapter_text: str, chapter_title: str) -> str:
        """Clean chapter text using GPT-4o-mini via Replicate"""
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
                        'cleaned': True,
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
        """Extract chapters from PDF as clean, formatted text"""
        try:
            # Try with PyMuPDF first (better text extraction)
            doc = fitz.open(pdf_path)
            
            # Skip first few pages (table of contents, copyright, etc.)
            start_page = self._find_story_start_page(doc)
            
            # Extract all text first
            full_text = ""
            for page_num in range(start_page, len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # Basic cleaning
                page_text = self._clean_page_text(page_text)
                
                if page_text.strip():
                    full_text += page_text + "\n\n"
            
            doc.close()
            
            # Split into chapters
            chapters = self._split_into_chapters(full_text)
            
            return chapters
            
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}, trying PyPDF2...")
            
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    
                    # Skip first few pages
                    start_page = max(3, len(pdf_reader.pages) // 20)
                    
                    for page_num in range(start_page, len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        # Basic cleaning
                        page_text = self._clean_page_text(page_text)
                        
                        if page_text.strip():
                            full_text += page_text + "\n\n"
                    
                    # Split into chapters
                    chapters = self._split_into_chapters(full_text)
                    
                    return chapters
                    
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
                return []
    
    def _find_story_start_page(self, doc) -> int:
        """Find where the actual story starts by looking for common patterns"""
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
                        return page_num
                
                # If page has substantial narrative content (not just TOC/copyright)
                if len(text) > 500 and self._is_narrative_text(text):
                    return page_num
                    
            except Exception as e:
                logger.warning(f"Error checking page {page_num}: {e}")
                continue
        
        # Default: skip first 5 pages
        return min(5, len(doc) // 4)
    
    def _is_narrative_text(self, text: str) -> bool:
        """Check if text appears to be narrative content"""
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
        
        if not chapter_breaks:
            # No chapters found, split by large gaps or treat as single chapter
            sections = re.split(r'\n\n\n+', text)
            sections = [section.strip() for section in sections if len(section.strip()) > 500]
            
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
        
        return chapters

# Initialize processor
processor = EbookProcessor()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "PDF Ebook Chapter Processor API",
        "version": "1.0.0",
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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(file: UploadFile = File(...)):
    """
    Extract chapters from PDF ebook (without AI cleaning)
    
    - **file**: PDF file to process (max 20MB)
    - Returns: Extracted chapters with basic cleaning
    """
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
            
            return ProcessingResponse(
                success=True,
                message=f"PDF processed with parallel AI cleaning. {ai_cleaned_count}/{len(cleaned_chapters)} chapters successfully cleaned with AI",
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
    
    # Verify Replicate token is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("❌ Replicate API token not found!")
        print("Please set REPLICATE_API_TOKEN environment variable")
        exit(1)
    
    print("🚀 Starting PDF Ebook Chapter Processor API...")
    print("🔑 Using Replicate for AI text cleaning")
    print("🧹 AI-powered text cleaning with GPT-4o-mini")
    print("📚 PDF ebook support: Text extraction + AI cleaning + chapter formatting")
    print(f"🌐 FastAPI server starting on port {port}...")
    
    uvicorn.run(app, host="0.0.0.0", port=port)