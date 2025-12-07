import os
import logging
import json
import uuid
import asyncio
from typing import List, Dict, Optional

import fitz  # PyMuPDF
import uvicorn
import httpx  # Required for sending webhooks
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat" 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DeepSeekScanner")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: Keep an in-memory DB if you still want to fetch results manually later
books_db = {}

# ============================================================================
# WEBHOOK HELPER
# ============================================================================
async def send_webhook(url: str, payload: dict):
    """
    Sends the processing result to the provided callback URL.
    """
    if not url:
        return

    logger.info(f"🚀 Sending webhook to {url}...")
    async with httpx.AsyncClient() as client:
        try:
            # We assume the client expects a POST request with a JSON body
            response = await client.post(url, json=payload, timeout=20.0)
            logger.info(f"✅ Webhook sent to {url}. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to send webhook to {url}: {e}")

# ============================================================================
# DEEPSEEK LOGIC
# ============================================================================
class DeepSeekAnalyzer:
    def __init__(self):
        if DEEPSEEK_API_KEY:
            self.client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
        else:
            self.client = None

    async def process_book(self, book_id: str, file_path: str, webhook_url: str):
        """
        Background task that processes the PDF and triggers a webhook upon completion.
        """
        result_payload = {
            "book_id": book_id,
            "status": "processing",
            "data": None,
            "error": None
        }

        # 1. Validation
        if not self.client:
            result_payload["status"] = "failed"
            result_payload["error"] = "Server Error: API Key missing."
            await send_webhook(webhook_url, result_payload)
            return
        
        try:
            logger.info(f"[{book_id}] 📖 Extracting text from PDF...")
            doc = fitz.open(file_path)
            full_text_map = [] 
            llm_input_lines = []
            
            # 2. Extract Text & Create Skeleton
            for i, page in enumerate(doc):
                text = page.get_text()
                full_text_map.append(text) 
                # Send page number + first 1000 chars to AI (Skeleton Strategy)
                clean_excerpt = text[:1000].replace('\n', ' ')
                llm_input_lines.append(f"PAGE_ID_{i}: {clean_excerpt}")

            doc.close()
            
            context_str = "\n".join(llm_input_lines)
            
            # Truncate if over safety limit (approx 500k chars)
            if len(context_str) > 500000:
                logger.warning(f"[{book_id}] ⚠️ Book is huge! Truncating...")
                context_str = context_str[:500000]

            # 3. Ask AI
            logger.info(f"[{book_id}] 🚀 Sending to DeepSeek V3...")
            chapters_metadata = await self.ask_deepseek(context_str)
            
            if not chapters_metadata:
                result_payload["status"] = "failed"
                result_payload["error"] = "AI returned no chapters. Is the book empty?"
                await send_webhook(webhook_url, result_payload)
                return

            logger.info(f"[{book_id}] ✅ Identified {len(chapters_metadata)} chapters.")

            # 4. Reconstruct Full Text based on Page Boundaries
            final_output = []
            for chap in chapters_metadata:
                start = chap.get("start_page", 0)
                end = chap.get("end_page", 0)
                
                # Retrieve Full Text locally
                chapter_text = ""
                # Clamp values to valid page ranges
                start = max(0, min(start, len(full_text_map)-1))
                end = max(0, min(end, len(full_text_map)-1))
                
                for p_idx in range(start, end + 1):
                    chapter_text += full_text_map[p_idx] + "\n"
                
                final_output.append({
                    "index": chap.get("index"),
                    "title": chap.get("title"),
                    "text": chapter_text
                })

            # Save to local memory (optional backup)
            books_db[book_id] = final_output

            # 5. Success Webhook
            result_payload["status"] = "completed"
            result_payload["data"] = final_output
            await send_webhook(webhook_url, result_payload)

        except Exception as e:
            logger.error(f"[{book_id}] DeepSeek Error: {str(e)}")
            result_payload["status"] = "failed"
            result_payload["error"] = str(e)
            await send_webhook(webhook_url, result_payload)

        finally:
            # 6. Cleanup
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"[{book_id}] 🧹 Temp file cleaned up.")

    async def ask_deepseek(self, context: str):
        prompt = f"""
        I have a book represented by Page IDs and text snippets.
        Identify the CHAPTERS or SECTIONS.

        Return a JSON List of objects:
        [
            {{ "index": 1, "title": "Chapter 1 Name", "start_page": 0, "end_page": 5 }},
            {{ "index": 2, "title": "Chapter 2 Name", "start_page": 6, "end_page": 10 }}
        ]

        Rules:
        1. Use the "PAGE_ID_X" markers to determine start_page and end_page.
        2. Cover the whole book. The end_page of Ch1 should be before start_page of Ch2.
        3. Return JSON ONLY. No markdown.

        BOOK CONTENT:
        {context}
        """

        response = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=4000,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
            # Handle cases where DeepSeek wraps json in a key like "chapters": [...]
            return data.get("chapters", data) if isinstance(data, dict) else data
        except json.JSONDecodeError:
            return None

analyzer = DeepSeekAnalyzer()

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_book(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    webhook_url: str = Form(...)  # Client must provide where to send results
):
    """
    Receives a file, starts processing in the background, and returns immediately.
    The result will be sent to 'webhook_url' via POST.
    """
    book_id = str(uuid.uuid4())
    file_path = f"temp_{book_id}.pdf"
    
    # Save file temporarily
    content = await file.read()
    with open(file_path, "wb") as f: 
        f.write(content)
    
    # Queue the heavy lifting
    background_tasks.add_task(analyzer.process_book, book_id, file_path, webhook_url)
    
    return {
        "message": "File accepted. Processing started in background.",
        "book_id": book_id,
        "callback_target": webhook_url
    }

# Optional: Fallback endpoint if webhook fails and you want to poll manually
@app.get("/book/{book_id}/chapters")
async def get_chapters_manual(book_id: str):
    if book_id in books_db:
        return {"status": "completed", "data": books_db[book_id]}
    return {"status": "not_found_or_processing"}

if __name__ == "__main__":
    # --- DYNAMIC PORT FOR RAILWAY ---
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Webhook Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)