import os
import logging
import json
import uuid
import asyncio
from typing import List, Dict

import fitz  # PyMuPDF
import uvicorn
# Added BackgroundTasks to imports
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI  # DeepSeek is OpenAI Compatible

# ============================================================================
# CONFIGURATION
# ============================================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY") 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat" 

if not DEEPSEEK_API_KEY:
    print("❌ ERROR: DEEPSEEK_API_KEY is missing!")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("DeepSeekScanner")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_log(self, client_id: str, message: str, level: str = "info"):
        print(f"[{client_id}] {message}")
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({
                    "type": "log", "level": level, "message": message
                })
            except: pass

    async def send_result(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                "type": "result", "data": data
            })

ws_manager = ConnectionManager()
books_db = {}

# ============================================================================
# DEEPSEEK LOGIC
# ============================================================================
class DeepSeekAnalyzer:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

    async def process_book(self, book_id: str, file_path: str, client_id: str):
        await ws_manager.send_log(client_id, "📖 Extracting text from PDF...")
        
        doc = fitz.open(file_path)
        full_text_map = [] 
        llm_input_lines = []
        
        for i, page in enumerate(doc):
            text = page.get_text()
            full_text_map.append(text) 
            # Send page number + first 1000 chars to AI
            clean_excerpt = text[:1000].replace('\n', ' ')
            llm_input_lines.append(f"PAGE_ID_{i}: {clean_excerpt}")

        doc.close()
        
        context_str = "\n".join(llm_input_lines)
        
        # Truncate if over 128k tokens (approx 500k chars)
        if len(context_str) > 500000:
            await ws_manager.send_log(client_id, "⚠️ Book is huge! Truncating...", "warning")
            context_str = context_str[:500000]

        await ws_manager.send_log(client_id, f"🚀 Sending {len(context_str)} chars to DeepSeek V3...", "info")

        try:
            chapters_metadata = await self.ask_deepseek(context_str)
            
            await ws_manager.send_log(client_id, f"✅ DeepSeek identified {len(chapters_metadata)} chapters!", "success")
            
            final_output = []
            for chap in chapters_metadata:
                start = chap.get("start_page", 0)
                end = chap.get("end_page", 0)
                
                # Retrieve Full Text
                chapter_text = ""
                start = max(0, min(start, len(full_text_map)-1))
                end = max(0, min(end, len(full_text_map)-1))
                
                for p_idx in range(start, end + 1):
                    chapter_text += full_text_map[p_idx] + "\n"
                
                final_output.append({
                    "index": chap.get("index"),
                    "title": chap.get("title"),
                    "text": chapter_text
                })

            books_db[book_id] = final_output
            await ws_manager.send_result(client_id, final_output)

        except Exception as e:
            await ws_manager.send_log(client_id, f"DeepSeek Error: {str(e)}", "error")

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
        return json.loads(content).get("chapters", []) if "chapters" in content else json.loads(content)

analyzer = DeepSeekAnalyzer()

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await ws_manager.connect(websocket, client_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)

@app.post("/upload")
async def upload_book(background_tasks: BackgroundTasks, file: UploadFile = File(...), client_id: str = "default"):
    book_id = str(uuid.uuid4())
    file_path = f"temp_{book_id}.pdf"
    content = await file.read()
    with open(file_path, "wb") as f: f.write(content)
    
    background_tasks.add_task(analyzer.process_book, book_id, file_path, client_id)
    return {"book_id": book_id}

@app.get("/book/{book_id}/chapter/{index}")
async def get_chapter(book_id: str, index: int):
    chapters = books_db.get(book_id, [])
    for c in chapters:
        if c["index"] == index: return c
    return {"error": "Not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)