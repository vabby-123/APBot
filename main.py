import os
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFLUENCE_BASE_URL = "https://apiit.atlassian.net/wiki/rest/api"
GEMINI_MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KNOWLEDGE_BASE_FILE = "knowledge_base.json"
METADATA_FILE = "kb_metadata.json"
UPDATE_INTERVAL_HOURS = 1  # More frequent updates
MAX_WORKERS = 5

# Simple user credentials (in production, use proper authentication)
ADMIN_CREDENTIALS = {
    "username": os.getenv("ADMIN_USERNAME", "admin"),
    "password": os.getenv("ADMIN_PASSWORD", "admin123")
}

STUDENT_CREDENTIALS = {
    "username": os.getenv("STUDENT_USERNAME", "student"),
    "password": os.getenv("STUDENT_PASSWORD", "student123")
}

# Initialize models
configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# FastAPI app
app = FastAPI(title="APBot", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    timestamp: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    timestamp: str

class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    spaces: List[str]
    last_update: Optional[str]
    is_updating: bool

class UpdateRequest(BaseModel):
    force_update: bool = False
    space_keys: Optional[List[str]] = None

class LoginRequest(BaseModel):
    username: str
    password: str

# Global variables
knowledge_base = []
is_updating = False
update_thread = None
connected_clients = {"admin": set(), "student": set()}
user_sessions = {}  # Store user sessions

# Authentication functions
def verify_credentials(username: str, password: str) -> Optional[str]:
    """Verify user credentials and return role"""
    if username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]:
        return "admin"
    elif username == STUDENT_CREDENTIALS["username"] and password == STUDENT_CREDENTIALS["password"]:
        return "student"
    return None

def get_current_user_role(request: Request) -> Optional[str]:
    """Get current user role from session"""
    session_id = request.cookies.get("session_id")
    return user_sessions.get(session_id)

# Include the enhanced classes from previous version (same as before)
class ConfluenceClient:
    def __init__(self):
        self.auth = (
            os.getenv("CONFLUENCE_USERNAME"),
            os.getenv("CONFLUENCE_API_TOKEN")
        )
        self.headers = {"Accept": "application/json"}
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update(self.headers)
    
    def get_all_spaces(self) -> List[Dict]:
        """Fetch all accessible Confluence spaces"""
        spaces = []
        start = 0
        limit = 50
        
        while True:
            url = f"{CONFLUENCE_BASE_URL}/space?limit={limit}&start={start}"
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "results" not in data:
                    break
                
                spaces.extend(data["results"])
                
                if len(data["results"]) < limit:
                    break
                start += limit
                
            except Exception as e:
                logger.error(f"Error fetching spaces: {e}")
                break
        
        return spaces
    
    def get_pages_from_space(self, space_key: str) -> List[Dict]:
        """Fetch all pages from a specific space with metadata"""
        pages = []
        start = 0
        limit = 50
        
        while True:
            url = f"{CONFLUENCE_BASE_URL}/content"
            params = {
                'spaceKey': space_key,
                'limit': limit,
                'start': start,
                'expand': 'version,space,metadata.labels,children.page'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "results" not in data:
                    break
                
                for page in data["results"]:
                    page_info = {
                        'id': page['id'],
                        'title': page['title'],
                        'type': page['type'],
                        'space_key': space_key,
                        'version': page.get('version', {}).get('number', 1),
                        'last_modified': page.get('version', {}).get('when', ''),
                        'web_url': f"https://apiit.atlassian.net/wiki{page.get('_links', {}).get('webui', '')}",
                        'api_url': f"{CONFLUENCE_BASE_URL}/content/{page['id']}?expand=body.storage,version"
                    }
                    pages.append(page_info)
                
                if len(data["results"]) < limit:
                    break
                start += limit
                
            except Exception as e:
                logger.error(f"Error fetching pages from space {space_key}: {e}")
                break
        
        return pages
    
    def get_all_pages(self, space_keys: List[str] = None) -> List[Dict]:
        """Fetch all pages from specified spaces or all accessible spaces"""
        all_pages = []
        
        if space_keys is None:
            spaces = self.get_all_spaces()
            space_keys = [space['key'] for space in spaces]
            logger.info(f"Found {len(space_keys)} spaces: {space_keys}")
        
        for space_key in space_keys:
            logger.info(f"Fetching pages from space: {space_key}")
            pages = self.get_pages_from_space(space_key)
            all_pages.extend(pages)
            logger.info(f"Found {len(pages)} pages in space {space_key}")
            time.sleep(0.5)
        
        return all_pages
    
    def get_page_content(self, page_info: Dict) -> Optional[str]:
        """Fetch the actual content of a page"""
        try:
            response = self.session.get(page_info['api_url'], timeout=30)
            response.raise_for_status()
            data = response.json()
            
            body = data.get('body', {}).get('storage', {}).get('value', '')
            if not body:
                return self._scrape_page_content(page_info['web_url'])
            
            soup = BeautifulSoup(body, 'html.parser')
            return soup.get_text(strip=True)
            
        except Exception as e:
            logger.error(f"Error fetching content for page {page_info['id']}: {e}")
            return self._scrape_page_content(page_info['web_url'])
    
    def _scrape_page_content(self, url: str) -> Optional[str]:
        """Fallback method to scrape page content from web URL"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            selectors = [
                'div[data-testid="content-body"]',
                '.wiki-content',
                '#main-content',
                '.main-content',
                'article',
                '.page-content'
            ]
            
            for selector in selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    return content_div.get_text(strip=True)
            
            body = soup.find('body')
            return body.get_text(strip=True) if body else None
            
        except Exception as e:
            logger.error(f"Error scraping content from {url}: {e}")
            return None

class KnowledgeProcessor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', ' ', text)
        text = re.sub(r'\.{2,}', '.', text)
        
        return text.strip()
    
    def chunk_content(self, text: str) -> List[str]:
        """Split content into manageable chunks with smart splitting"""
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    sentences = re.split(r'[.!?]+', paragraph)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        if len(current_chunk) + len(sentence) > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            current_chunk += " " + sentence if current_chunk else sentence
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def save_knowledge_base(self, knowledge_base: List[Dict], filename: str = KNOWLEDGE_BASE_FILE):
        """Save processed knowledge base to file"""
        with open(filename, 'w') as f:
            json.dump(knowledge_base, f, indent=2)
    
    def load_knowledge_base(self, filename: str = KNOWLEDGE_BASE_FILE) -> List[Dict]:
        """Load processed knowledge base from file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

class KnowledgeBaseManager:
    def __init__(self):
        self.confluence = ConfluenceClient()
        self.processor = KnowledgeProcessor()
        self.metadata = self.load_metadata()
        self.lock = threading.Lock()
    
    def load_metadata(self) -> Dict:
        """Load knowledge base metadata"""
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'last_update': None,
                'total_pages': 0,
                'pages_hash': {},
                'spaces': []
            }
    
    def save_metadata(self):
        """Save knowledge base metadata"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_content_hash(self, content: str) -> str:
        """Generate hash for content to detect changes"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def should_update(self) -> bool:
        """Check if knowledge base should be updated"""
        if not self.metadata['last_update']:
            return True
        
        last_update = datetime.fromisoformat(self.metadata['last_update'])
        return datetime.now() - last_update > timedelta(hours=UPDATE_INTERVAL_HOURS)
    
    async def update_knowledge_base_async(self, space_keys: List[str] = None, force_update: bool = False):
        """Asynchronously update the knowledge base"""
        global is_updating, knowledge_base
        
        if is_updating:
            return {"status": "already_updating"}
        
        if not force_update and not self.should_update():
            return {"status": "up_to_date"}
        
        is_updating = True
        
        try:
            # Broadcast update start to admin clients only
            await broadcast_message({
                "type": "update_status",
                "status": "started",
                "message": "Starting knowledge base update..."
            }, "admin")
            
            # Get all pages
            all_pages = self.confluence.get_all_pages(space_keys)
            
            await broadcast_message({
                "type": "update_status",
                "status": "progress",
                "message": f"Found {len(all_pages)} pages to process"
            }, "admin")
            
            # Load existing knowledge base
            existing_kb = self.processor.load_knowledge_base()
            existing_pages_dict = {page['id']: page for page in existing_kb}
            
            # Find pages that need updating
            pages_to_update = self.get_new_or_updated_pages(all_pages)
            
            await broadcast_message({
                "type": "update_status",
                "status": "progress", 
                "message": f"Processing {len(pages_to_update)} new/updated pages"
            }, "admin")
            
            # Process pages in batches
            batch_size = 10
            updated_pages = []
            
            for i in range(0, len(pages_to_update), batch_size):
                batch = pages_to_update[i:i + batch_size]
                
                await broadcast_message({
                    "type": "update_status",
                    "status": "progress",
                    "message": f"Processing batch {i//batch_size + 1}/{(len(pages_to_update)//batch_size)+1}"
                }, "admin")
                
                batch_results = await self.process_pages_batch_async(batch)
                updated_pages.extend(batch_results)
                
                # Update knowledge base file
                self._update_knowledge_base_file(existing_pages_dict, updated_pages)
                
                # Reload global knowledge base
                knowledge_base = list(existing_pages_dict.values())
            
            # Update metadata
            self.metadata['last_update'] = datetime.now().isoformat()
            self.metadata['total_pages'] = len(existing_pages_dict)
            self.metadata['spaces'] = list(set(page['space_key'] for page in all_pages))
            self.save_metadata()
            
            await broadcast_message({
                "type": "update_status",
                "status": "completed",
                "message": f"Knowledge base updated! Processed {len(updated_pages)} pages."
            }, "admin")
            
            return {
                "status": "completed",
                "processed_pages": len(updated_pages),
                "total_pages": len(existing_pages_dict)
            }
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            await broadcast_message({
                "type": "update_status",
                "status": "error",
                "message": f"Error updating knowledge base: {str(e)}"
            }, "admin")
            return {"status": "error", "message": str(e)}
        
        finally:
            is_updating = False
    
    def get_new_or_updated_pages(self, all_pages: List[Dict]) -> List[Dict]:
        """Identify pages that are new or have been updated"""
        new_or_updated = []
        
        for page in all_pages:
            page_id = page['id']
            current_version = page['version']
            
            if (page_id not in self.metadata['pages_hash'] or 
                self.metadata['pages_hash'][page_id].get('version', 0) < current_version):
                new_or_updated.append(page)
        
        return new_or_updated
    
    async def process_pages_batch_async(self, pages: List[Dict]) -> List[Dict]:
        """Process a batch of pages concurrently"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [
                loop.run_in_executor(executor, self.process_single_page, page)
                for page in pages
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_pages = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {result}")
            elif result:
                processed_pages.append(result)
        
        return processed_pages
    
    def process_single_page(self, page_info: Dict) -> Optional[Dict]:
        """Process a single page"""
        try:
            content = self.confluence.get_page_content(page_info)
            if not content:
                return None
            
            cleaned_content = self.processor.clean_text(content)
            chunks = self.processor.chunk_content(cleaned_content)
            
            if not chunks:
                return None
            
            embeddings = embedding_model.encode(chunks)
            
            content_hash = self.get_content_hash(cleaned_content)
            with self.lock:
                self.metadata['pages_hash'][page_info['id']] = {
                    'version': page_info['version'],
                    'hash': content_hash,
                    'last_processed': datetime.now().isoformat()
                }
                
            return {
                'id': page_info['id'],
                'title': page_info['title'],
                'url': page_info['web_url'],
                'space_key': page_info['space_key'],
                'version': page_info['version'],
                'chunks': chunks,
                'embeddings': [emb.tolist() for emb in embeddings],
                'last_modified': page_info['last_modified'],
                'content_hash': content_hash
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_info.get('id', 'unknown')}: {e}")
            return None
    
    def _update_knowledge_base_file(self, existing_pages: Dict, updated_pages: List[Dict]):
        """Update the knowledge base file with new/updated pages"""
        for page in updated_pages:
            existing_pages[page['id']] = page
        
        knowledge_base_list = list(existing_pages.values())
        self.processor.save_knowledge_base(knowledge_base_list)

# Initialize global objects
kb_manager = KnowledgeBaseManager()
processor = KnowledgeProcessor()

# Load initial knowledge base
knowledge_base = processor.load_knowledge_base()

# WebSocket functions
async def broadcast_message(message: dict, role: str = "all"):
    """Broadcast message to connected WebSocket clients based on role"""
    if role == "all":
        clients = connected_clients["admin"] | connected_clients["student"]
    else:
        clients = connected_clients.get(role, set())
    
    if clients:
        disconnected = set()
        for websocket in clients:
            try:
                await websocket.send_json(message)
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            for role_clients in connected_clients.values():
                role_clients.discard(ws)

def find_relevant_chunks(query: str, top_k: int = 5) -> List[Dict]:
    """Find most relevant knowledge base chunks for a query"""
    global knowledge_base
    
    if not knowledge_base:
        return []
    
    query_embedding = embedding_model.encode([query])
    
    all_chunks = []
    for doc in knowledge_base:
        for i, (chunk, emb) in enumerate(zip(doc['chunks'], doc['embeddings'])):
            all_chunks.append({
                'text': chunk,
                'embedding': np.array(emb),
                'url': doc['url'],
                'title': doc.get('title', doc['url']),
                'space_key': doc.get('space_key', 'Unknown'),
                'chunk_id': i,
                'doc_id': doc.get('id', 'unknown')
            })
    
    if not all_chunks:
        return []
    
    # Calculate similarity scores
    for chunk in all_chunks:
        chunk['similarity'] = cosine_similarity(
            query_embedding.reshape(1, -1), 
            chunk['embedding'].reshape(1, -1)
        )[0][0]
    
    # Sort by similarity and return top results
    all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    return all_chunks[:top_k]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_response(query: str) -> ChatResponse:
    """Generate a response using Gemini with RAG"""
    # Find relevant context
    relevant_chunks = find_relevant_chunks(query, top_k=7)
    
    # Prepare context
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(
            f"Source: {chunk['title']} (Space: {chunk['space_key']})\n"
            f"URL: {chunk['url']}\n"
            f"Content: {chunk['text']}\n"
            f"Relevance Score: {chunk['similarity']:.3f}"
        )
    
    context = "\n\n---\n".join(context_parts)
    
    # Enhanced prompt
    prompt = f"""You are an intelligent IT support chatbot with access to a comprehensive Confluence knowledge base. 

Your guidelines:
1. Use the provided context to answer questions accurately and helpfully
2. If the answer isn't in the context, clearly state that you don't have that information
3. Provide step-by-step instructions when appropriate
4. Be professional but friendly in your tone
5. If multiple solutions exist, provide the most relevant one first

Knowledge Base Context:
{context}

Question: {query}

Please provide a comprehensive, accurate answer based on the available information:"""
    
    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 2000,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        # Prepare sources
        sources = []
        if relevant_chunks and relevant_chunks[0]['similarity'] > 0.3:
            seen_urls = set()
            for chunk in relevant_chunks[:3]:
                if chunk['url'] not in seen_urls and chunk['similarity'] > 0.2:
                    sources.append({
                        'title': chunk['title'],
                        'url': chunk['url'],
                        'space': chunk['space_key'],
                        'relevance': round(chunk['similarity'], 3)
                    })
                    seen_urls.add(chunk['url'])
        
        return ChatResponse(
            response=response.text,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return ChatResponse(
            response=f"I encountered an error generating a response. Please try again. Error: {str(e)}",
            sources=[],
            timestamp=datetime.now().isoformat()
        )

# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_login_page():
    """Serve the login page"""
    return HTMLResponse(content=get_login_html())

@app.post("/login")
async def login(login_data: LoginRequest):
    """Handle login"""
    role = verify_credentials(login_data.username, login_data.password)
    if not role:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create session
    session_id = hashlib.md5(f"{login_data.username}{datetime.now()}".encode()).hexdigest()
    user_sessions[session_id] = role
    
    # Return success with redirect URL
    return {
        "success": True,
        "role": role,
        "session_id": session_id,
        "redirect_url": "/admin" if role == "admin" else "/student"
    }

@app.get("/admin", response_class=HTMLResponse)
async def get_admin_page(request: Request):
    """Serve the admin page"""
    role = get_current_user_role(request)
    if role != "admin":
        return RedirectResponse(url="/")
    
    return HTMLResponse(content=get_admin_html())

@app.get("/student", response_class=HTMLResponse)
async def get_student_page(request: Request):
    """Serve the student page"""
    role = get_current_user_role(request)
    if role != "student":
        return RedirectResponse(url="/")
    
    return HTMLResponse(content=get_student_html())

@app.get("/logout")
async def logout(request: Request):
    """Handle logout"""
    session_id = request.cookies.get("session_id")
    if session_id in user_sessions:
        del user_sessions[session_id]
    
    response = RedirectResponse(url="/")
    response.delete_cookie("session_id")
    return response

@app.get("/api/stats")
async def get_stats(request: Request) -> KnowledgeBaseStats:
    """Get knowledge base statistics (admin only)"""
    role = get_current_user_role(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    global knowledge_base, is_updating
    
    if not knowledge_base:
        total_chunks = 0
        spaces = []
    else:
        total_chunks = sum(len(doc.get('chunks', [])) for doc in knowledge_base)
        spaces = list(set(doc.get('space_key', 'Unknown') for doc in knowledge_base))
    
    return KnowledgeBaseStats(
        total_documents=len(knowledge_base),
        total_chunks=total_chunks,
        spaces=spaces,
        last_update=kb_manager.metadata.get('last_update'),
        is_updating=is_updating
    )

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage, request: Request) -> ChatResponse:
    """Handle chat messages (both admin and student)"""
    role = get_current_user_role(request)
    if not role:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return await generate_response(message.message)

@app.post("/api/update")
async def update_knowledge_base(request_data: UpdateRequest, request: Request, background_tasks: BackgroundTasks):
    """Trigger knowledge base update (admin only)"""
    role = get_current_user_role(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")
    
    background_tasks.add_task(
        kb_manager.update_knowledge_base_async,
        request_data.space_keys,
        request_data.force_update
    )
    return {"status": "update_started"}

@app.websocket("/ws/{role}")
async def websocket_endpoint(websocket: WebSocket, role: str):
    """WebSocket endpoint for real-time communication"""
    if role not in ["admin", "student"]:
        await websocket.close(code=4000)
        return
    
    await websocket.accept()
    connected_clients[role].add(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "chat":
                # Handle chat message
                response = await generate_response(data.get("message", ""))
                await websocket.send_json({
                    "type": "chat_response",
                    "data": response.dict()
                })
            elif data.get("type") == "get_stats" and role == "admin":
                # Handle stats request (admin only)
                stats_data = {
                    "total_documents": len(knowledge_base),
                    "total_chunks": sum(len(doc.get('chunks', [])) for doc in knowledge_base),
                    "spaces": list(set(doc.get('space_key', 'Unknown') for doc in knowledge_base)),
                    "last_update": kb_manager.metadata.get('last_update'),
                    "is_updating": is_updating
                }
                await websocket.send_json({
                    "type": "stats_response",
                    "data": stats_data
                })
    
    except WebSocketDisconnect:
        connected_clients[role].discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connected_clients[role].discard(websocket)

def get_login_html(error: str = None) -> str:
    """Generate the HTML content for the login page"""
    error_html = f'<div class="error" id="error-msg">{error}</div>' if error else '<div class="error" id="error-msg" style="display:none;"></div>'
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - APBot</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .login-container {{
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 400px;
            text-align: center;
        }}
        .login-container h1 {{
            color: #333;
            margin-bottom: 2rem;
        }}
        .form-group {{
            margin-bottom: 1.5rem;
            text-align: left;
        }}
        .form-group label {{
            display: block;
            margin-bottom: 0.5rem;
            color: #666;
            font-weight: 500;
        }}
        .form-group input {{
            width: 100%;
            padding: 1rem;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }}
        .form-group input:focus {{
            border-color: #667eea;
        }}
        .login-btn {{
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            cursor: pointer;
            transition: transform 0.3s;
        }}
        .login-btn:hover {{
            transform: translateY(-2px);
        }}
        .login-btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }}
        .error {{
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }}
        .credentials {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            font-size: 0.9rem;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="login-container">
        <h1>🤖 APBot</h1>
        {error_html}
        <form id="login-form">
            <div class="form-group">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
            </div>
            <div class="form-group">
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
            </div>
            <button type="submit" class="login-btn" id="login-btn">Login</button>
        </form>
        <div class="credentials">
            <strong>Demo Credentials:</strong><br>
            Admin: admin / admin123<br>
            Student: student / student123
        </div>
    </div>

    <script>
        document.getElementById('login-form').addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const loginBtn = document.getElementById('login-btn');
            const errorMsg = document.getElementById('error-msg');
            
            // Disable button and show loading
            loginBtn.disabled = true;
            loginBtn.textContent = 'Logging in...';
            errorMsg.style.display = 'none';
            
            try {{
                const response = await fetch('/login', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        username: username,
                        password: password
                    }})
                }});
                
                const data = await response.json();
                
                if (response.ok && data.success) {{
                    // Set session cookie
                    document.cookie = `session_id=${{data.session_id}}; max-age=86400; path=/`;
                    
                    // Redirect based on role
                    window.location.href = data.redirect_url;
                }} else {{
                    throw new Error(data.detail || 'Login failed');
                }}
            }} catch (error) {{
                errorMsg.textContent = error.message || 'Login failed. Please try again.';
                errorMsg.style.display = 'block';
            }} finally {{
                // Re-enable button
                loginBtn.disabled = false;
                loginBtn.textContent = 'Login';
            }}
        }});
        
        // Handle Enter key
        document.addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                document.getElementById('login-form').dispatchEvent(new Event('submit'));
            }}
        }});
    </script>
</body>
</html>
"""

def get_admin_html() -> str:
    """Generate the HTML content for the admin interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Panel - APBot</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .header-actions { display: flex; gap: 1rem; align-items: center; }
        
        .logout-btn, .update-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s;
        }
        
        .logout-btn { background: #f44336; }
        .logout-btn:hover { background: #d32f2f; }
        .update-btn:hover { background: #45a049; }
        .update-btn:disabled { background: #cccccc; cursor: not-allowed; }
        
        .main-container {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
            padding: 2rem;
        }
        
        .chat-section, .admin-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            text-align: center;
        }
        
        .chat-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 500px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
        }
        
        .message {
            margin-bottom: 1.5rem;
            padding: 1rem 1.5rem;
            border-radius: 15px;
            max-width: 85%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            margin-right: auto;
        }
        
        .sources {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .chat-input-container {
            padding: 2rem;
            border-top: 1px solid #e9ecef;
            background: white;
        }
        
        .input-wrapper { position: relative; }
        
        .chat-input {
            width: 100%;
            padding: 1rem 3rem 1rem 1.5rem;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input:focus { border-color: #667eea; }
        
        .send-btn {
            position: absolute;
            right: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .admin-content {
            padding: 2rem;
            flex: 1;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            color: #666;
            margin-top: 0.5rem;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        
        .status-online { background: #d4edda; color: #155724; }
        .status-updating { background: #fff3cd; color: #856404; }
        .status-offline { background: #f8d7da; color: #721c24; }
        
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
        
        .spaces-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
        }
        
        .space-tag {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            margin: 0.2rem;
        }
        
        .update-log {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.8rem;
        }
        
        .log-entry {
            margin-bottom: 0.5rem;
            padding: 0.3rem;
            border-radius: 5px;
        }
        
        .log-info { background: #cce5ff; color: #0066cc; }
        .log-success { background: #d4edda; color: #155724; }
        .log-error { background: #f8d7da; color: #721c24; }
        
        .welcome-message {
            text-align: center;
            color: #666;
            margin: 2rem 0;
            font-style: italic;
        }
        
        @media (max-width: 1200px) {
            .main-container {
                grid-template-columns: 1fr;
                gap: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛠️ Admin Panel - APBot</h1>
        <div class="header-actions">
            <button class="update-btn" id="update-btn" onclick="updateKnowledgeBase()">
                🔄 Update KB
            </button>
            <a href="/logout" class="logout-btn">🚪 Logout</a>
        </div>
    </div>

    <div class="main-container">
        <!-- Chat Section -->
        <div class="chat-section">
            <div class="section-header">
                <h2>💬 Test Chat Interface</h2>
            </div>
            
            <div class="chat-content">
                <div class="chat-messages" id="chat-messages">
                    <div class="welcome-message">
                        👋 Admin test interface. Ask questions to test the chatbot functionality.
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <div class="input-wrapper">
                        <input 
                            type="text" 
                            class="chat-input" 
                            id="chat-input" 
                            placeholder="Test a question..."
                            onkeypress="handleKeyPress(event)"
                        >
                        <button class="send-btn" onclick="sendMessage()">➤</button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Admin Section -->
        <div class="admin-section">
            <div class="section-header">
                <h2>📊 System Dashboard</h2>
            </div>
            
            <div class="admin-content">
                <div class="status-indicator" id="connection-status">
                    <div class="status-dot"></div>
                    <span>Connecting...</span>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number" id="stat-docs">0</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="stat-chunks">0</div>
                        <div class="stat-label">Content Chunks</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="stat-spaces">0</div>
                        <div class="stat-label">Spaces</div>
                    </div>
                </div>

                <div class="spaces-section">
                    <h4>🏢 Confluence Spaces</h4>
                    <div id="spaces-container">
                        <span class="space-tag">Loading...</span>
                    </div>
                </div>

                <div class="spaces-section">
                    <h4>🕒 Last Updated</h4>
                    <p id="last-update">Never</p>
                </div>

                <div class="update-log" id="update-log" style="display: none;">
                    <h4>📝 Update Log</h4>
                    <div id="log-content"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let isConnected = false;
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/admin`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                isConnected = true;
                updateConnectionStatus('online', 'Connected');
                requestStats();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            };
            
            ws.onclose = function() {
                isConnected = false;
                updateConnectionStatus('offline', 'Disconnected');
                setTimeout(initWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('offline', 'Connection Error');
            };
        }
        
        function handleWebSocketMessage(data) {
            switch(data.type) {
                case 'chat_response':
                    handleChatResponse(data.data);
                    break;
                case 'stats_response':
                    updateStats(data.data);
                    break;
                case 'update_status':
                    handleUpdateStatus(data);
                    break;
            }
        }
        
        function updateConnectionStatus(status, message) {
            const statusEl = document.getElementById('connection-status');
            statusEl.className = `status-indicator status-${status}`;
            statusEl.querySelector('span').textContent = message;
        }
        
        function requestStats() {
            if (isConnected) {
                ws.send(JSON.stringify({type: 'get_stats'}));
            }
        }
        
        function updateStats(stats) {
            document.getElementById('stat-docs').textContent = stats.total_documents;
            document.getElementById('stat-chunks').textContent = stats.total_chunks;
            document.getElementById('stat-spaces').textContent = stats.spaces.length;
            
            const lastUpdate = stats.last_update ? 
                new Date(stats.last_update).toLocaleString() : 'Never';
            document.getElementById('last-update').textContent = lastUpdate;
            
            const spacesContainer = document.getElementById('spaces-container');
            if (stats.spaces && stats.spaces.length > 0) {
                spacesContainer.innerHTML = stats.spaces.map(space => 
                    `<span class="space-tag">${space}</span>`
                ).join('');
            } else {
                spacesContainer.innerHTML = '<span class="space-tag">No spaces found</span>';
            }
            
            const updateBtn = document.getElementById('update-btn');
            if (stats.is_updating) {
                updateBtn.disabled = true;
                updateBtn.textContent = '🔄 Updating...';
                updateConnectionStatus('updating', 'Updating Knowledge Base');
            } else {
                updateBtn.disabled = false;
                updateBtn.textContent = '🔄 Update KB';
                if (isConnected) {
                    updateConnectionStatus('online', 'Connected');
                }
            }
        }
        
        function handleUpdateStatus(data) {
            const logContainer = document.getElementById('update-log');
            const logContent = document.getElementById('log-content');
            
            if (data.status === 'started') {
                logContainer.style.display = 'block';
                logContent.innerHTML = '';
            }
            
            const logEntry = document.createElement('div');
            logEntry.className = `log-entry log-${data.status === 'error' ? 'error' : 
                                              data.status === 'completed' ? 'success' : 'info'}`;
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${data.message}`;
            logContent.appendChild(logEntry);
            logContent.scrollTop = logContent.scrollHeight;
            
            if (data.status === 'completed' || data.status === 'error') {
                setTimeout(() => {
                    requestStats();
                }, 1000);
            }
        }
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message || !isConnected) return;
            
            addMessage(message, 'user');
            input.value = '';
            
            ws.send(JSON.stringify({
                type: 'chat',
                message: message
            }));
        }
        
        function handleChatResponse(data) {
            addMessage(data.response, 'bot', data.sources);
        }
        
        function addMessage(content, sender, sources = null) {
            const messagesContainer = document.getElementById('chat-messages');
            const messageEl = document.createElement('div');
            messageEl.className = `message ${sender}-message`;
            
            if (sender === 'bot') {
                const markdownContent = marked.parse(content);
                messageEl.innerHTML = `<div>${markdownContent}</div>`;
            } else {
                messageEl.textContent = content;
            }
            
            if (sources && sources.length > 0) {
                const sourcesEl = document.createElement('div');
                sourcesEl.className = 'sources';
                sourcesEl.innerHTML = `
                    <h4>📚 Sources:</h4>
                    ${sources.map(source => `
                        <div>
                            <a href="${source.url}" target="_blank">${source.title}</a>
                            (${(source.relevance * 100).toFixed(1)}%)
                        </div>
                    `).join('')}
                `;
                messageEl.appendChild(sourcesEl);
            }
            
            messagesContainer.appendChild(messageEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function updateKnowledgeBase() {
            if (!isConnected) return;
            
            fetch('/api/update', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    force_update: true
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Update triggered:', data);
            })
            .catch(error => {
                console.error('Error triggering update:', error);
            });
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            document.getElementById('chat-input').focus();
            
            // Request stats every 30 seconds
            setInterval(() => {
                if (isConnected) {
                    requestStats();
                }
            }, 30000);
        });
    </script>
</body>
</html>
"""

def get_student_html() -> str:
    """Generate the HTML content for the student interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IT Support Chat - APBot</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            color: white;
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .logout-btn {
            background: #f44336;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9rem;
            text-decoration: none;
            transition: background 0.3s;
        }
        
        .logout-btn:hover { background: #d32f2f; }
        
        .main-container {
            flex: 1;
            display: flex;
            justify-content: center;
            padding: 2rem;
        }
        
        .chat-container {
            width: 100%;
            max-width: 800px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 600px;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem 2rem;
            text-align: center;
        }
        
        .chat-messages {
            flex: 1;
            padding: 2rem;
            overflow-y: auto;
        }
        
        .message {
            margin-bottom: 1.5rem;
            padding: 1rem 1.5rem;
            border-radius: 15px;
            max-width: 85%;
            word-wrap: break-word;
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            margin-right: auto;
        }
        
        .message-time {
            font-size: 0.8rem;
            opacity: 0.7;
            margin-top: 0.5rem;
        }
        
        .sources {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .sources h4 {
            color: #667eea;
            margin-bottom: 0.5rem;
            font-size: 0.9rem;
        }
        
        .source-item {
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        
        .source-item a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        
        .source-item a:hover {
            text-decoration: underline;
        }
        
        .relevance-score {
            background: #e9ecef;
            padding: 0.2rem 0.5rem;
            border-radius: 10px;
            font-size: 0.7rem;
            margin-left: 0.5rem;
        }
        
        .chat-input-container {
            padding: 2rem;
            border-top: 1px solid #e9ecef;
            background: white;
        }
        
        .input-wrapper { position: relative; }
        
        .chat-input {
            width: 100%;
            padding: 1rem 3rem 1rem 1.5rem;
            border: 2px solid #e9ecef;
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input:focus { border-color: #667eea; }
        
        .send-btn {
            position: absolute;
            right: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.3s;
        }
        
        .send-btn:hover {
            transform: translateY(-50%) scale(1.1);
        }
        
        .typing-indicator {
            display: none;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            color: #666;
            font-style: italic;
        }
        
        .typing-dots::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80%, 100% { content: '...'; }
        }
        
        .welcome-message {
            text-align: center;
            color: #666;
            margin: 2rem 0;
            font-style: italic;
        }
        
        .connection-status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            z-index: 1000;
        }
        
        .status-online { background: #d4edda; color: #155724; }
        .status-offline { background: #f8d7da; color: #721c24; }
        
        @media (max-width: 768px) {
            .main-container { padding: 1rem; }
            .chat-container { height: calc(100vh - 200px); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 IT Support Assistant</h1>
        <a href="/logout" class="logout-btn">🚪 Logout</a>
    </div>

    <div class="connection-status" id="connection-status">
        Connecting...
    </div>

    <div class="main-container">
        <div class="chat-container">
            <div class="chat-header">
                <h2>💬 Ask me anything!</h2>
                <p>I'm here to help with IT support questions and procedures</p>
            </div>
            
            <div class="chat-messages" id="chat-messages">
                <div class="welcome-message">
                    👋 Hello! I'm your IT support assistant. I can help you with:
                    <br><br>
                    • Troubleshooting technical issues
                    <br>• IT procedures and policies
                    <br>• Software and hardware guidance
                    <br>• Account and access problems
                    <br><br>
                    What can I help you with today?
                </div>
            </div>
            
            <div class="typing-indicator" id="typing-indicator">
                🤖 Assistant is typing<span class="typing-dots"></span>
            </div>
            
            <div class="chat-input-container">
                <div class="input-wrapper">
                    <input 
                        type="text" 
                        class="chat-input" 
                        id="chat-input" 
                        placeholder="Type your question here..."
                        onkeypress="handleKeyPress(event)"
                    >
                    <button class="send-btn" onclick="sendMessage()">➤</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let isConnected = false;
        
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/student`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = function() {
                isConnected = true;
                updateConnectionStatus('online', '🟢 Connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'chat_response') {
                    handleChatResponse(data.data);
                }
            };
            
            ws.onclose = function() {
                isConnected = false;
                updateConnectionStatus('offline', '🔴 Disconnected');
                setTimeout(initWebSocket, 3000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('offline', '🔴 Connection Error');
            };
        }
        
        function updateConnectionStatus(status, message) {
            const statusEl = document.getElementById('connection-status');
            statusEl.className = `connection-status status-${status}`;
            statusEl.textContent = message;
        }
        
        function sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message || !isConnected) return;
            
            addMessage(message, 'user');
            input.value = '';
            showTypingIndicator();
            
            ws.send(JSON.stringify({
                type: 'chat',
                message: message
            }));
        }
        
        function handleChatResponse(data) {
            hideTypingIndicator();
            addMessage(data.response, 'bot', data.sources, data.timestamp);
        }
        
        function addMessage(content, sender, sources = null, timestamp = null) {
            const messagesContainer = document.getElementById('chat-messages');
            const messageEl = document.createElement('div');
            messageEl.className = `message ${sender}-message`;
            
            if (sender === 'bot') {
                const markdownContent = marked.parse(content);
                messageEl.innerHTML = `<div>${markdownContent}</div>`;
            } else {
                messageEl.textContent = content;
            }
            
            if (timestamp) {
                const timeEl = document.createElement('div');
                timeEl.className = 'message-time';
                timeEl.textContent = new Date(timestamp).toLocaleTimeString();
                messageEl.appendChild(timeEl);
            }
            
            if (sources && sources.length > 0) {
                const sourcesEl = document.createElement('div');
                sourcesEl.className = 'sources';
                sourcesEl.innerHTML = `
                    <h4>📚 Sources:</h4>
                    ${sources.map(source => `
                        <div class="source-item">
                            <a href="${source.url}" target="_blank">${source.title}</a>
                            <span class="relevance-score">${(source.relevance * 100).toFixed(1)}%</span>
                            <br><small>Space: ${source.space}</small>
                        </div>
                    `).join('')}
                `;
                messageEl.appendChild(sourcesEl);
            }
            
            messagesContainer.appendChild(messageEl);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function showTypingIndicator() {
            document.getElementById('typing-indicator').style.display = 'block';
        }
        
        function hideTypingIndicator() {
            document.getElementById('typing-indicator').style.display = 'none';
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        document.addEventListener('DOMContentLoaded', function() {
            initWebSocket();
            document.getElementById('chat-input').focus();
        });
    </script>
</body>
</html>
"""

# Background task for automatic updates
async def background_updater():
    """Background task to automatically update knowledge base"""
    while True:
        try:
            await asyncio.sleep(UPDATE_INTERVAL_HOURS * 3600)
            
            if not is_updating and kb_manager.should_update():
                logger.info("Starting automatic knowledge base update")
                await kb_manager.update_knowledge_base_async()
                
        except Exception as e:
            logger.error(f"Error in background updater: {e}")

# Start background updater
@app.on_event("startup")
async def startup_event():
    """Initialize background tasks on startup"""
    asyncio.create_task(background_updater())
    logger.info("APBot started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("APBot shutting down")

if __name__ == "__main__":
    # Ensure required environment variables are set
    required_vars = ["CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your .env file")
        exit(1)
    
    # Create static directories if they don't exist
    Path("static").mkdir(exist_ok=True)
    
    logger.info("Starting APBot...")
    print("🚀 Starting APBot...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("👤 Admin: admin / admin123")
    print("🎓 Student: student / student123")
    
    # uvicorn.run(
    #     app,
    #     host="0.0.0.0",
    #     port=8000,
    #     log_level="info",
    #     reload=False
    # )
    
    # Replace the uvicorn.run() call at the end of main.py with:
if __name__ == "__main__":
    # Check if running on HF Spaces
    port = int(os.getenv("PORT", 8000))
    
    # Ensure required environment variables are set
    required_vars = ["CONFLUENCE_USERNAME", "CONFLUENCE_API_TOKEN", "GEMINI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these in your Hugging Face Space settings")
        # Don't exit on HF Spaces, just warn
        if not os.getenv("SPACE_ID"):  # Only exit if not on HF Spaces
            exit(1)
    
    # Create static directories if they don't exist
    Path("static").mkdir(exist_ok=True)
    
    logger.info("Starting APBot...")
    print("🚀 Starting APBot...")
    print(f"📱 Server running on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )