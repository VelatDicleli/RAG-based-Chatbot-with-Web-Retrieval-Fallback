import os
import uuid
import tempfile
import logging
import time
from typing import Dict, Any, Optional, List
import traceback
import shutil
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage


from core.graph import build_rag_graph

rag_graph = build_rag_graph()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


sessions: Dict[str, Dict[str, Any]] = {}

class QueryModel(BaseModel):
    query: str

class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str

class QueryResponse(BaseModel):
    session_id: str
    query: str
    answer: str

class SessionNotFoundError(Exception):
    """Exception raised when a session is not found"""
    pass

class SessionNotReadyError(Exception):
    """Exception raised when a session is not ready for use"""
    pass

def get_session(session_id: str) -> Dict[str, Any]:
    """Get session by ID with validation"""
    if session_id not in sessions:
        logger.error(f"Session not found: {session_id}")
        raise SessionNotFoundError(f"Session not found: {session_id}")
    
    return sessions[session_id]

@app.post("/upload", response_model=SessionResponse)
async def upload_file(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a file and initialize a new RAG session"""
    try:
       
        session_id = str(uuid.uuid4())
        logger.info(f"Creating new session {session_id} for file {file.filename}")
        
        
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved to {file_path}")
        
       
        sessions[session_id] = {
            "file_path": file_path,
            "status": "initializing",
            "thread_id": str(uuid.uuid4()),
            "last_checkpoint": None,
            "creation_time": time.time(),
            "last_activity": time.time()
        }
        
        
        background_tasks.add_task(initialize_rag_session, session_id)
        
        return SessionResponse(
            session_id=session_id,
            status="initializing",
            message="Dosya yüklendi, oturum başlatılıyor..."
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Dosya yüklenirken hata oluştu: {str(e)}")

async def initialize_rag_session(session_id: str):
    """Initialize RAG session in background"""
    if session_id not in sessions:
        logger.error(f"Session {session_id} no longer exists")
        return
        
    session = sessions[session_id]
    
    try:
        logger.info(f"Initializing session {session_id}")
        
       
        config = {
            "configurable": {
                "thread_id": session["thread_id"]
            }
        }
        
        
        inputs = {
            "file_path": session["file_path"],
            "messages": [
                HumanMessage(content="Doküman işlemeyi başlat")
            ]
        }
        
       
        try:
            result = rag_graph.invoke(inputs, config)
            sessions[session_id]["status"] = "ready"
            sessions[session_id]["last_checkpoint"] = result
            sessions[session_id]["last_activity"] = time.time()
            logger.info(f"Session {session_id} initialized successfully")
        except Exception as graph_error:
            logger.error(f"RAG graph error: {graph_error}")
            logger.error(traceback.format_exc())
            sessions[session_id]["status"] = "error"
            sessions[session_id]["error"] = str(graph_error)
            
    except Exception as e:
        logger.error(f"Session initialization error: {e}")
        logger.error(traceback.format_exc())
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)

@app.post("/query/{session_id}", response_model=QueryResponse)
async def query_session(session_id: str, query: QueryModel):
    """Query an existing RAG session"""
    try:
       
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Oturum bulunamadı")
            
        session = sessions[session_id]
        
       
        if session["status"] != "ready":
            raise HTTPException(status_code=400, detail=f"Oturum hazır değil: {session['status']}")
        
       
        file_path = session["file_path"]
        if not os.path.exists(file_path):
            logger.error(f"File no longer exists: {file_path}")
            raise HTTPException(status_code=500, detail="Yüklenen dosya artık mevcut değil.")
        
        logger.info(f"Processing query for session {session_id}: {query.query}")
        
        
        config = {
            "configurable": {
                "thread_id": session["thread_id"]
            }
        }
        
      
        last_state = session.get("last_checkpoint", {})
        
       
        inputs = {
            "file_path": file_path,
            "collection": last_state.get("collection"),
            "messages": last_state.get("messages", []) + [
                HumanMessage(content=query.query)
            ]
        }
        
        
        result = rag_graph.invoke(inputs, config)
        
     
        session["last_checkpoint"] = result
        session["last_activity"] = time.time()
        
        
        answer = result.get("answer", "Üzgünüm, bir cevap oluşturulamadı.")
        
        logger.info(f"Query processed successfully for session {session_id}")
        
        return QueryResponse(
            session_id=session_id,
            query=query.query,
            answer=answer
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Sorgu işlenirken hata oluştu: {str(e)}")