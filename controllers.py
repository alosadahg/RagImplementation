from fastapi import FastAPI
from models import AskRequest, AskResponse
from contextlib import asynccontextmanager
import pandas as pd
from services.relevant_data_service import *
from config import data_src, data_index
from services.session_service import *
from services.llm_service import *

from fastapi import Depends

def get_data_index():
    global data_index
    if data_index is None:
        raise ValueError("Index not loaded yet.")
    return data_index

def get_data_src():
    global data_src
    if data_src is None:
        raise ValueError("Data source not loaded yet.")
    return data_src


@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_index, data_src
    # Startup logic: load resources
    print("[Startup] Loading FAISS index and CSV from Supabase...")
    data_index = load_index_files()
    data_src = load_csv_files()
    print("[Startup] Done loading resources.")
    
    yield  # Application is running

    # Shutdown logic (if any)
    print("[Shutdown] Cleaning up resources...")

app = FastAPI(
    title="Learning Assistant (with CodeChum) - FastAPI",
    description="Example API that uses Supabase, FAISS, and Groq to answer user queries.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {"message": "FastAPI server is up. Try POST /ask to chat."}

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest, data_index=Depends(get_data_index), data_src=Depends(get_data_src)):
    """
    Given a user prompt and optional session ID, 
    1) find relevant data from FAISS,
    2) call the LLM with that data,
    3) store the conversation in Supabase,
    4) return the assistant's response.
    """
    return await ask(request, data_index, data_src)

    