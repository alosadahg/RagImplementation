import os
import re
import random
import asyncio
import json
import uuid
from typing import Optional, List, Any

from contextlib import asynccontextmanager
import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=".env")

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

api_keys = [os.getenv("GROQ_API_KEY")]
api_index = 0
client = Groq(api_key=api_keys[api_index])
model_name = "llama-3.3-70b-versatile"

data_index = None
data_src = None

def load_from_bucket(file_name: str) -> str:
    """
    Download a file (like .index or .csv) from Supabase storage to the local filesystem.
    Returns the local path where the file is saved.
    """
    with open(file_name, "wb+") as f:
        response = supabase.storage.from_("rag").download(file_name)
        f.write(response)
    return file_name

def load_embeddings() -> faiss.Index:
    """
    Load the FAISS index from the local .index file.
    """
    index_path = load_from_bucket("course_embeddings_v3.index")
    return faiss.read_index(index_path)

def extract_filtered_json_data(data: pd.DataFrame, matched_keys: List[int]) -> List[Any]:
    """
    Given the DataFrame and a list of matched row indices,
    group them by topic/lesson and produce a structured JSON output.
    """
    filtered_data = data.iloc[matched_keys, :]

    grouped_json = (
        filtered_data.groupby(["topic", "lesson_title"], group_keys=False)
        .apply(
            lambda x: [
                list(x["course_title"].unique()),
                list(x["language"].unique()),
                x[["problem_title", "difficulty", "type"]]
                .drop_duplicates()
                .to_dict(orient="records"),
            ],
            include_groups=False,
        )
        .reset_index()
    )

    grouped_json.columns = ["topic", "lesson_title", "data"]

    final_output = []
    for _, row in grouped_json.iterrows():
        final_output.append({
            "supplementary_courses": row["data"][0],
            "topic": row["topic"],
            "lesson_title": row["lesson_title"],
            "practice_problems": row["data"][2],
            "languages": row["data"][1],
        })
    return final_output

def find_relevant_src(index: faiss.Index, data_src: pd.DataFrame, user_query: str) -> List[Any]:
    """
    Given the user query, encode with SentenceTransformer,
    search in FAISS index, and return relevant JSON data from data_src.
    """
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embeddings = embedding_model.encode([user_query])

    k = 10
    distances, indices = index.search(query_embeddings, k)
    # Filter out only matches with distance < 1
    good_results = pd.DataFrame([
        (idx, dist) for idx, dist in zip(indices[0], distances[0]) if dist < 1
    ])

    related_data = []
    if len(good_results) > 0:
        matched_keys = good_results[0].tolist()
        extracted_data = extract_filtered_json_data(data_src, matched_keys)
        related_data.extend(extracted_data)

    return related_data

async def call_api_with_retry(messages: list, max_retries: int = 5) -> str:
    """
    Call Groq LLM with a list of messages. Retry on RateLimitError, rotating API keys if needed.
    """
    global api_index, client
    retries = 0

    # Limit the system to the last ~7 messages to avoid too-long context
    messages = messages[-7:]

    while retries < max_retries:
        try:
            output = client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_tokens=1024,
                stream=True,
            )
            response = ""
            for chunk in output:
                content = chunk.choices[0].delta.content
                if content:
                    response += content
            return response
        except RateLimitError as e:
            error_msg = str(e)
            # rotate to next api key if you have multiple
            api_index = (api_index + 1) % len(api_keys)
            client = Groq(api_key=api_keys[api_index])

            # parse the wait time or just do an exponential backoff
            match_wait = re.search(r"Please try again in ([\d.]+)s", error_msg)
            if match_wait:
                wait_time = float(match_wait.group(1))
            else:
                wait_time = (2 ** retries) + random.uniform(0, 1)

            await asyncio.sleep(wait_time)
            retries += 1

    return "I'm currently experiencing high traffic. Please try again in a moment."

async def save_session_to_supabase(session_id, messages):
    """
    Save the entire conversation to Supabase (table: session_history).
    """
    data = {
        "session_id": str(session_id),
        "messages": json.dumps(messages, indent=4)
    }
    await asyncio.to_thread(lambda: supabase.table("session_history").upsert(data).execute())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: load resources
    global data_index, data_src
    print("[Startup] Loading FAISS index and CSV from Supabase...")
    data_index = load_embeddings()
    data_src = pd.read_csv(load_from_bucket("codechum_src.csv"))
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

SESSION_CACHE = {}

class AskRequest(BaseModel):
    """
    The shape of JSON that the client must send to /ask.
    """
    prompt: str = Field(..., description="User's query or message.")
    session_id: Optional[str] = Field(None, description="Unique session ID if you want to track conversation across requests.")

class AskResponse(BaseModel):
    """
    The shape of JSON we return. 
    """
    session_id: str
    response: str

@app.get("/")
def home():
    return {"message": "FastAPI server is up. Try POST /ask to chat."}

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """
    Given a user prompt and optional session ID, 
    1) find relevant data from FAISS,
    2) call the LLM with that data,
    3) store the conversation in Supabase,
    4) return the assistant's response.
    """

    # 1) Generate or retrieve session_id
    if not request.session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.session_id

    # 2) Retrieve or initialize session history from our in-memory cache
    if session_id not in SESSION_CACHE:
        # Add a system message as first in conversation if you want
        SESSION_CACHE[session_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    session_history = SESSION_CACHE[session_id]

    # 3) The user's message
    user_prompt_message = {
        "role": "user",
        "content": request.prompt
    }
    session_history.append(user_prompt_message)

    # 4) Find relevant data
    relevant_data = find_relevant_src(data_index, data_src, request.prompt)
    # If there's relevant data, inject it as a system message
    if relevant_data:
        data_str = json.dumps(relevant_data, indent=4)
        session_history.append({
            "role": "system",
            "content": f"Use this data from Codechum for suggestions:\n{data_str}"
        })

    # 5) Call the LLM
    assistant_reply = await call_api_with_retry(session_history)
    # add the assistant's reply to the conversation
    session_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    # 6) Save session to Supabase if you want (optional)
    save_session_to_supabase(session_id, session_history)

    # 7) Store updated conversation in our session cache
    SESSION_CACHE[session_id] = session_history

    # 8) Return the response to the client
    return AskResponse(session_id=session_id, response=assistant_reply)