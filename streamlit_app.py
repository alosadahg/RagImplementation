import os
import random
import asyncio
import re
import numpy as np
import faiss
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
import torch
import pandas as pd
import json
import uuid
import requests  # Import requests library for Lakera Guard API call

load_dotenv(dotenv_path='.env')

torch.classes.__path__ = []

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
lakera_guard_api_key = os.getenv('LAKERA_GUARD_API_KEY')  # API key for Lakera Guard
supabase: Client = create_client(url, key)

api_keys = [os.getenv('GROQ_API_KEY')]
api_index = 0
client = Groq(api_key=api_keys[api_index])
model = "llama-3.3-70b-versatile"

session_history = []
cache = {}

def load_from_bucket(file_name):
    with open(f"{file_name}", "wb+") as f:
        response = supabase.storage.from_("rag").download(f"{file_name}")
        f.write(response)
    return file_name

def load_embeddings():
    data_src_index = faiss.read_index(load_from_bucket('course_embeddings_v3.index'))
    return data_src_index

async def save_session_to_supabase(session_id, messages):
    data = {
        "session_id": str(session_id),
        "messages": json.dumps(messages, indent=4)
    }

    await asyncio.to_thread(lambda: supabase.table("session_history").upsert(data).execute())

def extract_filtered_json_data(data, matched_keys):
    filtered_data = data.iloc[matched_keys, :]

    grouped_json = (filtered_data.groupby(['topic', 'lesson_title'], group_keys=False)
        .apply(lambda x: [
            list(x['course_title'].unique()), 
            list(x['language'].unique()),  
            x[['problem_title', 'difficulty', 'type']].drop_duplicates().to_dict(orient='records')  
        ], include_groups=False)
    .reset_index())

    grouped_json.columns = ['topic', 'lesson_title', 'data']

    final_output = [
        {
            "supplementary_courses": row["data"][0], 
            "topic": row["topic"],
            "lesson_title": row["lesson_title"], 
            "practice_problems": row["data"][2],
            "languages": row["data"][1], 
        }
        for _, row in grouped_json.iterrows()
    ]
    return final_output

def find_relevant_src(index, data_src, user_query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embeddings = embedding_model.encode([user_query])

    k = 10
    distances, indices = index.search(query_embeddings, k)
    good_results = pd.DataFrame([(idx, dist) for idx, dist in zip(indices[0], distances[0]) if dist < 1])

    related_data = []
    if len(good_results) > 0:
        extracted_data = extract_filtered_json_data(data_src, good_results[0].tolist())
        related_data.extend(extracted_data)

    return related_data

def call_lakera_guard_api(user_message):
    """Function to send the user's message to Lakera Guard API."""
    headers = {
        "Authorization": f"Bearer {lakera_guard_api_key}",
    }
    payload = {
        "messages": [{"content": user_message, "role": "user"}]
    }
    response = requests.post(
        "https://api.lakera.ai/v2/guard",
        json=payload,
        headers=headers
    )
    if response.status_code == 200:
        return response.json()  # Returns the Lakera Guard response
    else:
        print(f"Error calling Lakera Guard API: {response.status_code}")
        return None

async def call_api_with_retry(messages, max_retries=5):
    global api_index, client
    retries = 0
    messages = messages[-7:]
    while retries < max_retries:
        try:
            output = client.chat.completions.create(
                messages=messages,
                model=model,
                max_tokens=1024,
                stream=True
            )
            response = ""
            for chunk in output:
                content = chunk.choices[0].delta.content
                if content:
                    response += content
            return response  
        except RateLimitError as e:
            error_msg = str(e)
            api_index = (api_index + 1) % len(api_keys) 
            client = Groq(api_key=api_keys[api_index])
            wait_time = float(re.search(r'Please try again in ([\d.]+)s', error_msg).group(1)) if re.search(r'Please try again in ([\d.]+)s', error_msg) else (2 ** retries) + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
            retries += 1
    return "I'm currently experiencing high traffic. Please try again in a moment."

def display_text(response):
    segments = re.split(r'(```.*?```)', response, flags=re.DOTALL)
    for segment in segments:
        if segment.startswith("```") and segment.endswith("```"):
            st.code(segment.strip("`\n")) 
        else:
            st.markdown(segment)

async def generate_response():
    response = await call_api_with_retry(st.session_state.messages)
    result = {"role": "assistant", "content": response}
    if len(st.session_state.messages) > 1:
        with st.chat_message("assistant"):
            display_text(response) 
    session_history.append(result)
    st.session_state.messages.append(result)
    await save_session_to_supabase(st.session_state.session_id, st.session_state.messages)

st.title("Bruno - Learning Assistant")

if "session_id" not in st.session_state:
    session_id = uuid.uuid4()
    st.session_state.session_id = session_id
    print("Session ID:", st.session_state.session_id)

if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = {"role": "system", "content": "Greet the user"}
    st.session_state.messages.append(system_prompt)
    session_history.append(system_prompt)
    asyncio.run(generate_response())

if "data_index" not in st.session_state:
    print("Initializing data index")
    st.session_state.data_index = load_embeddings()

if "data_src" not in st.session_state:
    print("Initializing data source")
    st.session_state.data_src = pd.read_csv(load_from_bucket('codechum_src.csv'))

data_index = st.session_state.data_index
data_src = st.session_state.data_src

for message in st.session_state.messages:
    if message["role"] != "system" and message["role"] != "tool":
        with st.chat_message(message["role"]):
            display_text(message["content"])

if prompt := st.chat_input("Ask something"):
    with st.chat_message("user"):
        display_text(prompt)
    
    # Call Lakera Guard to validate the input before processing
    lakera_guard_response = call_lakera_guard_api(prompt)
    if lakera_guard_response and lakera_guard_response.get("is_safe", True):
        relevant_data = find_relevant_src(data_index, data_src, prompt)
        user_prompt = {"role": "user", "content": prompt}
        session_history.append(user_prompt)
        st.session_state.messages.append(user_prompt)
        st.session_state.messages.append({"role": "system", "content": os.getenv('TEST_MODE_GUIDELINES')})
        if relevant_data:
            relevant_data_str = json.dumps(relevant_data, indent=4)
            st.session_state.messages.append({
                "role": "system",
                "content": "Include this data (have it in a list format) from Codechum for suggestions:\n" + relevant_data_str
            })
        asyncio.run(generate_response())
    else:
        st.session_state.messages.append({
            "role": "system",
            "content": "Warning: The input content may not be safe or valid."
        })
    
    for msg in st.session_state.messages:
        if msg['role'] == 'system':
            st.session_state.messages.remove(msg)
