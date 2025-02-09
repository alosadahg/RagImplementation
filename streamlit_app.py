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

load_dotenv(dotenv_path='.env')

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')
supabase: Client = create_client(url, key)gi

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
    course_index = faiss.read_index(load_from_bucket('course-embeddings.index'))
    problem_index = faiss.read_index(load_from_bucket('problem-embeddings.index'))
    return course_index, problem_index

def find_relevant_src(index, data_src, user_query):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embeddings = embedding_model.encode([user_query])
    k = 2
    distances, indices = index.search(query_embeddings, k)
    related_data = []
    for rank, idx in enumerate(indices[0], start=1):
        # print(distances[0][rank-1])
        if distances[0][rank-1] > 0.9:
            related_data.append(data_src[idx])
    return related_data

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
    # TODO: Add caching to minimize token limit error i guess
    # cache_key = tuple(msg["content"] for msg in session_history if msg["role"] == "user")
    # if cache_key in cache:
    #     response = cache[cache_key]
    # else:
    response = await call_api_with_retry(st.session_state.messages)
    # if response:
    #     cache[cache_key] = response
    
    result = {"role": "assistant", "content": response}
    if len(st.session_state.messages) > 1:
        with st.chat_message("assistant"):
            display_text(response) 
    session_history.append(result)
    st.session_state.messages.append(result)
     

st.title("Learning Assistant (with CodeChum)")

if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = {"role": "system", "content": "Greet the user"}
    st.session_state.messages.append(system_prompt)
    session_history.append(system_prompt)
    asyncio.run(generate_response())

course_index, problem_index = load_embeddings()
course_src = np.load(load_from_bucket('courses.npy'))
problems_src = np.load(load_from_bucket('problems.npy'))

for message in st.session_state.messages:
    if message["role"] != "system" and message["role"] != "tool":
        with st.chat_message(message["role"]):
            display_text(message["content"])

if prompt := st.chat_input("Ask something"):
    with st.chat_message("user"):
        display_text(prompt)
    
    relevant_courses = find_relevant_src(course_index, course_src, prompt)
    relevant_problems = find_relevant_src(problem_index, problems_src, prompt)
    relevant_data = relevant_courses + relevant_problems
    user_prompt = {"role": "user", "content": prompt}
    session_history.append(user_prompt)
    st.session_state.messages.append(user_prompt)
    
    rel_data = ""
    if len(relevant_courses) > 0:
        rel_data += f"Include this suggested courses (format as a list) from Codechum to guide the user: {"\n".join(relevant_courses)}\n"
    if len(relevant_problems) > 0:
        rel_data += f"Include this suggested problems (format as a list) from Codechum to guide the user: {"\n".join(relevant_problems)}\n"
    if not rel_data:
        rel_data = "State that you are not using sources from Codechum."
    st.session_state.messages.append({"role": "system", "content": os.getenv('TEST_MODE_GUIDELINES') + rel_data})
    
    asyncio.run(generate_response())
    for msg in st.session_state.messages:
        if msg['role'] == 'system':
            st.session_state.messages.remove(msg)