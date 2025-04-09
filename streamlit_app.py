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
from huggingface_hub import InferenceClient
import torch
import pandas as pd
import json
import uuid
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv(dotenv_path=".env")

torch.classes.__path__ = []

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

api_key = st.secrets["GROQ_API_KEY"]
api_index = 0
client = Groq(api_key=api_key)
model = "llama-3.3-70b-versatile"

session_history = []
cache = {}

def load_from_bucket(file_name):
    with open(f"{file_name}", "wb+") as f:
        response = supabase.storage.from_("rag").download(f"{file_name}")
        f.write(response)
    return file_name


def load_embeddings(file):
    data_src_index = faiss.read_index(load_from_bucket(file))
    return data_src_index


@st.cache_data
def load_embeddings_cached(file):
    """Loads embedding index file (cached)."""
    return load_embeddings(file)  # Assuming load_embeddings() is defined elsewhere

@st.cache_data
def load_data_cached(file):
    """Loads CSV source file (cached)."""
    return pd.read_csv(load_from_bucket(file))  # Assuming load_from_bucket() is defined elsewhere

def load_index_files():
    """Loads embedding index files into session state if not already loaded."""
    index_files = {
        "data_index": "course_embeddings_v3.index",
        "bst_index": "bst_embeddings.index",
        "advanced_trees_index": "advanced_trees_embeddings.index",
        "algorithms_index": "analysis_of_algorithms_embeddings.index",
        "hash_index": "hash_tables_embeddings.index",
        "sorting_index": "sorting_algorithms_embeddings.index",
        "memory_index": "stack_vs_heap_embeddings.index",
    }
    for key, file in index_files.items():
        if key not in st.session_state:
            print(f"Initializing {key}")
            st.session_state[key] = load_embeddings_cached(file)

def load_csv_files():
    """Loads CSV source files into session state if not already loaded."""
    csv_files = {
        "data_src": "codechum_src.csv",
        "bst_src": "bst_src.csv",
        "advanced_trees_src": "advanced_trees_src.csv",
        "algorithms_src": "analysis_of_algorithms_src.csv",
        "hash_src": "hash_tables_src.csv",
        "sorting_src": "sorting_algorithms_src.csv",
        "memory_src": "stack_vs_heap_src.csv",
    }
    for key, file in csv_files.items():
        if key not in st.session_state:
            print(f"Initializing {key}")
            st.session_state[key] = load_data_cached(file)

def append_relevant_data(label, data):
    """Appends relevant data to the session state messages."""
    if data:
        relevant_data_str = json.dumps(data, indent=4)
        st.session_state.messages.append({
            "role": "system",
            "content": f"{label}:\n{relevant_data_str}"
        })

def process_relevant_data(prompt):
    """Finds relevant data for each index-source pair and appends it to session messages."""
    datasets = {
        "Codechum": ("data_index", "data_src", "json"),
        "the lesson on CS244 BST": ("bst_index", "bst_src", "list"),
        "the lesson on CS244 Advanced Trees": ("advanced_trees_index", "advanced_trees_src", "list"),
        "the lesson on CS244 Analysis of Algorithms": ("algorithms_index", "algorithms_src", "list"),
        "the lesson on CS244 Hash Tables": ("hash_index", "hash_src", "list"),
        "the lesson on CS244 Sorting Algorithms": ("sorting_index", "sorting_src", "list"),
        "the lesson on CS244 Stack vs Heap Memory": ("memory_index", "memory_src", "list"),
    }

    for label, (index_key, src_key, format_type) in datasets.items():
        relevant_data = find_relevant_src(
            st.session_state[index_key], 
            st.session_state[src_key], 
            format_type, 
            prompt
        )
        append_relevant_data(f"Include this data from {label}", relevant_data)

async def save_session_to_supabase(session_id, messages):
    data = {"session_id": str(session_id), "messages": json.dumps(messages, indent=4)}

    await asyncio.to_thread(
        lambda: supabase.table("session_history").upsert(data).execute()
    )


def extract_filtered_json_data(data, matched_keys):
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

def extract_from_np(data_src, indices):
    related_data = []
    for index in indices:
        data_list = data_src["chunk"].tolist() 
        related_data.append(data_list[index])

    return related_data

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_cache():
    tokenizer = AutoTokenizer.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
    model = AutoModelForSequenceClassification.from_pretrained("protectai/deberta-v3-base-prompt-injection-v2")
    return model, tokenizer

def find_relevant_src(index, data_src, type, user_query):
    embedding_model = load_model()
    query_embeddings = embedding_model.encode([user_query])

    k = 4
    distances, indices = index.search(query_embeddings, k)
    good_results = pd.DataFrame(
        [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if dist < 1]
    )
    # display(good_results)

    # for _, row in good_results.iterrows():
    #     display(df_data_src.iloc[row[0].astype(int)])

    related_data = []
    if len(good_results) > 0:
        if type == "json":
            extracted_data = extract_filtered_json_data(data_src, good_results[0].tolist())
        else:
            extracted_data = extract_from_np(data_src, good_results[0].tolist())
        related_data.extend(extracted_data)

    return related_data


async def call_api_with_retry(messages, max_retries=5):
    global api_index, client
    retries = 0
    messages = messages[-7:]
    while retries < max_retries:
        try:
            output = client.chat.completions.create(
                messages=messages, model=model, max_tokens=1024, stream=True
            )
            response = ""
            for chunk in output:
                content = chunk.choices[0].delta.content
                if content:
                    response += content
            return response
        except RateLimitError as e:
            error_msg = str(e)
            # api_index = (api_index + 1) % len(api_keys)
            # client = Groq(api_key=api_keys[api_index])
            wait_time = (
                float(re.search(r"Please try again in ([\d.]+)s", error_msg).group(1))
                if re.search(r"Please try again in ([\d.]+)s", error_msg)
                else (2**retries) + random.uniform(0, 1)
            )
            await asyncio.sleep(wait_time)
            retries += 1
    return "I'm currently experiencing high traffic. Please try again in a moment."


def display_text(response):
    segments = re.split(r"(```.*?```)", response, flags=re.DOTALL)
    for segment in segments:
        if segment.startswith("```") and segment.endswith("```"):
            st.code(segment.strip("`\n"))
        else:
            st.markdown(segment)


async def generate_response():
    #
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
    if len(st.session_state.messages) > 2:
        await save_session_to_supabase(
            st.session_state.session_id, st.session_state.messages
        )


def is_injection(text, threshold=0.95):
    model, tokenizer = load_cache()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    injection_score = probabilities[0][1].item()  

    print("Injection Score:", injection_score)

    return injection_score >= threshold


st.title("Bruno - Your Purrfect Learning Companion")

if "session_id" not in st.session_state:
    session_id = uuid.uuid4()
    st.session_state.session_id = session_id
    print("Session ID:", st.session_state.session_id)

load_index_files()
load_csv_files()

if "messages" not in st.session_state:
    st.session_state.messages = []
    system_prompt = {"role": "system", "content": "Greet the user"}
    st.session_state.messages.append(system_prompt)
    session_history.append(system_prompt)
    asyncio.run(generate_response())

for message in st.session_state.messages:
    if message["role"] != "system" and message["role"] != "tool":
        with st.chat_message(message["role"]):
            display_text(message["content"])

if prompt := st.chat_input("Ask something"):
    with st.chat_message("user"):
        display_text(prompt)

    if is_injection(prompt):
        prompt = f"{st.secrets["PROMPT_INJECTION_FLAG_PROMPT"]} {prompt}"

    user_prompt = {"role": "user", "content": prompt}
    session_history.append(user_prompt)
    st.session_state.messages.append(user_prompt)
    st.session_state.messages.append(
        {"role": "system", "content": st.secrets["TEST_MODE_GUIDELINES"]}
    )
    process_relevant_data(prompt)
    # print(st.secrets['TEST_MODE_GUIDELINES'))
    asyncio.run(generate_response())
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            st.session_state.messages.remove(msg)
