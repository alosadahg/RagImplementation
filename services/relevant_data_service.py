from typing import Any, List
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import supabase
import json

def load_from_bucket(file_name: str):
    """
    Download a file (like .index or .csv) from Supabase storage to the local filesystem.
    Returns the local path where the file is saved.
    """
    with open(file_name, "wb+") as f:
        response = supabase.storage.from_("rag").download(file_name)
        f.write(response)
    return file_name

def load_embeddings(file):
    """
    Load the FAISS index from the local .index file.
    """
    index_path = load_from_bucket(file)
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

def extract_from_np(data_src, indices):
    related_data = []
    for index in indices:
        data_list = data_src["chunk"].tolist() 
        related_data.append(data_list[index])

    return related_data

def find_relevant_src(index, data_src: pd.DataFrame, type, user_query: str) -> List[Any]:
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
        if type == "json":
            extracted_data = extract_filtered_json_data(data_src, good_results[0].tolist())
        else:
            extracted_data = extract_from_np(data_src, good_results[0].tolist())
        related_data.extend(extracted_data)

    return related_data

def load_index_files():
    """Loads embedding index files into session state."""
    index_files_session = {}
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
        if key not in index_files_session:
            print(f"Initializing {key}")
            index_files_session[key] = load_embeddings(file)
    return index_files_session

def load_csv_files():
    """Loads CSV source files into session state."""
    csv_files_session = {}
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
        if key not in csv_files_session:
            print(f"Initializing {key}")
            csv_files_session[key] = pd.read_csv(load_from_bucket(file))
    return csv_files_session

def append_relevant_data(label, data, messages):
    """Appends relevant data to the session state messages."""
    if data:
        relevant_data_str = json.dumps(data, indent=4)
        messages.append({
            "role": "system",
            "content": f"{label}:\n{relevant_data_str}"
        })
    return messages

def process_relevant_data(indexes, src, prompt, messages):
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
    print(messages)
    for label, (index_key, src_key, format_type) in datasets.items():
        relevant_data = find_relevant_src(
            indexes[index_key], 
            src[src_key], 
            format_type, 
            prompt
        )
        messages = append_relevant_data(f"Include this data from {label}", relevant_data, messages)
    print(messages)
    return messages
