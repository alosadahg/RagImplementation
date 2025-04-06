from typing import Any, List
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import supabase

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

def find_relevant_src(index, data_src: pd.DataFrame, user_query: str) -> List[Any]:
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
