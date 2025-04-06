import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from supabase import create_client, Client
from groq import Groq

load_dotenv(dotenv_path=".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

API_KEYS = [os.getenv("GROQ_API_KEY")]
api_index = 0
client = Groq(api_key=API_KEYS[api_index])
MODEL_NAME = "llama-3.3-70b-versatile"

guidelines = os.getenv('TEST_MODE_GUIDELINES')
data_index = None  
data_src = None  

protectai_client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("PROTECTAI_API_KEY"),
)

PROMPT_INJECTION_FLAG = os.getenv("PROMPT_INJECTION_FLAG_PROMPT")