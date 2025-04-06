import asyncio
import json
import random
import re
import time
import uuid
from groq import RateLimitError, Groq
from services.session_service import save_session_to_supabase
from config import *
from models import AskRequest, AskResponse
from services.relevant_data_service import *

SESSION_CACHE = {}

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
                model=MODEL_NAME,
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
            api_index = (api_index + 1) % len(API_KEYS)
            client = Groq(api_key=API_KEYS[api_index])

            # parse the wait time or just do an exponential backoff
            match_wait = re.search(r"Please try again in ([\d.]+)s", error_msg)
            if match_wait:
                wait_time = float(match_wait.group(1))
            else:
                wait_time = (2 ** retries) + random.uniform(0, 1)

            await asyncio.sleep(wait_time)
            retries += 1

    return "I'm currently experiencing high traffic. Please try again in a moment."

def is_injection(text, threshold=0.95):
    classification_result = protectai_client.text_classification(
        text=text,
        model="protectai/deberta-v3-base-prompt-injection-v2",
    )
    print("Classification result:", classification_result)
    for result in classification_result:
        if result.label.upper() == "INJECTION" and result.score >= threshold:
            return True
    return False

async def ask(request: AskRequest, data_index, data_src) -> str:
    
    if not request.session_id:
        session_id = str(uuid.uuid4())
    else:
        session_id = request.session_id

    if session_id not in SESSION_CACHE:
        SESSION_CACHE[session_id] = [] 
    
    session_history = SESSION_CACHE[session_id] 

    if guidelines:
        session_history.append({
            "role": "system",
            "content": guidelines
        })

    if is_injection(request.prompt):
        request.prompt = f"{PROMPT_INJECTION_FLAG} {request.prompt}"

    user_prompt_message = {
        "role": "user",
        "content": request.prompt
    }
    
    session_history.append(user_prompt_message)
    session_history = process_relevant_data(data_index, data_src, request.prompt, session_history)

    start_time = time.time()
    assistant_reply = await call_api_with_retry(session_history)
    end_time = time.time()
    elapsed_time = end_time - start_time

    session_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    await save_session_to_supabase(session_id, session_history)
    SESSION_CACHE[session_id] = session_history

    return AskResponse(
        session_id=session_id,
        prompt=request.prompt,
        response=assistant_reply,
        response_time=elapsed_time
    )


