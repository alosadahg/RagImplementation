from dotenv import load_dotenv
from groq import Groq
import os
import streamlit as st
import re

load_dotenv(dotenv_path='.env')

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

model = "llama-3.3-70b-versatile"

MAX_HISTORY = 20

def summarize_history():
    if len(st.session_state.messages) > MAX_HISTORY:
        summary = client.chat.completions.create(
            messages=[{"role": "system", "content": "Summarize the following conversation: "}] 
                      + st.session_state.messages[:-MAX_HISTORY],
            model=model,
            max_tokens=512,
            stream=True
        )

        response = ""
        for chunk in summary:
            content = chunk.choices[0].delta.content
            if content != None: 
                response += content

        return [{"role": "system", "content": response}] + st.session_state.messages[-MAX_HISTORY:]
    return st.session_state.messages

def read_pseudocode_syntax():
    with open('./pseudocode/syntax.txt', 'r') as f:
        content = f.read()
        return content

def generate_response():
    output = client.chat.completions.create(
        messages=st.session_state.messages,
        model=model,
        max_tokens=1024,
        stream=True
    )

    response = ""
    for chunk in output:
        content = chunk.choices[0].delta.content
        if content != None: 
            response += content
    
    print(response) 
    
    if len(st.session_state.messages) > 1:
        with st.chat_message("assistant"):
            display_text(response)  

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

def display_text(response):
    segments = re.split(r'(```.*?```)', response, flags=re.DOTALL)
    for segment in segments:
        print("segment:" + segment)
        if segment.startswith("```") and segment.endswith("```"):
            st.code(segment[3:-3])
        else:
            break
    st.markdown(response)

st.title("Test Mode Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system","content": "Greet the user"}]
    generate_response()

for message in st.session_state.messages:
    if message["role"] != "system" and message["role"] != "tool":
        with st.chat_message(message["role"]):
            response = message["content"]
            display_text(response)  

if prompt := st.chat_input("Ask something"):
    with st.chat_message("user"):
        display_text(prompt) 
    print(os.getenv('TEST_MODE_GUIDELINES'))
    st.session_state.messages.append({"role": "system", "content": os.getenv('TEST_MODE_GUIDELINES')})
    # st.session_state.messages.append({"role": "tool", "content": f"When writing pseudocode, use this syntax {read_pseudocode_syntax()}", "tool_call_id": "pseudocode_syntax"})
    st.session_state.messages.append({"role": "user", "content": prompt})
    generate_response()

