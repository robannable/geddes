import streamlit as st
import requests
import json
import os
import csv
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full paths to the files
prompts_dir = os.path.join(script_dir, 'prompts')
about_file_path = os.path.join(script_dir, 'about.txt')
images_dir = os.path.join(script_dir, 'images')

# Try to get the API key from config.py, if it fails, look for it in Streamlit secrets
try:
    from config import PERPLEXITY_API_KEY
except ImportError:
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY")

def get_committee_prompt():
    prompt_file_path = os.path.join(prompts_dir, 'committee_prompt.txt')
    try:
        with open(prompt_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        st.warning(f"'{prompt_file_path}' not found. Using default prompt.")
        return ("You are a committee consisting of Patrick Geddes, Colin Ward, Christopher Alexander, "
                "Fritz Zwicky, and Ursula K. Le Guin. You will collaboratively respond to all queries "
                "through respectful dialogue and debate, each bringing your unique expertise and perspectives. "
                "Your goal is to provide a comprehensive and well-rounded answer, concluding with a summary "
                "from the member best suited to encapsulate the discussion.")

def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True  # Contains HTML
    except FileNotFoundError:
        st.warning(f"'{about_file_path}' not found. Using default about info.")
        return ("This application leverages Perplexity AI to simulate a committee of historical figures "
                "engaging in dialogue and debate to answer your questions. Experience a multifaceted "
                "perspective on various topics through the combined wisdom of experts in urban planning, "
                "anarchism, architecture, astrophysics, and literature."), False

def load_documents(directory='documents'):
    texts = []
    for filename in os.listdir(os.path.join(script_dir, directory)):
        filepath = os.path.join(script_dir, directory, filename)
        if filename.endswith('.pdf'):
            with open(filepath, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    texts.append(page.extract_text())
        elif filename.endswith(('.txt', '.md')):
            with open(filepath, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text('\n'.join(texts))
    return chunks

# Load document chunks at startup
document_chunks = load_documents()

def initialize_log_files():
    csv_file = os.path.join(script_dir, "chat_logs_committee.csv")
    json_file = os.path.join(script_dir, "chat_logs_committee.json")
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'date', 'time', 'question', 'response'])
    
    if not os.path.exists(json_file):
        with open(json_file, 'w') as f:
            json.dump([], f)
    else:
        with open(json_file, 'r+') as f:
            try:
                logs = json.load(f)
                if not isinstance(logs, list):
                    logs = []
            except json.JSONDecodeError:
                logs = []
            f.seek(0)
            json.dump(logs, f)
            f.truncate()
    
    return csv_file, json_file


def update_chat_logs(user_name, question, response, csv_file, json_file):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_name, date, time, question, response])
    with open(json_file, 'r+') as f:
        try:
            logs = json.load(f)
            if not isinstance(logs, list):
                logs = []
        except json.JSONDecodeError:
            logs = []
        logs.append({
            "name": user_name,
            "date": date,
            "time": time,
            "question": question,
            "response": response
        })
        f.seek(0)
        json.dump(logs, f, indent=4)
        f.truncate()

def get_chat_history(user_name, csv_file):
    history = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header row
        for row in reader:
            if row[0] == user_name:
                history.append({
                    "date": row[1],
                    "time": row[2],
                    "question": row[3],
                    "response": row[4]
                })
    return history

# Initialize log files
csv_file, json_file = initialize_log_files()

def get_perplexity_response(prompt, api_key, document_chunks, committee_prompt):
    relevant_chunks = [chunk for chunk in document_chunks if any(keyword in chunk.lower() for keyword in prompt.lower().split())]
    context = "\n".join(relevant_chunks[:3])  # Use top 3 relevant chunks
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": committee_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            return "Error: Unexpected response format from API."
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return f"Error: API request failed - {str(e)}"

# Custom CSS for improved visibility, dark theme compatibility, and proper formatting
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;
        padding: 20px;
        border-radius: 10px;
    }
    .chat-entry {
        margin-bottom: 20px;
    }
    .chat-name {
        font-weight: bold;
        font-size: 1.1em;
    }
    .chat-timestamp {
        font-style: italic;
        color: #555555;
        font-size: 0.9em;
    }
    .chat-question {
        margin-top: 5px;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .chat-response {
        margin-bottom: 10px;
    }
    .divider {
        border-top: 1px solid #cccccc;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit UI
st.title("Committee Debate Chat")

# Sidebar for About information
about_content, contains_html = get_about_info()
st.sidebar.header("About")
if contains_html:
    st.sidebar.markdown(about_content, unsafe_allow_html=True)
else:
    st.sidebar.info(about_content)

# Introduction section
st.markdown("### Enter your name and question to engage with the committee. Your queries will be addressed through a dynamic debate among the committee members.")

# User input
user_name = st.text_input("Enter your name")
user_question = st.text_area("Enter your question")

if st.button("Send") and user_name and user_question:
    committee_prompt = get_committee_prompt()
    response = get_perplexity_response(user_question, PERPLEXITY_API_KEY, document_chunks, committee_prompt)
    st.markdown("### Response")
    st.markdown(f"**Committee Members:**")
    st.markdown(response)
    update_chat_logs(user_name, user_question, response, csv_file, json_file)

# Chat History Toggle Button
if st.button("Show Chat History") and user_name:
    st.markdown("### Chat History")
    history = get_chat_history(user_name, csv_file)
    if history:
        for entry in history:
            st.markdown("---")
            st.markdown(f"**Question:** {entry['question']}")
            st.markdown(f"**Response:** {entry['response']}")
    else:
        st.info("No chat history available.")

