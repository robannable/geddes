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

# Try to get the API key from config.py, if it fails, look for it in Streamlit secrets
try:
    from config import PERPLEXITY_API_KEY
except ImportError:
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY")

def get_patrick_prompt():
    prompt_file_path = os.path.join(prompts_dir, 'patrick_geddes_prompt.txt')
    try:
        with open(prompt_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        st.warning(f"'{prompt_file_path}' not found. Using default prompt.")
        return "You are Patrick Geddes, a Scottish biologist, sociologist, and town planner..."

def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True  # Contains HTML
    except FileNotFoundError:
        st.warning(f"'{about_file_path}' not found. Using default about info.")
        return "This app uses Perplexity AI to simulate a conversation with Patrick Geddes...", False

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
    csv_file = os.path.join(script_dir, "chat_logs.csv")
    json_file = os.path.join(script_dir, "chat_logs.json")
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

def get_perplexity_response(prompt, api_key, document_chunks):
    relevant_chunks = [chunk for chunk in document_chunks if any(keyword in chunk.lower() for keyword in prompt.lower().split())]
    context = "\n".join(relevant_chunks[:3])  # Use top 3 relevant chunks
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    character_prompt = get_patrick_prompt()
    data = {
        "model": "llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": character_prompt},
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

# Custom CSS for improved visibility and design consistency
st.markdown("""
<style>
body {
    font-family: 'Arial', sans-serif;
}
.stApp {
    background-color: #ffffff;
}
h1 {
    font-size: 2em;
}
input[type="text"], textarea {
    font-size: 1.1em;
}
.sidebar .sidebar-content {
    background-color: #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Chat with Patrick Geddes")

# Sidebar for About information
about_content, contains_html = get_about_info()
st.sidebar.header("About")
if contains_html:
    st.sidebar.markdown(about_content, unsafe_allow_html=True)
else:
    st.sidebar.info(about_content)

# Introduction section with image and personal introduction
col1, col2 = st.columns([0.8, 3.2])
with col1:
    try:
        st.image("images/patrick_geddes.jpg", width=130)
    except Exception as e:
        st.write("Image not available")
with col2:
    st.markdown("""
    <div class="intro-text">
Greetings, dear inquirer! I am Patrick Geddes, a man of many hats - biologist, sociologist, geographer, and yes, a bit of a revolutionary in the realm of town planning, if I do say so myself.<br><br>Now, my eager student, what's your name? And more importantly, what burning question about our shared world shall we explore together? Remember, "By leaves we live" - so let your curiosity bloom and ask away!
    <div>
""", unsafe_allow_html=True)

# User input section
st.markdown("### Enter your name and question to start the conversation.")
user_name = st.text_input("Enter your name", key="name_input", help="Type your name here.")
user_question = st.text_area("Enter your question", key="question_input", help="Type your question here.")

if st.button("Send") and user_name and user_question:
    response = get_perplexity_response(user_question, PERPLEXITY_API_KEY, document_chunks)
    st.markdown("### Response")
    st.markdown(f"**Patrick Geddes**: {response}")
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
