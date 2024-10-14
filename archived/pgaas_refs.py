import streamlit as st
import requests
import json
import os
import csv
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pytesseract
from PIL import Image

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
                    texts.append((page.extract_text(), filename))
        elif filename.endswith(('.txt', '.md')):
            with open(filepath, 'r', encoding='utf-8') as file:
                texts.append((file.read(), filename))
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image)
            texts.append((text, filename))
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_with_filenames = [(chunk, filename) for text, filename in texts for chunk in text_splitter.split_text(text)]
    return chunks_with_filenames

# Load document chunks at startup
document_chunks_with_filenames = load_documents()

def initialize_log_files():
    csv_file = os.path.join(script_dir, "chat_logs.csv")
    json_file = os.path.join(script_dir, "chat_logs.json")
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'date', 'time', 'question', 'response', 'doc_files'])
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

def update_chat_logs(user_name, question, response, doc_files, csv_file, json_file):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_name, date, time, question, response, doc_files])
    
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
            "response": response,
            "doc_files": doc_files
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
            if row[0] == user_name:  # Ensure this matches the correct index for user name
                history.append({
                    "name": row[0],
                    "date": row[1],
                    "time": row[2],
                    "question": row[3],
                    "response": row[4],
                    "doc_files": row[5]
                })
    return history

# Initialize log files
csv_file, json_file = initialize_log_files()

def get_perplexity_response(prompt, api_key):
    relevant_chunks_with_filenames = [(chunk.lower(), filename) for chunk, filename in document_chunks_with_filenames if any(keyword in chunk.lower() for keyword in prompt.lower().split())]
    
    context_chunks_with_filenames = relevant_chunks_with_filenames[:3]  # Use top 3 relevant chunks
    
    context_chunks = [chunk for chunk, _ in context_chunks_with_filenames]
    context_filenames_list = [filename for _, filename in context_chunks_with_filenames]

    context_text = "\n".join(context_chunks)

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
            {"role": "user", "content": f"Context: {context_text}\n\nQuestion: {prompt}"}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"], context_filenames_list
        else:
            return "Error: Unexpected response format from API.", context_filenames_list
    
    except requests.RequestException as e:
        st.error(f"API request failed: {e}")
        return f"Error: API request failed - {str(e)}", context_filenames_list

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
.chat-entry {
    margin-bottom: 10px;
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
       Welcome to the Patrick Geddes conversational AI. Here you can explore ideas and insights from one of the pioneers of urban planning.
       """)

# Input section for user queries
user_name_input = st.text_input("Enter your name:")
prompt_input = st.text_area("Ask Patrick Geddes a question:")

if st.button('Submit'):
    if user_name_input and prompt_input:
        response_content, used_filenames_list = get_perplexity_response(prompt_input.strip(), PERPLEXITY_API_KEY.strip())

        update_chat_logs(user_name=user_name_input.strip(),
                         question=prompt_input.strip(),
                         response=response_content,
                         doc_files="; ".join(set(used_filenames_list)),
                         csv_file=csv_file,
                         json_file=json_file)

        # Display latest response immediately after submission
        st.markdown(f"""
<div style="color: red;">
<strong>{user_name_input.strip()}</strong> - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
<div style="color: red;">
<strong>Question:</strong> {prompt_input.strip()}
</div>
<br>
<div>
<strong>Response:</strong> {response_content}
</div>
<div>
<strong>Doc Files:</strong> {"; ".join(set(used_filenames_list))}
</div>
<hr>
""", unsafe_allow_html=True)

# Display Chat History
if st.button('Show Chat History'):
    chat_history = get_chat_history(user_name_input.strip(), csv_file)
    for entry in chat_history:
        st.markdown(f"""
<div style="color: red;">
<strong>{entry['name']}</strong> - {entry['date']} {entry['time']}
</div>
<div style="color: red;">
<strong>Question:</strong> {entry['question']}
</div>
<br>
<div>
<strong>Response:</strong> {entry['response']}
</div>
<div>
<strong>Doc Files:</strong> {entry['doc_files']}
</div>
<hr>
""", unsafe_allow_html=True)

