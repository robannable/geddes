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
prompt_file_path = os.path.join(script_dir, 'patrick_prompt.txt')
about_file_path = os.path.join(script_dir, 'about.txt')

# Try to get the API key from config.py, if it fails, look for it in Streamlit secrets
try:
    from config import PERPLEXITY_API_KEY
except ImportError:
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY")

def get_patrick_prompt():
    try:
        with open(prompt_file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: '{prompt_file_path}' not found. Using default prompt.")
        return "You are Patrick Geddes, a Scottish biologist, sociologist, and town planner..."

def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True  # Return the content and a flag indicating it contains HTML
    except FileNotFoundError:
        print(f"Warning: '{about_file_path}' not found. Using default about info.")
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
                    f.seek(0)
                    json.dump(logs, f)
                    f.truncate()
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
        print(f"Full error details: {e}")
        print(f"Response content: {e.response.content if e.response else 'No response'}")
        return f"Error: API request failed - {str(e)}"

# Custom CSS for improved visibility, dark theme compatibility, and proper formatting
st.markdown("""
<style>
    body {
        font-family: Georgia, serif;
        background-color: #FF6B35;
        color: black;
    }
    .stTextInput > div > div > input {
        color: black;
        background-color: white;
        border: 2px solid #FF6B35;
        border-radius: 5px;
    }
    .stTextArea textarea {
        color: black !important;
        background-color: #FFE0D3 !important;
        border: 2px solid #FF6B35;
        border-radius: 5px;
        opacity: 1 !important;
    }
    .stButton > button {
        color: #FF6B35;
        background-color: white;
        border: 2px solid #FF6B35;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: bold;
        color: #FF6B35;
    }
    h2, h3 {
        font-family: Georgia, serif;
        font-weight: bold;
        color: black;
    }
    .stMarkdown {
        color: black;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    .intro-section > div {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .intro-text {
        padding-left: 0 !important;
        margin-left: -0.5rem !important;
        color: black;
    }
    /* Targeting the Streamlit column containing the text */
    .css-1l269bu {
        padding-left: 0 !important;
    }
    /* Dark theme adjustments */
    @media (prefers-color-scheme: dark) {
        body {
            color: white;
            background-color: #1E1E1E;
        }
        .stTextInput > div > div > input,
        .stTextArea textarea {
            color: white !important;
            background-color: #333333 !important;
            border-color: #FF6B35;
        }
        h2, h3, .stMarkdown, .intro-text {
            color: white;
        }
        .sidebar .sidebar-content {
            background-color: #2D2D2D;
        }
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Chat with Patrick")

# Introduction section with image
col1, col2 = st.columns([0.8, 3.2])
with col1:
    try:
        st.image("images/patrick_geddes.jpg", width=130, output_format="JPG")
    except Exception as e:
        st.write("Image not available")
        print(f"Error loading image: {e}")
with col2:
    st.markdown("""
    <div class="intro-text">
    Greetings, dear inquirer! I am Patrick Geddes, a man of many hats - biologist, sociologist, geographer, and yes, a bit of a revolutionary in the realm of town planning, if I do say so myself.<br><br>Now, my eager student, what's your name? And more importantly, what burning question about our shared world shall we explore together? Remember, "By leaves we live" - so let your curiosity bloom and ask away!
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""

# User name input
if not st.session_state.user_name:
    user_name = st.text_input("Please enter your name:", key="user_name_input")
    if user_name:
        st.session_state.user_name = user_name

# Check if API key is set and user has entered their name
if not PERPLEXITY_API_KEY:
    st.error("Please set your Perplexity API key in the config.py file or Streamlit secrets.")
elif st.session_state.user_name:
    # Chat interface
    user_input = st.text_input("Your question:", key="user_input")
    if st.button("Send"):
        if user_input:
            response = get_perplexity_response(user_input, PERPLEXITY_API_KEY, document_chunks)
            st.session_state.chat_history.append((user_input, response))
            # Log the conversation
            update_chat_logs(st.session_state.user_name, user_input, response, csv_file, json_file)

    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You (Question {i+1}):**")
        st.text_area("", value=question, height=50, disabled=True, key=f"q{i}")
        st.markdown(f"**Patrick Geddes (Answer {i+1}):**")
        st.text_area("", value=answer, height=250, disabled=True, key=f"a{i}")
        st.markdown("---")

    # Option to view chat history
    if st.button("View My Chat History"):
        history = get_chat_history(st.session_state.user_name, csv_file)
        st.write(history)

# Display information about the app
st.sidebar.header("About")
about_content, is_html = get_about_info()
if is_html:
    st.sidebar.markdown(about_content, unsafe_allow_html=True)
else:
    st.sidebar.info(about_content)


# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit and Perplexity AI")
