import streamlit as st
import requests
import json
import pygame
import os
import csv
import streamlit as st
import requests
import json
import pygame
import os
import csv
from datetime import datetime
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import html
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
from requests.exceptions import RequestException

# First, define the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

import logging
import time

# Set up minimal logging for document loading only
log_dir = os.path.join(script_dir, "debug_logs")
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.now().strftime("%d-%m-%Y")
log_file = os.path.join(log_dir, f"{current_date}_rag_loading.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - RAG Loading - %(message)s',
    encoding='utf-8'
)

# Initialize directories
sound_dir = os.path.join(script_dir, 'sounds')
prompts_dir = os.path.join(script_dir, 'prompts')
about_file_path = os.path.join(script_dir, 'about.txt')

# Initialize pygame for audio
pygame.mixer.init()

# Load sound file
ding_sound = pygame.mixer.Sound(os.path.join(sound_dir, 'ding2.wav'))

# Replace PerplexityAPIHandler with OllamaHandler
class OllamaHandler:
    def __init__(self, model_name="mistral", max_retries=3, timeout=30):
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = "http://localhost:11434/api"
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        return session

    def get_available_models(self):
        try:
            response = self.session.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models['models']]
            return []
        except:
            return []

    def make_request(self, data):
        headers = {"Content-Type": "application/json"}
        prompt = data["messages"][-1]["content"]  # Extract the prompt
        
        request_data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/generate",
                    headers=headers,
                    json=request_data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return {"choices": [{"message": {"content": response.json()["response"]}}]}
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)

# Modify the connection check function
def check_api_connection():
    """Check if we can connect to the Ollama server"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

@st.cache_data
def get_patrick_prompt():
    prompt_file_path = os.path.join(prompts_dir, 'patrick_geddes_prompt.txt')
    try:
        with open(prompt_file_path, 'r') as file:
            prompt = file.read().strip()
        prompt += "\n\nWhen responding to users, consider their name and potential gender implications. Avoid making assumptions based on stereotypes and strive for inclusive language. Adapt your language and examples to be appropriate for all users, regardless of their perceived gender."
        return prompt
    except FileNotFoundError:
        st.warning(f"'{prompt_file_path}' not found. Using default prompt.")
        return "You are Patrick Geddes, a Scottish biologist, sociologist, and town planner. When responding to users, consider their name and potential gender implications. Avoid making assumptions based on stereotypes and strive for inclusive language. Adapt your language and examples to be appropriate for all users, regardless of their perceived gender."

@st.cache_data
def get_about_info():
    try:
        with open(about_file_path, 'r') as file:
            return file.read().strip(), True  # Contains HTML
    except FileNotFoundError:
        st.warning(f"'{about_file_path}' not found. Using default about info.")
        return "This app uses Perplexity AI to simulate a conversation with Patrick Geddes...", False

@st.cache_data
def load_documents(directories=['documents', 'history', 'students']):
    total_start_time = time.time()
    texts = []
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    # Define system directories to ignore
    ignore_dirs = {
        '__pycache__',
        '.ipynb_checkpoints',
        '.git',
        'debug_logs',
        'logs',
        'sounds',
        'prompts',
        'images'
    }
    
    for directory in directories:
        dir_path = os.path.join(script_dir, directory)
        if os.path.exists(dir_path):
            dir_start_time = time.time()
            files_processed = 0
            
            for item in os.listdir(dir_path):
                # Skip if item is in ignored directories or is hidden
                if item in ignore_dirs or item.startswith('.'):
                    continue
                    
                filepath = os.path.join(dir_path, item)
                
                # Skip if it's a directory
                if os.path.isdir(filepath):
                    continue
                    
                # Skip files with today's date in the history folder
                if directory == 'history' and current_date in item:
                    continue
                
                file_start_time = time.time()
                
                try:
                    if item.endswith('.pdf'):
                        with open(filepath, 'rb') as file:
                            pdf_reader = PdfReader(file)
                            for page in pdf_reader.pages:
                                texts.append((page.extract_text(), item))
                    elif item.endswith(('.txt', '.md')):
                        with open(filepath, 'r', encoding='utf-8') as file:
                            texts.append((file.read(), item))
                    elif item.endswith(('.png', '.jpg', '.jpeg')):
                        image = Image.open(filepath)
                        text = pytesseract.image_to_string(image)
                        texts.append((text, item))
                        
                    file_time = time.time() - file_start_time
                    logging.info(f"Loaded {item} in {file_time:.2f} seconds")
                    files_processed += 1
                    
                except Exception as e:
                    logging.error(f"Failed to load {item}: {str(e)}")
            
            dir_time = time.time() - dir_start_time
            logging.info(f"Directory {directory}: processed {files_processed} files in {dir_time:.2f} seconds")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks_with_filenames = [(chunk, filename) for text, filename in texts for chunk in text_splitter.split_text(text)]
    
    total_time = time.time() - total_start_time
    logging.info(f"Total RAG loading completed in {total_time:.2f} seconds - Created {len(chunks_with_filenames)} chunks from {len(texts)} documents")
    
    return chunks_with_filenames


@st.cache_resource
def compute_tfidf_matrix(document_chunks):
    documents = [chunk for chunk, _ in document_chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Load document chunks and compute TF-IDF matrix at startup
document_chunks_with_filenames = load_documents(['documents', 'history', 'students'])
vectorizer, tfidf_matrix = compute_tfidf_matrix(document_chunks_with_filenames)

def initialize_log_files():
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y")
    csv_file = os.path.join(logs_dir, f"{current_date}_response_log.csv")
    json_file = os.path.join(logs_dir, f"{current_date}_response_log.json")

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'date', 'time', 'question', 'response', 'unique_files', 'chunk1_score', 'chunk2_score', 'chunk3_score'])

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

def write_markdown_history(user_name, question, response, csv_file):
    history_dir = os.path.join(script_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    current_date = datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    md_file = os.path.join(history_dir, f"{current_date}_conversation_history.md")
    
    with open(md_file, 'a', encoding='utf-8') as f:
        f.write(f"## Date: {current_date} | Time: {current_time}\n\n")
        f.write(f"### User: {user_name}\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"**Patrick Geddes:** {response}\n\n")
        f.write("---\n\n")

def update_chat_logs(user_name, question, response, unique_files, chunk_info, csv_file, json_file):
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    # Store raw data for logging
    raw_response = response
    # HTML escape only for web output (not for logging)
    encoded_response = html.escape(response)

    # Use raw data for unique files and chunk info
    unique_files_str = " - ".join(unique_files)

    # Parse chunk_info to extract scores and filenames (using raw data)
    chunk_scores = []
    for chunk in chunk_info:
        try:
            parts = chunk.split(', score: ')
            if len(parts) == 2:
                score = float(parts[1].strip(')'))
                filename = parts[0].split(' (chunk ')[0]
                chunk_scores.append(f"{score:.4f} - {filename}")
            else:
                chunk_scores.append(chunk)
        except Exception as e:
            st.warning(f"Error parsing chunk info: {e}")
            chunk_scores.append(chunk)

    # Ensure we always have 3 entries, even if there are fewer chunks
    while len(chunk_scores) < 3:
        chunk_scores.append("")

    # Write to markdown history
    write_markdown_history(user_name, question, raw_response, csv_file)

    # Write raw data to CSV log
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [user_name, date, time, question, raw_response, unique_files_str] + chunk_scores
        writer.writerow(row)

    # Write raw data to JSON log
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
            "response": raw_response,
            "unique_files": unique_files,
            "chunk_info": chunk_info
        })
        f.seek(0)
        json.dump(logs, f, indent=4)
        f.truncate()

    # Return the encoded response for web output
    return encoded_response

def get_all_chat_history(user_name, logs_dir):
    history = []
    for filename in os.listdir(logs_dir):
        if filename.endswith('_response_log.csv'):
            file_path = os.path.join(logs_dir, filename)
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                for row in reader:
                    if row[0] == user_name:
                        history.append({
                            "name": row[0],
                            "date": row[1] if len(row) > 1 else "",
                            "time": row[2] if len(row) > 2 else "",
                            "question": row[3] if len(row) > 3 else "",
                            "response": row[4] if len(row) > 4 else "",
                            "unique_files": row[5] if len(row) > 5 else "",
                            "chunk_info": [
                                row[6] if len(row) > 6 else "",
                                row[7] if len(row) > 7 else "",
                                row[8] if len(row) > 8 else ""]
                        })
    return sorted(history, key=lambda x: (x['date'], x['time']), reverse=True)

def load_today_history():
    current_date = datetime.now().strftime("%d-%m-%Y")
    history_dir = os.path.join(script_dir, "history")
    today_file = os.path.join(history_dir, f"{current_date}_conversation_history.md")
    

    if os.path.exists(today_file):
        try:
            with open(today_file, 'r', encoding='utf-8') as file:
                content = file.read()
                logging.info(f"Successfully loaded today's history: {len(content)} characters")
                return content
        except Exception as e:
            logging.error(f"Error reading today's history: {str(e)}", exc_info=True)
            return ""
    else:
        logging.info("No history file found for today")
        return ""

# Modify the response function
def get_ollama_response(user_name, prompt):
    today_history = load_today_history()
    
    try:
        # Initialize Ollama handler
        api_handler = OllamaHandler(model_name="mistral")
        
        # Transform the prompt
        prompt_vector = vectorizer.transform([prompt])
        cosine_similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-3:][::-1]
        
        context_chunks_with_filenames = [document_chunks_with_filenames[i] for i in top_indices]
        context_chunks = [chunk for chunk, _ in context_chunks_with_filenames]
        context_filenames_list = [filename for _, filename in context_chunks_with_filenames]
        context_text = "\n".join(context_chunks)
        
        # Build context layers
        history_context = "\n".join([chunk for chunk, filename in document_chunks_with_filenames if 'history' in filename.lower()])
        normalized_name = user_name.lower().strip()
        student_context = "\n".join([chunk for chunk, filename in document_chunks_with_filenames if 'students' in filename.lower() and normalized_name in filename.lower()])
        
        # Prepare request
        character_prompt = get_patrick_prompt()
        user_message = f"""Context: {context_text}
        Previous conversation history: {history_context}
        Today's conversations: {today_history}
        Student project context: {student_context}
        Instructions: The above context includes general reference material, conversation history, and specific information about the student's project proposal. Use this information to:
        1. Provide responses that connect to the student's specific project interests
        2. Draw parallels between historical concepts and the student's research
        3. Maintain consistency with previous conversations
        4. Suggest relevant connections between their project and Geddes' work
        User's name: {user_name}
        Current date: {datetime.now().strftime("%d-%m-%Y")}
        Question: {prompt}"""

        data = {
            "messages": [
                {"role": "system", "content": character_prompt},
                {"role": "user", "content": user_message}
            ]
        }

        # Make request with error handling
        response_json = api_handler.make_request(data)
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            chunk_info = [f"{filename} (chunk {i+1}, score: {cosine_similarities[top_indices[i]]:.4f})" 
                         for i, filename in enumerate(context_filenames_list)]
            return response_json["choices"][0]["message"]["content"], list(set(context_filenames_list)), chunk_info
            
        return "Error: Unexpected response format from local server.", [], []
        
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", [], []

# Streamlit UI
st.title("The Ghost of Geddes...")

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
    Greetings, dear inquirer! I am Patrick Geddes, a man of many hats - biologist, sociologist, geographer, and yes, a bit of a revolutionary in the realm of town planning, if I do say so myself. 
    
    Now, my eager student, what's your name? And more importantly, what burning question about our shared world shall we explore together? 
    Remember, "By leaves we live" - so let your curiosity bloom and ask away!
    """, unsafe_allow_html=True)

#model choice
available_models = OllamaHandler().get_available_models()
if not available_models:
    available_models = ["mistral", "llama2", "codellama", "vicuna"]

selected_model = st.sidebar.selectbox(
    "Select Model",
    available_models,
    index=available_models.index("mistral") if "mistral" in available_models else 0
)

# Input section for user queries
user_name_input = st.text_input("Enter your name:")
prompt_input = st.text_area("Discuss your project with Patrick:")

if st.button('Submit'):
    if user_name_input and prompt_input:
        with st.spinner('Re-animating Geddes Ghost...'):
            try:
                # Get the latest file paths
                csv_file, json_file = initialize_log_files()
                
                # Get response and update logs
                response_content, unique_files, chunk_info = get_ollama_response(
                    user_name_input.strip(),
                    prompt_input.strip(),
                    model_name=selected_model
                )
                
                # Check for error messages in response
                if isinstance(response_content, str) and "error" in response_content.lower():
                    st.error(response_content)
                    st.stop()
                
                # If successful, update logs and display response
                encoded_response = update_chat_logs(
                    user_name=user_name_input.strip(),
                    question=prompt_input.strip(),
                    response=response_content,
                    unique_files=unique_files,
                    chunk_info=chunk_info,
                    csv_file=csv_file,
                    json_file=json_file
                )

                # Play sound only on successful response
                ding_sound.play()
                
                # Display latest response immediately after submission
                st.markdown(f"**Patrick Geddes:** {encoded_response}", unsafe_allow_html=True)
                st.markdown(f"**Sources:** {' - '.join(html.escape(file) for file in unique_files)}", unsafe_allow_html=True)
                st.markdown(f"**Chunks used:** {' - '.join(html.escape(chunk) for chunk in chunk_info)}", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.stop()
    else:
        st.warning("Please enter both your name and a question.")


# Chat history button
if st.button('Show Chat History'):
    logs_dir = os.path.join(script_dir, "logs")
    history = get_all_chat_history(user_name_input, logs_dir)
    for entry in history:
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <p style="color: black; font-weight: bold;">Name: {entry['name']}</p>
        <p style="color: black; font-weight: bold;">Date: {entry['date']} | Time: {entry['time']}</p>
        <p style="color: #FFA500; font-weight: bold;">Question:</p>
        <p>{entry['question']}</p>
        <p style="color: #FFA500; font-weight: bold;">Patrick Geddes:</p>
        <p>{entry['response']}</p>
        <p style="color: black; font-weight: bold;">Sources:</p>
        <p>{entry['unique_files']}</p>
        <p style="color: black; font-weight: bold;">Document relevance:</p>
        {html.escape(entry['chunk_info'][0])} - {html.escape(entry['chunk_info'][1])} - {html.escape(entry['chunk_info'][2])}
        """, unsafe_allow_html=True)