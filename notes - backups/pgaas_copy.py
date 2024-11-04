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
from dataclasses import dataclass
from typing import List, Dict
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

# Add these classes after your existing imports but before any function definitions
@dataclass
class ContextItem:
    content: str
    timestamp: datetime  # Changed from datetime.datetime
    source: str
    relevance_score: float = 0.0

class EnhancedContextManager:
    def __init__(self, max_memory_items: int = 10):
        self.max_memory_items = max_memory_items
        self.conversation_memory: List[ContextItem] = []
        self.context_weights = {
            'student_specific': 1.5,
            'recent_conversation': 1.3,
            'historical': 1.2,
            'general': 1.0
        }
    
    def add_conversation(self, content: str, source: str):
        context_item = ContextItem(
            content=content,
            timestamp=datetime.now(),  # Changed from datetime.datetime.now()
            source=source
        )
        self.conversation_memory.append(context_item)
        if len(self.conversation_memory) > self.max_memory_items:
            self.conversation_memory.pop(0)
    
    def get_weighted_context(self, query: str, user_name: str) -> Dict[str, List[ContextItem]]:
        categorized_context = {
            'student_specific': [],
            'recent_conversation': [],
            'historical': [],
            'general': []
        }
        
        # Categorize conversation memory
        for item in self.conversation_memory:
            if user_name.lower() in item.source.lower():
                categorized_context['student_specific'].append(item)
            else:
                categorized_context['recent_conversation'].append(item)
        
        return categorized_context

# Set up API
class PerplexityAPIHandler:
    def __init__(self, api_key, max_retries=3, timeout=30):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504, 524],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def make_request(self, data):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 524:
                    raise TimeoutError("Server timeout error (524)")
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.ConnectionError:
                if attempt == self.max_retries - 1:
                    raise ConnectionError("Failed to establish connection with API")
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.exceptions.Timeout:
                if attempt == self.max_retries - 1:
                    raise TimeoutError("Request timed out")
                time.sleep(2 ** attempt)
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code in [429, 524]:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(5)  # Rate limit backoff
                else:
                    raise


# Streamlit secrets API location
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

def check_api_connection():
    """Check if we can connect to the Perplexity API"""
    try:
        session = requests.Session()
        # Instead of checking a health endpoint, we'll do a minimal API call
        response = session.post(
            'https://api.perplexity.ai/chat/completions',
            headers={
                "Authorization": f"Bearer {st.secrets['PERPLEXITY_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-70b-instruct",
                "messages": [{"role": "user", "content": "test"}]
            },
            timeout=5
        )
        # Consider both 200 (success) and 401 (auth error) as "connected"
        return response.status_code in [200, 401]
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

# Add a context weighting system to prioritize different types of documents
def weight_context_chunks(prompt, context_chunks_with_filenames, vectorizer, tfidf_matrix):
    """Weight context chunks based on document type and relevance"""
    prompt_vector = vectorizer.transform([prompt])
    base_similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    
    weighted_similarities = []
    for i, (chunk, filename) in enumerate(context_chunks_with_filenames):
        # Base similarity score
        score = base_similarities[i]
        
        # Apply weights based on document type
        if 'students' in filename.lower():
            score *= 1.5  # Prioritize student-specific content
        elif 'history' in filename.lower():
            score *= 1.2  # Give preference to historical context
        elif any(term in chunk.lower() for term in ['project', 'research', 'study']):
            score *= 1.3  # Boost project-related content
            
        weighted_similarities.append(score)
    
    return np.array(weighted_similarities)

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
    
def get_temporal_context(today_history, max_history_chunks=5):
    """Process conversation history with temporal weighting"""
    history_chunks = today_history.split("##")
    
    # Sort chunks by timestamp (assuming they start with timestamp)
    history_chunks.sort(key=lambda x: x.split("|")[0] if "|" in x else "", reverse=True)
    
    # Take most recent chunks and apply temporal weighting
    recent_chunks = []
    for i, chunk in enumerate(history_chunks[:max_history_chunks]):
        temporal_weight = 1 / (i + 1)  # More recent = higher weight
        recent_chunks.append({
            'content': chunk,
            'weight': temporal_weight
        })
    
    return recent_chunks    

def assemble_enhanced_context(
    user_name: str,
    prompt: str,
    context_manager: EnhancedContextManager,
    top_chunks: List[tuple],
    today_history: str
) -> str:
    """
    Assembles context with improved structure and weighting
    
    Args:
        user_name: Name of the current user
        prompt: Current user query
        context_manager: Instance of EnhancedContextManager
        top_chunks: List of (chunk, filename) tuples from RAG
        today_history: Today's conversation history
    
    Returns:
        Structured context string for the LLM
    """
    # Get weighted context from memory
    categorized_context = context_manager.get_weighted_context(prompt, user_name)
    
    # Process RAG chunks
    for chunk, filename in top_chunks:
        context_item = ContextItem(
            content=chunk,
            timestamp=datetime.now(),
            source=filename
        )
        
        if user_name.lower() in filename.lower():
            categorized_context['student_specific'].append(context_item)
        elif 'history' in filename.lower():
            categorized_context['historical'].append(context_item)
        else:
            categorized_context['general'].append(context_item)
    
    # Assemble final context
    context_sections = []
    
    # Student-specific context (highest priority)
    if categorized_context['student_specific']:
        context_sections.append("Student-Specific Context:")
        context_sections.extend([
            f"- {item.content}" 
            for item in categorized_context['student_specific'][:3]
        ])
    
    # Recent conversations
    if categorized_context['recent_conversation']:
        context_sections.append("\nRecent Relevant Discussions:")
        context_sections.extend([
            f"- {item.content}" 
            for item in sorted(
                categorized_context['recent_conversation'],
                key=lambda x: x.timestamp,
                reverse=True
            )[:3]
        ])
    
    # Historical context
    if categorized_context['historical']:
        context_sections.append("\nHistorical Background:")
        context_sections.extend([
            f"- {item.content}" 
            for item in categorized_context['historical'][:2]
        ])
    
    # General knowledge
    if categorized_context['general']:
        context_sections.append("\nGeneral Reference:")
        context_sections.extend([
            f"- {item.content}" 
            for item in categorized_context['general'][:2]
        ])
    
    return "\n".join(context_sections)

def get_perplexity_response(user_name, prompt):
    today_history = load_today_history()
    api_handler = PerplexityAPIHandler(st.secrets['PERPLEXITY_API_KEY'])
    
    try:
        # Transform the prompt
        prompt_vector = vectorizer.transform([prompt])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-3:][::-1]

        # Get context chunks and filenames
        context_chunks_with_filenames = [document_chunks_with_filenames[i] for i in top_indices]
        context_chunks = [chunk for chunk, _ in context_chunks_with_filenames]
        context_filenames_list = [filename for _, filename in context_chunks_with_filenames]
        
        # Get enhanced context
        enhanced_context = assemble_enhanced_context(
            user_name=user_name,
            prompt=prompt,
            context_manager=st.session_state.context_manager,
            top_chunks=context_chunks_with_filenames,
            today_history=today_history
        )
        
        # Update context memory
        st.session_state.context_manager.add_conversation(
            content=prompt,
            source=f"user_{user_name}"
        )
        
        # Prepare API request
        character_prompt = get_patrick_prompt()
        user_message = f"""
        Enhanced Context:
        {enhanced_context}
        
        User's name: {user_name}
        Current date: {datetime.now().strftime("%d-%m-%Y")}
        Question: {prompt}
        """

        data = {
            "model": "llama-3.1-70b-instruct",
            "messages": [
                {"role": "system", "content": character_prompt},
                {"role": "user", "content": user_message}
            ]
        }

        # Make API request with error handling
        response_json = api_handler.make_request(data)
        
        if "choices" in response_json and len(response_json["choices"]) > 0:
            # Create chunk info with scores
            chunk_info = [
                f"{filename} (chunk {i+1}, score: {cosine_similarities[top_indices[i]]:.4f})" 
                for i, filename in enumerate(context_filenames_list)
            ]
            return response_json["choices"][0]["message"]["content"], context_filenames_list, chunk_info

    except ConnectionError as e:
        return ("I apologize, but I'm having trouble connecting to my knowledge base. "
               "Please check your internet connection and try again."), [], []
    except TimeoutError as e:
        return ("I apologize, but the server is taking too long to respond. "
               "Please try again in a moment."), [], []
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", [], []

if 'context_manager' not in st.session_state:
    st.session_state.context_manager = EnhancedContextManager()

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
                response_content, unique_files, chunk_info = get_perplexity_response(
                    user_name_input.strip(), 
                    prompt_input.strip()
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