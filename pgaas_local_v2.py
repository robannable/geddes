"""
Patrick Geddes AI Assistant System (PGAAS)
----------------------------------------
A Retrieval-Augmented Generation (RAG) implementation that simulates interactions
with Patrick Geddes, the Scottish polymath. The system combines:

- RAG-based knowledge retrieval from multiple document sources
- Dynamic cognitive modes with temperature variation (0.7-0.9)
- Context-aware response generation using Llama 3.1 70B
- Comprehensive logging and response evaluation

The system processes user queries through three main document categories:
1. Authoritative documents (core knowledge)
2. Historical records (past interactions)
3. Student-specific content (personalized context)

Responses are generated using a structured context assembly process and
cognitive mode selection system that mimics Geddes' teaching approach.

Author: Rob Annable
Last Updated: 05-11-2024
Version: 1.0
"""

import re
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

# Set up logging
log_dir = os.path.join(script_dir, "debug_logs")
os.makedirs(log_dir, exist_ok=True)
current_date = datetime.now().strftime("%d-%m-%Y")
log_file = os.path.join(log_dir, f"{current_date}_rag_loading.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)

logger = logging.getLogger(__name__)  # Add this line to create the logger instance

# Initialize directories
sound_dir = os.path.join(script_dir, 'sounds')
prompts_dir = os.path.join(script_dir, 'prompts')
about_file_path = os.path.join(script_dir, 'about.txt')

# Initialize pygame for audio
pygame.mixer.init()

# Load sound file
ding_sound = pygame.mixer.Sound(os.path.join(sound_dir, 'ding2.wav'))

# Define constants at the top of the file
CONTEXT_WEIGHTS = {
    'student_specific': 1.5,  # Highest priority
    'project': 1.3,          # Project-related content
    'historical': 1.2,       # Historical context
    'general': 1.0          # Base documents
}

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
        self.context_weights = CONTEXT_WEIGHTS
    
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

class GeddesCognitiveModes:
    def __init__(self):
        self.modes = {
            'survey': {
                'keywords': [
                    'what', 'describe', 'analyze', 'observe', 'examine', 'study',
                    'investigate', 'explore', 'map', 'document', 'record', 'measure',
                    'identify', 'catalogue', 'survey', 'inspect', 'review', 'assess',
                    'where', 'when', 'who', 'which', 'look', 'find', 'discover'
                ],
                'prompt_prefix': 'Let us first survey and observe...',
                'temperature': 0.7
            },
            'synthesis': {
                'keywords': [
                    'how', 'connect', 'relate', 'integrate', 'combine', 'synthesize',
                    'weave', 'blend', 'merge', 'link', 'bridge', 'join', 'unite',
                    'pattern', 'relationship', 'network', 'system', 'structure',
                    'framework', 'together', 'between', 'across', 'through',
                    'interconnect', 'associate', 'correlate'
                ],
                'prompt_prefix': 'Now, let us weave together these disparate threads...',
                'temperature': 0.8
            },
            'proposition': {
                'keywords': [
                    'why', 'propose', 'suggest', 'could', 'might', 'imagine',
                    'envision', 'create', 'design', 'develop', 'innovate', 'transform',
                    'improve', 'enhance', 'advance', 'future', 'potential', 'possible',
                    'alternative', 'solution', 'strategy', 'plan', 'vision',
                    'hypothesis', 'theory', 'concept'
                ],
                'prompt_prefix': 'Let us venture forth with a proposition...',
                'temperature': 0.9
            }
        }
        logger.info("Initializing new GeddesCognitiveModes")

    def get_mode_parameters(self, prompt: str) -> dict:
        # Convert prompt to lowercase for matching
        prompt_lower = prompt.lower()
        
        # Count keyword matches for each mode with weighted scoring
        mode_scores = {}
        for mode, params in self.modes.items():
            # Count exact keyword matches
            exact_matches = sum(
                1 for keyword in params['keywords'] 
                if f" {keyword} " in f" {prompt_lower} "  # Add spaces to ensure whole word matching
            )
            
            # Count partial matches (for compound words or variations)
            partial_matches = sum(
                0.5 for keyword in params['keywords']
                if keyword in prompt_lower and f" {keyword} " not in f" {prompt_lower} "
            )
            
            # Combine scores
            mode_scores[mode] = exact_matches + partial_matches
        
        # Select mode with highest score (default to 'survey' if tied or no matches)
        selected_mode = max(
            mode_scores.items(),
            key=lambda x: (x[1], x[0] == 'survey')  # Prioritize survey mode in ties
        )[0]
        
        # Log the selected mode and score
        logger.info(f"Selected mode: {selected_mode} (score: {mode_scores[selected_mode]})")
        
        return {
            'mode': selected_mode,
            'prompt_prefix': self.modes[selected_mode]['prompt_prefix'],
            'temperature': self.modes[selected_mode]['temperature']
        }

class OllamaAPIHandler:
    def __init__(self, base_url="http://localhost:11434", timeout=60):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retry_strategy))
        
        # Try to preload model if not already loaded
        if not hasattr(st.session_state, 'model_preloaded'):
            preload_model()
            st.session_state.model_preloaded = True

    def make_request(self, messages, temperature=0.7, max_retries=3):
        try:
            # Extract the actual prompt from the messages
            prompt = messages[-1]["content"]
            
            data = {
                "model": "llama2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_ctx": 4096
                }
            }

            logger.info("Starting API request...")
            
            for attempt in range(max_retries):
                try:
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json=data,
                        timeout=self.timeout,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    logger.info("API request successful")
                    
                    return {
                        "message": {
                            "content": result.get("response", "")
                        }
                    }
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout on attempt {attempt + 1} of {max_retries}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)  # Wait 5 seconds before retrying
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(5)
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            logger.error(f"Request data: {data}")
            raise

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

class ErrorHandler:
    @staticmethod
    def handle_operation(operation_name: str = None, operation=None, *args, **kwargs):
        """
        Handle operations with error catching
        
        Args:
            operation_name (str): Name of the operation for logging
            operation (callable): Function to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
        
        Returns:
            tuple: (result, error_message)
        """
        if operation is None:
            return None, "No operation provided"
            
        try:
            return operation(*args, **kwargs), None
        except Exception as e:
            error_msg = f"Error in {operation_name or 'unnamed operation'}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg

@st.cache_data
def get_about_info():
    about_file_path = path_manager.get_file_path('base', 'about.txt')
    
    def read_about_file():
        with open(about_file_path, 'r') as f:
            return f.read().strip()
    
    result, error = ErrorHandler.handle_operation(
        operation_name="read_about_file",
        operation=read_about_file
    )
    
    return (result, True) if result else ("Default about info...", False)

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
def weight_context_chunks(prompt, chunks_with_filenames, vectorizer, tfidf_matrix):
    # Convert prompt to TF-IDF vector
    prompt_vector = vectorizer.transform([prompt])
    
    # Compute cosine similarities
    similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    
    # Apply weights based on document type and date
    weighted_similarities = similarities.copy()
    current_date = datetime.now().strftime("%d-%m-%Y")
    
    for i, (_, filename) in enumerate(chunks_with_filenames):
        # Extra weight for today's conversations
        if current_date in filename:
            weighted_similarities[i] *= 2.0  # Double the weight for today's content
        
        # Apply existing weights
        if "student_specific" in filename.lower():
            weighted_similarities[i] *= CONTEXT_WEIGHTS['student_specific']
        elif "project" in filename.lower():
            weighted_similarities[i] *= CONTEXT_WEIGHTS['project']
        elif "history" in filename.lower():
            weighted_similarities[i] *= CONTEXT_WEIGHTS['historical']
        else:
            weighted_similarities[i] *= CONTEXT_WEIGHTS['general']
    
    logger.info(f"Applied context weights: {CONTEXT_WEIGHTS}")
    logger.info(f"Today's date: {current_date}")
    
    return weighted_similarities

class TimeManager:
    @staticmethod
    def get_current_date():
        return datetime.now().strftime("%d-%m-%Y")
    
    @staticmethod
    def get_current_time():
        return datetime.now().strftime("%H:%M:%S")
    
    @staticmethod
    def get_current_datetime():
        now = datetime.now()
        return now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S")

class PathManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.paths = {
            'logs': os.path.join(base_dir, "logs"),
            'debug_logs': os.path.join(base_dir, "debug_logs"),
            'sounds': os.path.join(base_dir, "sounds"),
            'prompts': os.path.join(base_dir, "prompts"),
            'history': os.path.join(base_dir, "history"),
            'documents': os.path.join(base_dir, "documents"),
            'students': os.path.join(base_dir, "students"),
            'images': os.path.join(base_dir, "images")
        }
        
        # Create directories if they don't exist
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
    
    def get_path(self, key):
        return self.paths.get(key)
    
    def get_file_path(self, key, filename):
        return os.path.join(self.paths.get(key, self.base_dir), filename)

# Initialize at the start
path_manager = PathManager(script_dir)

def initialize_log_files():
    current_date = TimeManager.get_current_date()
    csv_file = path_manager.get_file_path('logs', f"{current_date}_response_log.csv")
    json_file = path_manager.get_file_path('logs', f"{current_date}_response_log.json")
    
    # Initialize CSV if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'date', 'time', 'name', 'question', 'response', 
                'unique_files', 'chunk1_score', 'chunk2_score', 'chunk3_score',
                'cognitive_mode', 'response_length', 'creative_markers', 'temperature'
            ])
            writer.writeheader()
    
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
    try:
        # Get current metrics from session state
        metrics = getattr(st.session_state, 'current_metrics', {})
        
        # Prepare CSV row with all metrics
        csv_row = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'name': user_name,
            'question': question,
            'response': response,
            'unique_files': ' - '.join(unique_files),
            'chunk1_score': metrics.get('chunk1_score', 'N/A'),
            'chunk2_score': metrics.get('chunk2_score', 'N/A'),
            'chunk3_score': metrics.get('chunk3_score', 'N/A'),
            'cognitive_mode': metrics.get('cognitive_mode', {}),
            'response_length': metrics.get('response_length', 0),
            'creative_markers': metrics.get('creative_markers', 0),
            'temperature': metrics.get('temperature', {0.7: 0, 0.8: 0, 0.9: 0})
        }
        
        # Use context managers for file operations
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(csv_row)
        
        # Prepare JSON entry
        json_entry = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'name': user_name,
            'question': question,
            'response': response,
            'unique_files': unique_files,
            'chunk_info': chunk_info,
            'cognitive_mode': metrics.get('cognitive_mode', {}),
            'evaluation': {
                'mode_distribution': metrics.get('cognitive_mode', {}),
                'avg_response_length': metrics.get('response_length', 0),
                'creative_markers_frequency': metrics.get('creative_markers', 0),
                'temperature_effectiveness': metrics.get('temperature', {0.7: 0, 0.8: 0, 0.9: 0})
            }
        }
        
        # Combine JSON read/write operations
        try:
            with open(json_file, 'r+', encoding='utf-8') as f:
                try:
                    chat_history = json.load(f)
                except json.JSONDecodeError:
                    chat_history = []
                chat_history.append(json_entry)
                f.seek(0)
                f.truncate()
                json.dump(chat_history, f, indent=2, ensure_ascii=False)
        except FileNotFoundError:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump([json_entry], f, indent=2, ensure_ascii=False)
                
        return response
    except Exception as e:
        logger.error(f"Error updating chat logs: {str(e)}")
        return f"An error occurred: {str(e)}", [], []

def get_all_chat_history(user_name, logs_dir):
    history = []
    for filename in os.listdir(logs_dir):
        if filename.endswith('_response_log.csv'):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)  # Changed to DictReader
                    for row in reader:
                        # Update date parsing to handle both formats
                        try:
                            date = datetime.strptime(row['date'], '%d-%m-%Y').strftime('%d-%m-%Y')
                        except ValueError:
                            try:
                                date = datetime.strptime(row['date'], '%Y-%m-%d').strftime('%d-%m-%Y')
                            except ValueError:
                                continue
                            
                        if row['name'] == user_name:
                            history.append({
                                "name": row['name'],
                                "date": date,
                                "time": row.get('time', ""),
                                "question": row.get('question', ""),
                                "response": row.get('response', ""),
                                "unique_files": row.get('unique_files', ""),
                                "chunk_info": [
                                    row.get('chunk1_score', ""),
                                    row.get('chunk2_score', ""),
                                    row.get('chunk3_score', "")
                                ]
                            })
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                continue
                
    return sorted(history, key=lambda x: (
        datetime.strptime(x['date'], '%d-%m-%Y'),
        x['time']
    ), reverse=True)

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
) -> dict:
    """
    Assembles context with improved structure and weighting
    """
    # Get weighted context from memory
    categorized_context = context_manager.get_weighted_context(prompt, user_name)
    
    # Process and categorize RAG chunks with relevance scores
    rag_context = {
        'authoritative': [],
        'historical': [],
        'student_specific': [],
        'recent_interactions': []
    }
    
    # Process RAG chunks with scores
    for chunk, filename in top_chunks:
        # Calculate chunk relevance (assuming cosine similarity score is available)
        relevance_score = cosine_similarity(
            vectorizer.transform([prompt]), 
            vectorizer.transform([chunk])
        )[0][0]
        
        context_item = {
            'content': chunk,
            'source': filename,
            'relevance': relevance_score
        }
        
        # Categorize based on source
        if 'documents' in filename.lower():
            rag_context['authoritative'].append(context_item)
        elif 'history' in filename.lower():
            rag_context['historical'].append(context_item)
        elif user_name.lower() in filename.lower():
            rag_context['student_specific'].append(context_item)
    
    # Sort each category by relevance
    for category in rag_context:
        rag_context[category] = sorted(
            rag_context[category],
            key=lambda x: x['relevance'],
            reverse=True
        )[:3]  # Keep top 3 most relevant chunks per category
    
    return rag_context

class ResponseProcessor:
    @staticmethod
    def process_llm_response(response_json, user_name, prompt, unique_files, chunk_info):
        if "choices" not in response_json or not response_json["choices"]:
            logger.error("Invalid response format from API")
            return None, [], []
            
        response_content = response_json["choices"][0]["message"]["content"]
        
        # Get mode parameters with explicit mode handling
        mode_params = st.session_state.cognitive_modes.get_mode_parameters(prompt)
        selected_mode = mode_params.get('mode', 'survey')
        temperature = mode_params.get('temperature', 0.7)
        
        # Get evaluation metrics
        evaluation_results = st.session_state.response_evaluator.evaluate_response(
            response=response_content,
            mode=selected_mode,
            temperature=temperature
        )
        
        # Format chunk info for CSV
        chunk_scores = [f"{info.split(' (Score: ')[0]} (score: {info.split(' (Score: ')[1].rstrip(')')})" 
                       for info in chunk_info[:3]] if chunk_info else []
        while len(chunk_scores) < 3:
            chunk_scores.append("N/A")
            
        # Create temperature distribution dictionary
        temperature_distribution = {
            0.7: 1 if abs(temperature - 0.7) < 0.1 else 0,
            0.8: 1 if abs(temperature - 0.8) < 0.1 else 0,
            0.9: 1 if abs(temperature - 0.9) < 0.1 else 0
        }
            
        # Add metrics to session state for CSV logging
        st.session_state.current_metrics = {
            'chunk1_score': chunk_scores[0],
            'chunk2_score': chunk_scores[1],
            'chunk3_score': chunk_scores[2],
            'cognitive_mode': evaluation_results['mode_distribution'],
            'response_length': evaluation_results['avg_response_length'],
            'creative_markers': evaluation_results.get('creative_markers_frequency', 0),
            'temperature': temperature_distribution
        }
        
        return response_content, unique_files, chunk_info

def get_llm_response(user_name, prompt):
    try:
        # Get document chunks and compute relevance
        weighted_similarities = weight_context_chunks(
            prompt, 
            document_chunks_with_filenames,
            vectorizer, 
            tfidf_matrix
        )
        
        # Get top chunks and info
        top_indices = weighted_similarities.argsort()[-5:][::-1]
        top_chunks = [document_chunks_with_filenames[i] for i in top_indices]
        unique_files = list(set(filename for _, filename in top_chunks))
        chunk_info = [
            f"{document_chunks_with_filenames[i][1]} (Score: {weighted_similarities[i]:.3f})" 
            for i in top_indices
        ]
        
        # Load today's history and get mode parameters
        today_history = load_today_history()
        mode_params = st.session_state.cognitive_modes.get_mode_parameters(prompt)
        selected_mode = mode_params.get('mode', 'survey')
        temperature = mode_params.get('temperature', 0.7)

        # Get list of recent history files
        history_dir = os.path.join(script_dir, "history")
        history_files = sorted([f for f in os.listdir(history_dir) if f.endswith('_conversation_history.md')], reverse=True)[:7]
        
        # Process today's conversations
        today_conversations = []
        today_date = datetime.now().strftime("%d-%m-%Y")
        
        if isinstance(today_history, list):
            for entry in today_history:
                if isinstance(entry, dict):
                    today_conversations.append({
                        'time': entry.get('time', 'unknown time'),
                        'name': entry.get('name', 'unknown user'),
                        'topic': entry.get('question', 'unknown topic')[:100] + '...'
                    })

        # Create temporal context
        time_context = f"""
        Current date and time: {datetime.now().strftime('%d-%m-%Y %H:%M')}
        
        Today's conversations ({today_date}):
        {' '.join([f"- {conv['time']}: {conv['name']} asked about {conv['topic']}" for conv in today_conversations]) if today_conversations else "No previous conversations today."}
        
        Available conversation history from:
        {', '.join(f.split('_')[0] for f in history_files)}
        
        Important: Only reference conversations that are explicitly listed above. Do not fabricate or imagine conversations that are not in this history.
        """

        # Get enhanced context structure
        rag_context = assemble_enhanced_context(
            user_name=user_name,
            prompt=prompt,
            context_manager=st.session_state.context_manager,
            top_chunks=top_chunks,
            today_history=today_history
        )

        # Format the complete prompt
        formatted_prompt = f"""
        {get_patrick_prompt()}

        {time_context}

        Based on the following authoritative sources and context, please provide a response:

        Authoritative Knowledge:
        {' '.join(chunk['content'] for chunk in rag_context['authoritative'])}

        Historical Context:
        {' '.join(chunk['content'] for chunk in rag_context['historical'])}

        Student-Specific Context:
        {' '.join(chunk['content'] for chunk in rag_context['student_specific'])}

        User's Question: {prompt}
        User's Name: {user_name}
        Current Mode: {selected_mode}
        """

        # Make API request using the handler
        api_handler = OllamaAPIHandler()
        messages = [{"role": "user", "content": formatted_prompt}]
        response_json = api_handler.make_request(messages, temperature=temperature)
        
        response_content = response_json.get("message", {}).get("content", "")
        
        return ResponseProcessor.process_llm_response(
            {"choices": [{"message": {"content": response_content}}]}, 
            user_name, 
            prompt, 
            unique_files, 
            chunk_info
        )

    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}")
        return f"An unexpected error occurred: {str(e)}", [], []

def preload_model(model_name="llama2", timeout=60):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": "Hello",
        "stream": False,
        "options": {
            "num_ctx": 4096
        }
    }
    try:
        logger.info("Attempting to preload model...")
        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()
        logger.info(f"Successfully preloaded model: {model_name}")
    except requests.exceptions.Timeout:
        logger.warning("Model preload timed out - this is okay, model may still be loading")
    except Exception as e:
        logger.error(f"Failed to preload model: {str(e)}")

class ResponseEvaluator:
    def __init__(self):
        self.mode_history = []
        self.response_lengths = []
        self.creative_markers = [
            "imagine", "could", "might", "perhaps", "possible",
            "suggest", "consider", "envision", "creative", "innovative"
        ]

    def evaluate_response(self, response, mode, temperature):
        """Evaluates the response based on multiple criteria"""
        # Update mode history
        self.mode_history.append(mode)
        
        # Calculate response length
        response_length = len(response.split())
        self.response_lengths.append(response_length)
        avg_length = sum(self.response_lengths) / len(self.response_lengths)
        
        # Count creative markers
        creative_count = sum(1 for marker in self.creative_markers if marker.lower() in response.lower())
        
        # Calculate mode distribution
        mode_counts = {}
        for m in self.mode_history:
            mode_counts[m] = mode_counts.get(m, 0) + 1
        
        # Create temperature distribution
        temperature_distribution = {
            0.7: 1 if abs(temperature - 0.7) < 0.1 else 0,
            0.8: 1 if abs(temperature - 0.8) < 0.1 else 0,
            0.9: 1 if abs(temperature - 0.9) < 0.1 else 0
        }
        
        evaluation = {
            'mode_distribution': mode_counts,
            'avg_response_length': avg_length,
            'creative_markers_frequency': creative_count,
            'temperature_effectiveness': temperature_distribution
        }
        
        logger.info(f"Response evaluation: {evaluation}")
        return evaluation

class SessionStateManager:
    @staticmethod
    def initialize_session():
        """Initialize all required session state variables"""
        if 'initialized' not in st.session_state:
            logger.info("Initializing new session state")
            
            # Initialize core components
            st.session_state.cognitive_modes = GeddesCognitiveModes()
            st.session_state.context_manager = EnhancedContextManager()
            st.session_state.response_evaluator = ResponseEvaluator()
            
            # Initialize conversation tracking
            st.session_state.conversation_history = []
            st.session_state.current_metrics = {}
            
            # Initialize model state
            if not hasattr(st.session_state, 'model_preloaded'):
                preload_model()
                st.session_state.model_preloaded = True
            
            # Mark initialization as complete
            st.session_state.initialized = True
            logger.info("Session state initialization complete")
    
    @staticmethod
    def get_state(key, default=None):
        """Safely get a value from session state"""
        return getattr(st.session_state, key, default)
    
    @staticmethod
    def set_state(key, value):
        """Safely set a value in session state"""
        setattr(st.session_state, key, value)
    
    @staticmethod
    def ensure_initialized():
        """Ensure session is initialized"""
        if not getattr(st.session_state, 'initialized', False):
            SessionStateManager.initialize_session()

# Initialize session state before the Streamlit UI
SessionStateManager.ensure_initialized()

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
        with st.spinner('Re-animating Geddes Ghost... This might take a minute when we first revive him...'):
            try:
                # Get the latest file paths
                csv_file, json_file = initialize_log_files()
                
                # Get response and update logs
                response_content, unique_files, chunk_info = get_llm_response(
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

                # Add this line to write markdown history
                write_markdown_history(
                    user_name=user_name_input.strip(),
                    question=prompt_input.strip(),
                    response=response_content,
                    csv_file=csv_file
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
        </div>
        """, unsafe_allow_html=True)