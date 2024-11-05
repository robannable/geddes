# admin_dashboard.py

import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
import json
import csv
import os
import numpy as np
from collections import defaultdict
import time


# Get the script directory and create debug_logs folder
script_dir = os.path.dirname(os.path.abspath(__file__))
debug_logs_dir = os.path.join(script_dir, "debug_logs")
os.makedirs(debug_logs_dir, exist_ok=True)

# Configure logging to write to debug_logs folder
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(debug_logs_dir, 'admin_dashboard.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    def __init__(self):
        self.metrics = {
            'mode_distribution': {'survey': 0, 'synthesis': 0, 'proposition': 0},
            'response_lengths': [],
            'creative_markers': {
                'metaphor': 0,
                'ecological_reference': 0,
                'speculative_proposition': 0,
                'cross-disciplinary': 0
            },
            'temperature_effectiveness': {0.7: [], 0.8: [], 0.9: []}
        }
        logger.info("ResponseEvaluator initialized")
    
    def evaluate_response(self, response: str, mode: str, temperature: float) -> dict:
        # Update mode distribution
        self.metrics['mode_distribution'][mode] = self.metrics['mode_distribution'].get(mode, 0) + 1
        
        # Update response length
        response_length = len(response.split())
        self.metrics['response_lengths'].append(response_length)
        
        # Update temperature effectiveness
        if temperature in self.metrics['temperature_effectiveness']:
            self.metrics['temperature_effectiveness'][temperature].append(response_length)
        
        # Analyze for creative markers
        lower_response = response.lower()
        if any(word in lower_response for word in ['like', 'as', 'metaphor', 'akin']):
            self.metrics['creative_markers']['metaphor'] += 1
        if any(word in lower_response for word in ['ecology', 'nature', 'environment', 'organic']):
            self.metrics['creative_markers']['ecological_reference'] += 1
        if any(word in lower_response for word in ['could', 'might', 'suggest', 'propose']):
            self.metrics['creative_markers']['speculative_proposition'] += 1
        if any(word in lower_response for word in ['across', 'between', 'integrate', 'combine']):
            self.metrics['creative_markers']['cross-disciplinary'] += 1
        
        # Calculate averages for temperature effectiveness
        temp_effectiveness = {
            temp: sum(lengths) / len(lengths) if lengths else 0
            for temp, lengths in self.metrics['temperature_effectiveness'].items()
        }
        
        return {
            'mode_distribution': self.metrics['mode_distribution'],
            'avg_response_length': sum(self.metrics['response_lengths']) / len(self.metrics['response_lengths']),
            'creative_markers_frequency': dict(self.metrics['creative_markers']),
            'temperature_effectiveness': temp_effectiveness
        }
    
    def _check_creative_marker(self, response: str, marker: str) -> bool:
        """Check for presence of creative markers in response"""
        marker_patterns = {
            'metaphor': r'like|as if|resembles',
            'cross-disciplinary': r'biology|sociology|economics|art',
            'historical_parallel': r'historically|in the past|reminds me of',
            'ecological_reference': r'nature|ecosystem|organic',
            'speculative_proposition': r'what if|imagine|consider'
        }
        return bool(re.search(marker_patterns[marker], response.lower()))
    
    def _generate_evaluation_report(self) -> dict:
        """Generate summary report of metrics"""
        return {
            'mode_distribution': dict(self.metrics['mode_distribution']),
            'avg_response_length': np.mean(self.metrics['response_lengths']),
            'creative_markers_frequency': dict(self.metrics['creative_markers']),
            'temperature_effectiveness': {
                temp: np.mean(lengths)
                for temp, lengths in self.metrics['temperature_effectiveness'].items()
            }
        }

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["admin_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        return True

def load_response_data(logs_dir):
    """Load and process all response logs"""
    all_data = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('_response_log.csv'):
            file_path = os.path.join(logs_dir, filename)
            
            try:
                # First try utf-8-sig with error handling
                df = pd.read_csv(file_path, encoding='utf-8-sig', on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
                logger.info(f"Reading {filename} - columns found: {df.columns.tolist()}")
            except UnicodeDecodeError:
                try:
                    # If that fails, try latin1 with error handling
                    df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', quoting=csv.QUOTE_MINIMAL)
                    logger.info(f"Reading {filename} - columns found: {df.columns.tolist()}")
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {str(e)}")
                    continue
            
            # Check for missing columns and add them with None values
            new_columns = ['cognitive_mode', 'response_length', 'creative_markers', 'temperature']
            for col in new_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Basic data cleaning
            if 'date' in df.columns:
                try:
                    # Convert date strings to datetime objects
                    df['date'] = pd.to_datetime(df['date'])
                    # Clean up any problematic strings in the DataFrame
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            df[col] = df[col].astype(str).replace({r'\n': ' ', r'\r': ' '}, regex=True)
                    all_data.append(df)
                    logger.info(f"Successfully loaded {filename} with {len(df)} rows")
                except Exception as e:
                    logger.error(f"Unable to parse dates in {filename}: {str(e)}")
                    continue
    
    try:
        if all_data:
            # Ensure all DataFrames have the same columns
            columns = set()
            for df in all_data:
                columns.update(df.columns)
            
            # Add missing columns to each DataFrame
            for i in range(len(all_data)):
                for col in columns:
                    if col not in all_data[i].columns:
                        all_data[i][col] = None
            
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('date')
            logger.info(f"Total combined rows: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No data frames were successfully loaded")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error combining data frames: {str(e)}")
        return pd.DataFrame()

def analyze_chunk_scores(df):
    """Analyze document chunk relevance scores"""
    scores = []
    files = []
    
    for _, row in df.iterrows():
        for chunk_col in ['chunk1_score', 'chunk2_score', 'chunk3_score']:
            if pd.notna(row[chunk_col]):
                try:
                    parts = row[chunk_col].split(' - ')
                    if len(parts) >= 2:
                        score = float(parts[0])
                        filename = parts[1]
                        scores.append(score)
                        files.append(filename)
                except:
                    continue
    
    return pd.DataFrame({'score': scores, 'file': files})

def display_chunk_analysis(df):
    st.subheader("Document Chunk Relevance Analysis")
    chunk_data = analyze_chunk_scores(df)
    
    if not chunk_data.empty:
        avg_scores = chunk_data.groupby('file')['score'].agg(['mean', 'count']).reset_index()
        avg_scores = avg_scores.sort_values('mean', ascending=False)
        
        fig = px.bar(avg_scores, 
                     x='file', 
                     y='mean',
                     color='count',
                     title="Average Relevance Score by Document",
                     labels={'mean': 'Average Score', 
                            'file': 'Document', 
                            'count': 'Times Used'})
        st.plotly_chart(fig)

        fig = px.histogram(chunk_data, 
                          x='score',
                          nbins=20,
                          title="Distribution of Relevance Scores",
                          labels={'score': 'Relevance Score', 
                                 'count': 'Frequency'})
        st.plotly_chart(fig)

def display_response_times(df):
    st.subheader("Response Time Analysis")
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
        response_times = df.groupby(df['date'].dt.date)['time'].count().reset_index()
        
        fig = px.line(response_times,
                      x='date',
                      y='time',
                      title="Responses per Day",
                      labels={'date': 'Date', 
                             'time': 'Number of Responses'})
        st.plotly_chart(fig)

def display_user_interactions(df):
    st.subheader("User Interaction Patterns")
    user_stats = df.groupby('name').agg({
        'question': 'count',
        'date': 'nunique'
    }).reset_index()
    user_stats.columns = ['User', 'Total Questions', 'Active Days']
    
    fig = px.bar(user_stats,
                 x='User',
                 y=['Total Questions', 'Active Days'],
                 title="User Engagement Metrics",
                 barmode='group')
    st.plotly_chart(fig)

def display_response_metrics():
    st.subheader("Response Quality Metrics")
    
    # Load data from CSV
    logs_dir = os.path.join(script_dir, "logs")
    df = load_response_data(logs_dir)
    
    if df.empty:
        st.info("No response data available yet. Try using the system first.")
        return
        
    # Log the data we're working with
    logger.info(f"Loaded dataframe with columns: {df.columns.tolist()}")
    logger.info(f"Sample cognitive_mode: {df['cognitive_mode'].iloc[0] if 'cognitive_mode' in df.columns else None}")
    logger.info(f"Sample creative_markers: {df['creative_markers'].iloc[0] if 'creative_markers' in df.columns else None}")
    logger.info(f"Sample temperature: {df['temperature'].iloc[0] if 'temperature' in df.columns else None}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mode distribution chart
        try:
            mode_counts = defaultdict(int)
            for mode_str in df['cognitive_mode'].dropna():
                try:
                    if isinstance(mode_str, str) and mode_str.strip():
                        mode_dict = eval(mode_str.strip())
                        if isinstance(mode_dict, dict):
                            for mode, count in mode_dict.items():
                                mode_counts[mode] += int(count)
                except:
                    continue
            
            if mode_counts:
                mode_data = pd.DataFrame(
                    list(mode_counts.items()),
                    columns=['Mode', 'Count']
                )
                fig = px.bar(mode_data, x='Mode', y='Count',
                            title="Cognitive Mode Distribution")
                st.plotly_chart(fig)
        except Exception as e:
            logger.error(f"Error processing mode distribution: {str(e)}")
    
    with col2:
        # Creative markers frequency
        try:
            marker_counts = defaultdict(int)
            for markers_str in df['creative_markers'].dropna():
                try:
                    if isinstance(markers_str, str) and markers_str.strip():
                        markers_dict = eval(markers_str.strip())
                        if isinstance(markers_dict, dict):
                            for marker, count in markers_dict.items():
                                marker_counts[marker] += int(count)
                except:
                    continue
            
            if marker_counts:
                markers_data = pd.DataFrame(
                    list(marker_counts.items()),
                    columns=['Marker', 'Frequency']
                )
                fig = px.bar(markers_data, x='Marker', y='Frequency',
                            title="Creative Markers Frequency")
                st.plotly_chart(fig)
        except Exception as e:
            logger.error(f"Error processing creative markers: {str(e)}")
    
    # Temperature impact analysis
    try:
        temp_data = []
        for idx, row in df.iterrows():
            try:
                if isinstance(row.get('temperature'), str) and row['temperature'].strip():
                    temp_dict = eval(row['temperature'].strip())
                    if isinstance(temp_dict, dict):
                        for temp, length in temp_dict.items():
                            try:
                                value = float(length) if isinstance(length, (int, float)) else float(np.mean(length))
                                temp_data.append({
                                    'Temperature': float(temp),
                                    'Response Length': value,
                                    'Response Number': idx + 1
                                })
                            except:
                                continue
            except:
                continue
        
        if temp_data:
            temp_df = pd.DataFrame(temp_data)
            fig = px.line(temp_df, x='Response Number', y='Response Length',
                         title="Temperature Impact on Response Length")
            st.plotly_chart(fig)
    except Exception as e:
        logger.error(f"Error processing temperature impact: {str(e)}")

def create_performance_dashboard(df):
    st.header("System Performance Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(
                datetime.now() - timedelta(days=7),
                datetime.now()
            )
        )
    
    with col2:
        metrics_type = st.selectbox(
            "Metrics Type",
            ["All", "Chunk Scores", "Response Times", "User Interactions"]
        )

    if metrics_type in ["All", "Chunk Scores"]:
        display_chunk_analysis(df)
    
    if metrics_type in ["All", "Response Times"]:
        display_response_times(df)
    
    if metrics_type in ["All", "User Interactions"]:
        display_user_interactions(df)

def create_document_usage_analysis(df):
    st.header("Document Usage Analysis")
    
    doc_usage = defaultdict(int)
    for files in df['unique_files'].dropna():
        for doc in str(files).split(' - '):
            doc_usage[doc.strip()] += 1
    
    doc_df = pd.DataFrame({
        'document': list(doc_usage.keys()),
        'usage_count': list(doc_usage.values())
    }).sort_values('usage_count', ascending=False)
    
    fig = px.bar(doc_df,
                 x='document',
                 y='usage_count',
                 title="Document Usage Frequency",
                 labels={'document': 'Document', 
                        'usage_count': 'Times Used'})
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    
    st.subheader("Document Co-occurrence Analysis")
    doc_cooccurrence = defaultdict(lambda: defaultdict(int))
    for files in df['unique_files'].dropna():
        docs = [doc.strip() for doc in str(files).split(' - ')]
        for i, doc1 in enumerate(docs):
            for doc2 in docs[i+1:]:
                doc_cooccurrence[doc1][doc2] += 1
                doc_cooccurrence[doc2][doc1] += 1
    
    docs = sorted(list(doc_usage.keys()))
    matrix = [[doc_cooccurrence[doc1][doc2] for doc2 in docs] for doc1 in docs]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=docs,
        y=docs,
        colorscale='Viridis'
    ))
    fig.update_layout(
        title="Document Co-occurrence Matrix",
        xaxis_tickangle=45
    )
    st.plotly_chart(fig)

def create_user_analysis(df):
    st.header("User Interaction Analysis")
    
    st.subheader("User Activity Timeline")
    user_activity = df.groupby(['date', 'name']).size().reset_index(name='questions')
    fig = px.line(user_activity, 
                  x='date', 
                  y='questions', 
                  color='name',
                  title="Questions per User Over Time")
    st.plotly_chart(fig)
    
    st.subheader("User Engagement Metrics")
    user_metrics = df.groupby('name').agg({
        'question': ['count', 'max'],
        'date': 'nunique'
    }).reset_index()
    user_metrics.columns = ['User', 'Total Questions', 'Longest Question', 'Active Days']
    st.dataframe(user_metrics)
    
    st.subheader("Response Complexity by User")
    df['response_length'] = df['response'].str.len()
    avg_response_length = df.groupby('name')['response_length'].mean().reset_index()
    fig = px.bar(avg_response_length,
                 x='name',
                 y='response_length',
                 title="Average Response Length by User",
                 labels={'response_length': 'Average Characters', 
                        'name': 'User'})
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Admin Dashboard", layout="wide")
    
    # Initialize ResponseEvaluator in session state if not already present
    if 'response_evaluator' not in st.session_state:
        st.session_state.response_evaluator = ResponseEvaluator()
        logger.info("Initialized ResponseEvaluator in session state")
    
    # Add auto-refresh
    st.cache_data.clear()
    auto_refresh = st.sidebar.checkbox('Enable auto-refresh', value=True)
    if auto_refresh:
        st.empty()
        time.sleep(30)
        st.rerun()
    
    if not check_password():
        st.stop()
        
    st.title("PGaaS Admin Dashboard")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    df = load_response_data(logs_dir)
    
    if df.empty:
        st.error("No data found in logs directory")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Metrics",
        "Document Analysis",
        "User Analysis",
        "Response Quality"
    ])
    
    with tab1:
        create_performance_dashboard(df)
    with tab2:
        create_document_usage_analysis(df)
    with tab3:
        create_user_analysis(df)
    with tab4:
        display_response_metrics()

if __name__ == "__main__":
    main()