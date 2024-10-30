# admin_dashboard_optimized.py

import streamlit as st
from collections import defaultdict
from datetime import datetime, timedelta
import os
import csv
from typing import Dict, List, Tuple
import altair as alt
import sqlite3
import gc

# Memory optimization settings
CHUNK_SIZE = 1000
CACHE_TTL = 3600  # 1 hour
MAX_RECORDS_DISPLAY = 1000

def check_password():
    """Simple password protection with minimal memory usage"""
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", key="password",
                     on_change=lambda: setattr(st.session_state, "password_correct", 
                     st.session_state.password == st.secrets["admin_password"]))
        return False
    return st.session_state.password_correct

@st.cache_data(ttl=CACHE_TTL)
def load_response_data(logs_dir: str) -> List[Dict]:
    """Load data in chunks using CSV reader instead of pandas"""
    data = []
    for filename in os.listdir(logs_dir):
        if not filename.endswith('_response_log.csv'):
            continue
            
        file_path = os.path.join(logs_dir, filename)
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(row)
                    if len(data) >= MAX_RECORDS_DISPLAY:
                        break
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
            
    return data

def process_chunk_scores(data: List[Dict]) -> List[Tuple[float, str]]:
    """Process chunk scores without pandas"""
    scores = []
    for row in data:
        for chunk_key in ['chunk1_score', 'chunk2_score', 'chunk3_score']:
            if chunk_key in row and row[chunk_key]:
                try:
                    parts = row[chunk_key].split(' - ')
                    if len(parts) >= 2:
                        scores.append((float(parts[0]), parts[1]))
                except:
                    continue
    return scores

def display_chunk_analysis(data: List[Dict]):
    """Display chunk analysis using Altair"""
    scores = process_chunk_scores(data)
    if scores:
        # Aggregate scores
        score_dict = defaultdict(list)
        for score, file in scores:
            score_dict[file].append(score)
            
        # Calculate averages
        avg_scores = [
            {"file": file, "avg_score": sum(scores)/len(scores), 
             "count": len(scores)}
            for file, scores in score_dict.items()
        ]
        
        # Create Altair chart
        chart = alt.Chart(avg_scores).mark_bar().encode(
            x='file:N',
            y='avg_score:Q',
            color='count:Q'
        ).properties(
            title="Average Relevance Score by Document"
        )
        st.altair_chart(chart)

def display_user_interactions(data: List[Dict]):
    """Display user interactions using Altair"""
    user_stats = defaultdict(lambda: {"questions": 0, "days": set()})
    
    for row in data:
        name = row['name']
        user_stats[name]["questions"] += 1
        user_stats[name]["days"].add(row['date'])
    
    user_data = [
        {"user": name, "questions": stats["questions"], 
         "active_days": len(stats["days"])}
        for name, stats in user_stats.items()
    ]
    
    chart = alt.Chart(user_data).mark_bar().encode(
        x='user:N',
        y='questions:Q'
    ).properties(
        title="Questions per User"
    )
    st.altair_chart(chart)

def create_document_usage_analysis(data: List[Dict]):
    """Analyze document usage with minimal memory footprint"""
    doc_usage = defaultdict(int)
    
    for row in data:
        if 'unique_files' in row and row['unique_files']:
            for doc in str(row['unique_files']).split(' - '):
                doc_usage[doc.strip()] += 1
    
    chart_data = [
        {"document": doc, "count": count}
        for doc, count in sorted(doc_usage.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
    ]
    
    chart = alt.Chart(chart_data).mark_bar().encode(
        x='document:N',
        y='count:Q'
    ).properties(
        title="Top 10 Most Used Documents"
    )
    st.altair_chart(chart)

def cleanup_memory():
    """Force garbage collection"""
    gc.collect()

def main():
    st.set_page_config(page_title="Admin Dashboard", layout="wide")
    
    if not check_password():
        st.stop()
        
    st.title("PGaaS Admin Dashboard (Optimized)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    
    with st.spinner("Loading data..."):
        data = load_response_data(logs_dir)
        
    if not data:
        st.error("No data found in logs directory")
        return
        
    # Use tabs for organization but load data only when tab is selected
    tab1, tab2 = st.tabs(["Usage Metrics", "Document Analysis"])
    
    with tab1:
        st.header("Usage Metrics")
        display_user_interactions(data)
        cleanup_memory()
        
    with tab2:
        st.header("Document Analysis")
        create_document_usage_analysis(data)
        cleanup_memory()
        
    # Cleanup at the end
    cleanup_memory()

if __name__ == "__main__":
    main()