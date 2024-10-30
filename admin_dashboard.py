# admin_dashboard_optimized.py

import streamlit as st
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import altair as alt
import gc
from typing import Dict, List
import html

# Configuration
CACHE_TTL = 3600  # 1 hour
MAX_RECORDS = 1000

def check_password():
    """Returns `True` if the user had the correct password."""
    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", key="password",
                     on_change=lambda: setattr(st.session_state, "password_correct", 
                     st.session_state.password == st.secrets["admin_password"]))
        return False
    return st.session_state.password_correct

@st.cache_data(ttl=CACHE_TTL)
def load_response_data(logs_dir: str) -> List[Dict]:
    """Load data using CSV reader instead of pandas"""
    data = []
    for filename in os.listdir(logs_dir):
        if not filename.endswith('_response_log.csv'):
            continue
            
        file_path = os.path.join(logs_dir, filename)
        try:
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({
                        'name': row['name'],
                        'date': datetime.strptime(row['date'], '%d-%m-%Y'),
                        'time': row['time'],
                        'question': row['question'],
                        'response': row['response'],
                        'unique_files': row['unique_files'].split(' - ') if row['unique_files'] else [],
                        'chunk_scores': [row.get(f'chunk{i}_score', '') for i in range(1, 4)]
                    })
                    if len(data) >= MAX_RECORDS:
                        break
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
    return data

def create_user_metrics(data: List[Dict]) -> List[Dict]:
    """Calculate user metrics"""
    user_stats = defaultdict(lambda: {'questions': 0, 'active_days': set()})
    
    for row in data:
        name = row['name']
        user_stats[name]['questions'] += 1
        user_stats[name]['active_days'].add(row['date'].date())
    
    return [
        {
            'user': name,
            'questions': stats['questions'],
            'active_days': len(stats['active_days'])
        }
        for name, stats in user_stats.items()
    ]

def display_user_analysis(data: List[Dict]):
    """Display user interaction metrics using Altair"""
    user_metrics = create_user_metrics(data)
    
    # Questions per user chart
    chart = alt.Chart(user_metrics).mark_bar().encode(
        x=alt.X('user:N', title='User'),
        y=alt.Y('questions:Q', title='Number of Questions'),
        tooltip=['user', 'questions', 'active_days']
    ).properties(
        title='User Engagement',
        width=600
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Display metrics table
    st.dataframe(user_metrics)

def analyze_document_usage(data: List[Dict]):
    """Analyze document usage patterns"""
    doc_usage = defaultdict(int)
    
    for row in data:
        for doc in row['unique_files']:
            doc_usage[doc.strip()] += 1
    
    usage_data = [
        {'document': doc, 'count': count}
        for doc, count in sorted(doc_usage.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
    ]
    
    chart = alt.Chart(usage_data).mark_bar().encode(
        x=alt.X('document:N', title='Document'),
        y=alt.Y('count:Q', title='Usage Count'),
        tooltip=['document', 'count']
    ).properties(
        title='Top 10 Most Used Documents',
        width=600
    )
    
    st.altair_chart(chart, use_container_width=True)

def analyze_response_patterns(data: List[Dict]):
    """Analyze response patterns over time"""
    daily_counts = defaultdict(int)
    
    for row in data:
        daily_counts[row['date'].date()] += 1
    
    time_data = [
        {'date': date, 'responses': count}
        for date, count in sorted(daily_counts.items())
    ]
    
    chart = alt.Chart(time_data).mark_line().encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('responses:Q', title='Number of Responses'),
        tooltip=['date', 'responses']
    ).properties(
        title='Daily Response Pattern',
        width=600
    )
    
    st.altair_chart(chart, use_container_width=True)

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
        
    # Use tabs for better organization
    tab1, tab2, tab3 = st.tabs([
        "User Analysis",
        "Document Analysis",
        "Response Patterns"
    ])
    
    with tab1:
        st.header("User Analysis")
        display_user_analysis(data)
        cleanup_memory()
        
    with tab2:
        st.header("Document Analysis")
        analyze_document_usage(data)
        cleanup_memory()
        
    with tab3:
        st.header("Response Patterns")
        analyze_response_patterns(data)
        cleanup_memory()
    
    # Final cleanup
    cleanup_memory()

if __name__ == "__main__":
    main()