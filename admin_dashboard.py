# admin_dashboard_optimized.py

import streamlit as st
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import altair as alt
import pandas as pd
import gc

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
def load_response_data(logs_dir):
    """Load data using CSV reader with minimal memory usage"""
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
                        'question': row['question'],
                        'response': row['response'],
                        'unique_files': row['unique_files'].split(' - ') if row['unique_files'] else [],
                        'chunk_scores': [row.get(f'chunk{i}_score', '') for i in range(1, 4)]
                    })
                    if len(data) >= MAX_RECORDS:
                        break
        except Exception as e:
            st.error(f"Error reading {filename}: {str(e)}")
    return pd.DataFrame(data)

def display_user_interactions(df):
    """Display user interaction metrics using Altair"""
    st.subheader("User Interaction Patterns")
    
    # Calculate user statistics
    user_stats = df.groupby('name').agg({
        'question': 'count',
        'date': 'nunique'
    }).reset_index()
    user_stats.columns = ['User', 'Total_Questions', 'Active_Days']
    
    # Create Altair chart
    chart = alt.Chart(user_stats).mark_bar().encode(
        x=alt.X('User:N', title='User'),
        y=alt.Y('Total_Questions:Q', title='Number of Questions'),
        tooltip=['User', 'Total_Questions', 'Active_Days']
    ).properties(
        title='User Engagement',
        width=600
    )
    
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(user_stats)

def create_document_usage_analysis(df):
    """Analyze document usage patterns"""
    st.header("Document Usage Analysis")
    
    # Calculate document usage
    doc_usage = defaultdict(int)
    for files in df['unique_files'].dropna():
        for doc in files:
            doc_usage[doc.strip()] += 1
    
    # Create DataFrame for visualization
    doc_df = pd.DataFrame({
        'Document': list(doc_usage.keys()),
        'Usage_Count': list(doc_usage.values())
    }).sort_values('Usage_Count', ascending=False)
    
    # Create Altair chart
    chart = alt.Chart(doc_df).mark_bar().encode(
        x=alt.X('Document:N', sort='-y'),
        y=alt.Y('Usage_Count:Q'),
        tooltip=['Document', 'Usage_Count']
    ).properties(
        title='Document Usage Frequency',
        width=600
    )
    
    st.altair_chart(chart, use_container_width=True)

def analyze_response_patterns(df):
    """Analyze response patterns over time"""
    st.header("Response Patterns")
    
    # Calculate daily response counts
    daily_counts = df.groupby(df['date'].dt.date).size().reset_index()
    daily_counts.columns = ['Date', 'Count']
    
    # Create Altair chart
    chart = alt.Chart(daily_counts).mark_line().encode(
        x=alt.X('Date:T'),
        y=alt.Y('Count:Q'),
        tooltip=['Date', 'Count']
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
        df = load_response_data(logs_dir)
        
    if df.empty:
        st.error("No data found in logs directory")
        return
        
    # Use tabs for better organization
    tab1, tab2, tab3 = st.tabs([
        "User Analysis",
        "Document Analysis",
        "Response Patterns"
    ])
    
    with tab1:
        display_user_interactions(df)
        cleanup_memory()
        
    with tab2:
        create_document_usage_analysis(df)
        cleanup_memory()
        
    with tab3:
        analyze_response_patterns(df)
        cleanup_memory()
    
    # Final cleanup
    cleanup_memory()

if __name__ == "__main__":
    main()