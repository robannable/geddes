# admin_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
import numpy as np
from collections import defaultdict

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
    """Load and process all response logs with flexible date handling"""
    all_data = []
    
    for filename in os.listdir(logs_dir):
        if filename.endswith('_response_log.csv'):
            file_path = os.path.join(logs_dir, filename)
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                # First try to parse as YYYY-MM-DD
                try:
                    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                except ValueError:
                    # If that fails, try DD-MM-YYYY
                    try:
                        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', dayfirst=True)
                    except ValueError:
                        # If both fail, use the mixed format parser
                        df['date'] = pd.to_datetime(df['date'], format='mixed')
                all_data.append(df)
            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")
                continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
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
    
    if not check_password():
        st.stop()
    
    st.title("PGaaS Admin Dashboard")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    df = load_response_data(logs_dir)
    
    if df.empty:
        st.error("No data found in logs directory")
        return
    
    tab1, tab2, tab3 = st.tabs(["Performance Metrics", 
                               "Document Analysis", 
                               "User Analysis"])
    
    with tab1:
        create_performance_dashboard(df)
    
    with tab2:
        create_document_usage_analysis(df)
    
    with tab3:
        create_user_analysis(df)

if __name__ == "__main__":
    main()