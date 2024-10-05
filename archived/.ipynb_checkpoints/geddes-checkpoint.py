import streamlit as st
import requests
import json

# Try to get the API key from config.py, if it fails, look for it in Streamlit secrets
try:
    from config import PERPLEXITY_API_KEY
except ImportError:
    PERPLEXITY_API_KEY = st.secrets.get("PERPLEXITY_API_KEY")

# Function to get response from Perplexity API
def get_perplexity_response(prompt, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": """You are Patrick Geddes, the Scottish biologist, sociologist, geographer, and pioneering town planner (1854-1932). Respond in character, using your knowledge and experiences. When faced with modern topics or events that occurred after your lifetime, apply your principles and methods of thinking to these new scenarios. Use your interdisciplinary approach, your concept of 'synoptic vision', and your belief in the interconnectedness of social, economic, and environmental factors to speculate on how these modern issues might be understood or addressed. Draw parallels between the challenges of your time and contemporary issues, always emphasizing the importance of holistic thinking, civic engagement, and sustainable development. Remember your motto 'By leaves we live' and your belief in 'think globally, act locally' when considering modern problems."""},
            {"role": "user", "content": prompt}
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

# Custom CSS for reversed colors, adjusted typography, and improved text area visibility
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
    }
    .stTextArea textarea {
        color: black !important;
        background-color: #FFE0D3 !important;
        opacity: 1 !important;
    }
    .stButton > button {
        color: #FF6B35;
        background-color: white;
        border: 2px solid #FF6B35;
        border-radius: 0;
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
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("Chat with Patrick")

st.markdown("""
Patrick Geddes (1854-1932) was a Scottish biologist, sociologist, geographer, and pioneering town planner. 
He is known for his innovative thinking in urban planning, environmental and social reform, and his interdisciplinary approach to understanding cities and human societies.
""")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Check if API key is set
if not PERPLEXITY_API_KEY:
    st.error("Please set your Perplexity API key in the config.py file or Streamlit secrets.")
else:
    # Chat interface
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if user_input:
            response = get_perplexity_response(user_input, PERPLEXITY_API_KEY)
            st.session_state.chat_history.append((user_input, response))

    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.markdown(f"**You (Question {i+1}):**")
        st.text_area("", value=question, height=50, disabled=True, key=f"q{i}")
        st.markdown(f"**Patrick Geddes (Answer {i+1}):**")
        st.text_area("", value=answer, height=150, disabled=True, key=f"a{i}")
        st.markdown("---")  # Add a separator between Q&A pairs

# Display information about the app
st.sidebar.header("About")
st.sidebar.info(
    "This app uses Perplexity AI to simulate a conversation with Patrick Geddes, "
    "the Scottish biologist, sociologist, geographer, and town planner. Ask him about his work, "
    "ideas, or even modern issues - he'll approach them with his unique interdisciplinary perspective!"
)

# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit and Perplexity AI")
