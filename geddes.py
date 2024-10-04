import streamlit as st
import requests
import json

# Directly include the API key (replace with your actual API key)
perplexity_api_key = "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"

def get_perplexity_response(prompt, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are Patrick Geddes, the Scottish biologist, geographer, and town planner. Respond in character, using your knowledge and experiences. If asked about modern topics or events after your lifetime, politely explain that you can't comment on future events."},
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
        return f"Error: API request failed - {str(e)}"

# Streamlit UI
st.title("Chat with Patrick Geddes")

st.markdown("""
Patrick Geddes (1854-1932) was a Scottish biologist, geographer, and pioneering town planner. 
He is known for his innovative thinking in urban planning and sociological studies.
""")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Check if API key is set
if not perplexity_api_key or perplexity_api_key == "your_perplexity_api_key_here":
    st.error("Please replace 'your_perplexity_api_key_here' with your actual Perplexity API key in the script.")
else:
    # Chat interface
    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if user_input:
            response = get_perplexity_response(user_input, perplexity_api_key)
            st.session_state.chat_history.append((user_input, response))

    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.text_area(f"You (Question {i+1}):", value=question, height=50, disabled=False)
        st.text_area(f"Patrick Geddes (Answer {i+1}):", value=answer, height=200, disabled=False)
        st.markdown("---")  # Add a separator between Q&A pairs

# Display information about the app
st.sidebar.header("About")
st.sidebar.info(
    "This app uses Perplexity AI to simulate a conversation with Patrick Geddes, "
    "the Scottish biologist, geographer, and town planner. Ask him about his work, "
    "ideas, or life experiences!"
)

# Add a footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created with Streamlit and Perplexity AI")
