import streamlit as st
import requests

st.set_page_config(page_title="Chatbot Interface", layout="centered")
st.title("RAG Chatbot")

if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

def query_rag_api(user_query):
    api_url = "https://your-api-endpoint.com/query"  # Replace with your RAG API endpoint
    headers = {"Content-Type": "application/json"}
    data = {"query": user_query}

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        result = response.json()
        return result.get("response", "Error: No response from RAG API")
    except Exception as e:
        return f"Error: {e}"

st.write("Enter your query below:")
user_query = st.text_input("")

if st.button("Send") and user_query:
    response = query_rag_api(user_query)
    st.session_state["conversation"].append(("User", user_query))
    st.session_state["conversation"].append(("Bot", response))

for speaker, message in st.session_state["conversation"]:
    if speaker == "User":
        st.markdown(f"**User:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
