import sys
from pathlib import Path
import streamlit as st
from transformers import AutoModel, AutoTokenizer
import subprocess
import threading
from pathway.xpacks.llm.vector_store import VectorStoreClient
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
import os
from LLM_Agent.LLM_Agent import LLMAgent
from rerankers.models.models import colBERT
# from ..rerankers.models.models import colBERT

PATHWAY_PORT = 8765
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# Add the project folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


# Function to start the VectorStoreServer
def start_vector_store_server():
    try:
        # Run setup.py
        setup_process = subprocess.run([sys.executable, '../DataBase/setup.py'], check=True)
        setup_process.check_returncode()  # Ensure setup.py ran successfully

        st.success("setup.py and experiment.py executed successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while running the scripts: {e}")

server_thread = threading.Thread(target=start_vector_store_server)
server_thread.daemon = True 
server_thread.start()

# Connect to the VectorStoreClient
client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

from RAG import RAG  # Import your RAG class
# Define cached loading functions for each model
@st.cache_resource
def load_bge_m3():
    model = AutoModel.from_pretrained("BAAI/bge-m3")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    return model, tokenizer

@st.cache_resource
def load_smol_lm():
    return HuggingFaceHub(
        repo_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        model_kwargs = {"temperature":0.5, "max_length": 64, "max_new_tokens": 512}
    )
    model = AutoModel.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    return model, tokenizer

@st.cache_resource
def load_colbert():
    model = colBERT()
    # model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    # tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    return model, None

# Load all models
bge_m3_model, bge_m3_tokenizer = load_bge_m3()
smol_lm_model = load_smol_lm()
gemini_model = LLMAgent(google_api_key="AIzaSyDMrrCnkl9tu3sJDHq4C-LYu0qHdbXAA00")
colbert_model, colbert_tokenizer = load_colbert()

# Initialize your RAG pipeline using these cached models
rag = RAG(vb=client, llm=gemini_model)

# Initialize session state for question history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit interface
st.title("Chatbot using RAG Pipeline")

# Sidebar for configuration
st.sidebar.header("Pipeline Configuration")
retrieval_mode = st.sidebar.selectbox("Select Retrieval Mode", ["simple", "intermediate", "complex"])
reranker_mode = st.sidebar.selectbox("Select Reranker Mode", ["simple", "intermediate", "complex"])

# Configure the RAG pipeline with your parser
rag.retrieval_agent_prep(
    q_model=smol_lm_model,
    parser=StrOutputParser(),  # Replace this with the actual parser you provide
    reranker=colbert_model,
    mode=retrieval_mode
)
rag.reranker_prep(reranker=colbert_model, mode=reranker_mode)

# Main chat interface using `st.chat`
st.header("Ask the Chatbot")

# Streamlit chat messages loop
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Bot:** {entry['answer']}")

# Process new question if entered
if question := st.chat_input("Type your question here:"):
    with st.spinner("Retrieving answer..."):
        # Get the answer from the RAG pipeline
        answer = rag.query(question)
        print(type(answer))
        # Store question and answer in the session history
        st.session_state.history.append({"question": question, "answer": answer})

        # Display the current answer immediately in the chat
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot:** {answer}")

# Sidebar footer
st.sidebar.text("RAG Pipeline Chatbot with Streamlit")
st.sidebar.text("Developed with LangGraph & LangChain")