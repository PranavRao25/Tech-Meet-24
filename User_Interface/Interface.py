import sys
from pathlib import Path
import streamlit as st
from transformers import pipeline
from pathway.xpacks.llm.vector_store import VectorStoreClient
from langchain_core.output_parsers import StrOutputParser
import os
from pathway.xpacks.llm.vector_store import VectorStoreClient
import toml
import torch
# from ..rerankers.models.models import colBERT

PATHWAY_PORT = 8765
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add the project folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from RAG import RAG
from AutoWrapper import AutoWrapper
from LLM_Agent.LLM_Agent import LLMAgent
from rerankers.models.models import colBERT, BGE_M3

config = toml.load("../config.toml")
HF_TOKEN = config['HF_TOKEN']
GEMINI_API = config['GEMINI_API']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Function to start the VectorStoreServer
@st.cache_resource
def vb_prep():
    # return VectorStoreClient(
    #     host="127.0.0.1",
    #     port=PATHWAY_PORT,
    # )

    HOST = "127.0.0.1"
    PORT = 8666

    return VectorStoreClient(host=HOST, port=PORT)

# Define cached loading functions for each model
@st.cache_resource
def load_bge_m3():
    return BGE_M3(DEVICE), None

@st.cache_resource
def load_smol_lm():
    return AutoWrapper("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    
@st.cache_resource
def load_smol_lms():
    return "HuggingFaceTB/SmolLM2-1.7B-Instruct"

@st.cache_resource
def load_colbert():
    model = colBERT(DEVICE)
    # model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    # tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    return model, None

@st.cache_resource
def load_moe():
    return "microsoft/deberta-v3-small"


# Load all models
bge_m3_model, bge_m3_tokenizer = load_bge_m3()
smol_lm_model = load_smol_lm()
moe_model = load_moe()
gemini_model = LLMAgent(google_api_key=GEMINI_API)
colbert_model, colbert_tokenizer = load_colbert()
client = vb_prep()

# Initialize session state for question history if it doesn't exist
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit interface
st.title(".pathway Chatbot")

# Sidebar for configuration
st.sidebar.header("Pipeline Configuration")
retrieval_mode = st.sidebar.selectbox("Select Retrieval Mode", ["simple", "intermediate", "complex"])
reranker_mode = st.sidebar.selectbox("Select Reranker Mode", ["simple", "intermediate", "complex"])

# Initialize your RAG pipeline using these cached models
rag = RAG(vb=client, llm=gemini_model)

# Configure the RAG pipeline with your parser
rag.retrieval_agent_prep(
    q_model=smol_lm_model,
    parser=StrOutputParser(),  # Replace this with the actual parser you provide
    reranker=colbert_model,
    mode="simple"
)

rag.retrieval_agent_prep(
    q_model=smol_lm_model,
    parser=StrOutputParser(),  # Replace this with the actual parser you provide
    reranker=colbert_model,
    mode="intermediate"
)

rag.retrieval_agent_prep(
    q_model=smol_lm_model,
    parser=StrOutputParser(),  # Replace this with the actual parser you provide
    reranker=colbert_model,
    mode="complex"
)

# suppot them bgem3 shit
rag.reranker_prep(reranker=colbert_model, mode="simple")
rag.reranker_prep(reranker=bge_m3_model, mode="intermediate")
rag.reranker_prep(reranker=bge_m3_model, mode="complex")
rag.moe_prep(moe_model)
rag.step_back_prompt_prep(model=smol_lm_model)
rag.web_search_prep()
rag.set()

# Main chat interface using `st.chat`
st.header("Team_41")

# Streamlit chat messages loop
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Bot:** {entry['answer']}")

# Process new question if entered
if question := st.chat_input("Type your question here:"):
    with st.spinner("Retrieving answer..."):
        # Get the answer from the RAG pipeline
        answer = rag.query(question)
        # Store question and answer in the session history
        st.session_state.history.append({"question": question, "answer": answer})

        # Display the current answer immediately in the chat
        st.markdown(f"**You:** {question}")
        st.markdown(f"**Bot:** {answer}")

# Sidebar footer
st.sidebar.text("RAG Pipeline Chatbot with Streamlit")
st.sidebar.text("Developed with LangGraph & LangChain")
