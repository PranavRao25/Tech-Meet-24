import sys
from pathlib import Path
import streamlit as st
from transformers import AutoModel, AutoTokenizer
import subprocess
from pathway.xpacks.llm.vector_store import VectorStoreClient
from langchain_core.output_parsers import StrOutputParser

PATHWAY_PORT = 8765

# Add the project folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Function to run setup.py and experiment.py
def run_scripts():
    try:
        # Run setup.py
        setup_process = subprocess.run([sys.executable, 'Database/setup.py'], check=True)
        setup_process.check_returncode()  # Ensure setup.py ran successfully

        st.success("setup.py and experiment.py executed successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred while running the scripts: {e}")

# Run the setup.py and experiment.py before initializing the rest of the application
run_scripts()
client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)
from RAG import RAG  # Import your RAG class

# Define cached loading functions for each model
@st.cache_resource
def load_bge_m3():
    model = AutoModel.from_pretrained("bge-m3")
    tokenizer = AutoTokenizer.from_pretrained("bge-m3")
    return model, tokenizer

@st.cache_resource
def load_smol_lm():
    model = AutoModel.from_pretrained("smol-LM")
    tokenizer = AutoTokenizer.from_pretrained("smol-LM")
    return model, tokenizer

@st.cache_resource
def load_colbert():
    model = AutoModel.from_pretrained("ColBERT")
    tokenizer = AutoTokenizer.from_pretrained("ColBERT")
    return model, tokenizer

# Load all models
bge_m3_model, bge_m3_tokenizer = load_bge_m3()
smol_lm_model, smol_lm_tokenizer = load_smol_lm()
colbert_model, colbert_tokenizer = load_colbert()

# Initialize your RAG pipeline using these cached models
rag = RAG(vb=client, llm=smol_lm_model)

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
    st.chat.message(f"**You:** {entry['question']}")
    st.chat.message(f"**Bot:** {entry['answer']}", is_user=False)

# Process new question if entered
if question := st.chat_input("Type your question here:"):
    with st.spinner("Retrieving answer..."):
        # Get the answer from the RAG pipeline
        answer = rag.query(question)

        # Store question and answer in the session history
        st.session_state.history.append({"question": question, "answer": answer})

        # Display the current answer immediately in the chat
        st.chat.message(f"**You:** {question}")
        st.chat.message(f"**Bot:** {answer}", is_user=False)

        # Option to evaluate the answer quality
        # if st.checkbox("Evaluate Answer Quality"):
        #     with st.spinner("Running RAGAS evaluation..."):
        #         evaluation_df = rag.ragas_evaluate(raise_exceptions=True)
        #         st.subheader("Evaluation Metrics")
        #         st.write(evaluation_df)

# Sidebar footer
st.sidebar.text("RAG Pipeline Chatbot with Streamlit")
st.sidebar.text("Developed with LangGraph & LangChain")
