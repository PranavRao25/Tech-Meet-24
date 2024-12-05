import sys
import smtplib
from pathlib import Path
import pandas as pd
import altair as alt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import streamlit as st
from transformers import pipeline
from pathway.xpacks.llm.vector_store import VectorStoreClient
from langchain_core.output_parsers import StrOutputParser
import os
from pathway.xpacks.llm.vector_store import VectorStoreClient
import toml
import torch
from langchain_community.llms import HuggingFaceHub
from langchain.llms.ollama import Ollama
# from ..rerankers.models.models import colBERT

PATHWAY_PORT = 8765
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Add the project folder to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from RAG import RAG
from AutoWrapper import AutoWrapper
from LLM_Agent.LLM_Agent import LLMAgent
from rerankers.models.models import colBERT, BGE_M3
from MOE.llm_query_classifier import QueryClassifier

config = toml.load("../config.toml")
HF_TOKEN = config['HF_TOKEN']
GEMINI_API = config['GEMINI_API']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Your SMTP details
SMTP_SERVER = "smtp.gmail.com"  # Example for Gmail
SMTP_PORT = 587
SENDER_EMAIL = "your_email@example.com"
SENDER_PASSWORD = "your_email_password"
RECIPIENT_EMAIL = "recipient_email@example.com"

class RetrieverClient:
    def __init__(self, host, port, k=10, timeout=60, *args, **kwargs):
        self.retriever = VectorStoreClient(host=host, port=port, timeout=timeout, *args, **kwargs)
        self.k = k
    def query(self, text:str):
        return self.retriever.query(text, k=self.k)
    __call__ = query

# Function to start the VectorStoreServer
@st.cache_resource
def vb_prep():
    # return VectorStoreClient(
    #     host="127.0.0.1",
    #     port=PATHWAY_PORT,
    # )

    HOST = "127.0.0.1"
    PORT = 8666

    return RetrieverClient(host=HOST, port=PORT, k=10, timeout=120)

# Define cached loading functions for each model
@st.cache_resource
def load_bge_m3():
    return BGE_M3(), None

@st.cache_resource
def load_smol_lm(temp=0.5):
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": temp, "max_length": 64, "max_new_tokens": 512, "return_full_text":False}
    )
    
@st.cache_resource
def load_colbert():
    model = colBERT()
    # model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    # tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    return model, None

@st.cache_resource
def load_thresholder():
    return HuggingFaceHub( #change this to the correct model
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.3, "max_length": 64, "max_new_tokens": 512, "return_full_text":False}
    )

# Load all models
bge_m3_model, bge_m3_tokenizer = load_bge_m3()
smol_lm_model = load_smol_lm()
moe_model = load_smol_lm(temp=0.3)
gemini_model = LLMAgent(google_api_key=GEMINI_API)
colbert_model, colbert_tokenizer = load_colbert()
thresolder_model = load_thresholder()
client = vb_prep()

# Streamlit interface
st.title(".pathway Chatbot")

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
rag.web_search_prep(model=smol_lm_model)
rag.thresholder_prep(model=thresolder_model)
rag.set()

# Sidebar for configuration
st.sidebar.header("Pipeline Configuration")
# retrieval_mode = st.sidebar.selectbox("Select Retrieval Mode", ["simple", "intermediate", "complex"])
# reranker_mode = st.sidebar.selectbox("Select Reranker Mode", ["simple", "intermediate", "complex"])

# Function to send email
def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        st.success("Report sent successfully!")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")

# Initialize session state for question history and feedback counters
if "history" not in st.session_state:
    st.session_state.history = []
if "likes" not in st.session_state:
    st.session_state.likes = 0
if "dislikes" not in st.session_state:
    st.session_state.dislikes = 0
if "positive_feedback" not in st.session_state:
    st.session_state["positive_feedback"] = 0
if "negative_feedback" not in st.session_state:
    st.session_state["negative_feedback"] = 0

# Main chat interface using `st.chat`
st.header("Team_41")

# Streamlit chat messages loop
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Bot:** {entry['answer']}")

# Process new question if entered
if question := st.chat_input("Type your question here:"):
    # Immediately display the question
    st.markdown(f"**You:** {question}")

    with st.spinner("Processing..."):
        # Get the answer from the RAG pipeline
        answer = rag.query(question)

        # Store question and answer in the session history
        st.session_state.history.append({"question": question, "answer": answer})

        # Display the answer after processing
        st.markdown(f"**Bot:** {answer}")

    # Add thumbs-up and thumbs-down buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Like"):
            st.session_state.likes += 1  # Increment likes

    with col2:
        if st.button("👎 Dislike"):
            st.session_state.dislikes += 1  # Increment dislikes

    # Add additional feedback (Yes/No)
    col1, col2, _ = st.columns([1, 1, 6])  # Adjust proportions for reduced spacing
    with col1:
        if st.button("👍 Yes"):
            st.session_state["positive_feedback"] += 1
            # st.success("Thank you for your feedback! 😊")

    with col2:
        if st.button("👎 No"):
            st.session_state["negative_feedback"] += 1
            # st.error("Thank you for your feedback! We will improve. 😔")

# Sidebar: Display Feedback Stats
st.sidebar.header("Feedback Stats")
st.sidebar.write(f"👍 Likes: {st.session_state.likes}")
st.sidebar.write(f"👎 Dislikes: {st.session_state.dislikes}")

# Add the Report Button
if st.button("Send Report"):
    # Compile the report body with history, likes, and dislikes
    report_body = "Chat History:\n"
    for entry in st.session_state.history:
        report_body += f"You: {entry['question']}\nBot: {entry['answer']}\n\n"
    
    report_body += f"Likes: {st.session_state.likes}\nDislikes: {st.session_state.dislikes}\n"
    report_body += f"Positive Feedback: {st.session_state['positive_feedback']}\nNegative Feedback: {st.session_state['negative_feedback']}"

    # Send the email report
    send_email("Chatbot Interaction Report", report_body)

# Data for Pie Chart
feedback_data = pd.DataFrame({
    "Feedback": ["Positive", "Negative"],
    "Count": [
        st.session_state["positive_feedback"],
        st.session_state["negative_feedback"]
    ]
})

# Sidebar footer
st.sidebar.text("RAG Pipeline Chatbot with Streamlit")
st.sidebar.text("Developed with LangGraph & LangChain")

# Sidebar: Pie Chart
st.sidebar.title("Feedback Summary")
if feedback_data["Count"].sum() > 0:
    chart = alt.Chart(feedback_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Feedback", type="nominal", scale=alt.Scale(range=["green", "red"])),
        tooltip=["Feedback", "Count"]
    ).properties(
        width=250,
        height=250
    )
    st.sidebar.altair_chart(chart, use_container_width=True)
else:
    st.sidebar.write("No feedback recorded yet.")
