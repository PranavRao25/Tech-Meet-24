import os
from tabulate import tabulate
import json
import sys
import smtplib
from pathlib import Path
import pandas as pd
import altair as alt
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from transformers import pipeline
from pathway.xpacks.llm.vector_store import VectorStoreClient
from langchain_core.output_parsers import StrOutputParser
import os
from pathway.xpacks.llm.vector_store import VectorStoreClient
import toml
import torch
from langchain_community.llms import HuggingFaceHub
# from ..rerankers.models.models import colBERT

PATHWAY_PORT = 8765
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(str(Path(__file__).resolve().parent.parent))
# Add the project folder to the Python path

from RAG import RAG
from AutoWrapper import AutoWrapper
from LLM_Agent.OAI_LLM_Agent import LLMAgent
from rerankers.models.models import colBERT, BGE_M3

config = toml.load("../config.toml")
HF_TOKEN = config['HF_TOKEN']
GEMINI_API = config['GEMINI_API']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

class RetrieverClient:
    def __init__(self, host, port, k=10, timeout=60, *args, **kwargs):
        self.retriever = VectorStoreClient(host=host, port=port, timeout=timeout, *args, **kwargs)
        self.k = k
    def query(self, text:str):
        return self.retriever.query(text, k=self.k)
    __call__ = query

# Function to start the VectorStoreServer
def vb_prep():
    # return VectorStoreClient(
    #     host="127.0.0.1",
    #     port=PATHWAY_PORT,
    # )

    HOST = "127.0.0.1"
    PORT = 8666

    return RetrieverClient(host=HOST, port=PORT, k=20, timeout=120)

# Define cached loading functions for each model
def load_bge_m3():
    return BGE_M3(), None

def load_smol_lm(temp=0.5):
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": temp, "max_length": 64, "max_new_tokens": 512, "return_full_text":False}
    )
    
def load_colbert():
    model = colBERT()
    # model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
    # tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    return model, None

def load_thresholder(temp=0.3):
    return HuggingFaceHub( #change this to the correct model
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.3, "max_length": 64, "max_new_tokens": 16, "return_full_text":False}
    )

# Load all models
bge_m3_model, bge_m3_tokenizer = load_bge_m3()
smol_lm_model = load_smol_lm()
moe_model = load_thresholder(temp=0.3)
oai_model = LLMAgent(model_name="gpt-4o-mini", temperature=0.7, max_tokens=256)
colbert_model, colbert_tokenizer = load_colbert()
thresolder_model = load_thresholder()
client = vb_prep()

# Streamlit interface
# st.title(".pathway Chatbot")

# Initialize your RAG pipeline using these cached models
rag = RAG(vb=client, llm=oai_model)

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

rag.reranker_prep(reranker=colbert_model, mode="simple")
rag.reranker_prep(reranker=bge_m3_model, mode="intermediate")
rag.reranker_prep(reranker=bge_m3_model, mode="complex")
rag.moe_prep(moe_model)
rag.step_back_prompt_prep(model=smol_lm_model)
rag.web_search_prep(model=smol_lm_model)
rag.thresholder_prep(model=thresolder_model)
rag.set()

os.environ['OPENAI_API_KEY'] = ''

with open("./test50.json", 'r') as f:
	data = json.load(f)

list_qa = data['data'][:5]

questions = [qa['question'] for qa in list_qa]
answers = [qa['answer'] for qa in list_qa]
contexts = [qa['context'] for qa in list_qa]

results_df = rag.ragas_evaluate(questions, answers)
print(tabulate(results_df))
results_df.to_csv("test_results.csv")