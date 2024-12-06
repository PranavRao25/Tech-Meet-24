# from flask import Flask, render_template, request

# app = Flask(__name__)

# # Store input and output history
# lst = []

# @app.route("/", methods=['POST', 'GET'])
# def main():
#     global lst
#     if request.method == "POST":
#         st = str(request.form.get('intext'))
#         mode = request.form.get('mode')
#         print(mode)
#         output = model_function(st)
#         lst.append((st, output))
#         return render_template("index.html", ans=lst)
#     else:
#         return render_template("index.html", ans=lst)

# def model_function(inp):
#     import time
#     time.sleep(3)  
#     return f"Processed: {inp}"

# if __name__ == "__main__":
#     app.run(debug=True)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask import Flask, render_template, request, session
from transformers import pipeline
import toml
import os
import torch
from pathway.xpacks.llm.vector_store import VectorStoreClient
from RAG import RAG
from rerankers.models.models import colBERT, BGE_M3
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from LLM_Agent.LLM_Agent import LLMAgent

class RetrieverClient:
    def __init__(self, host, port, k=10, timeout=60, *args, **kwargs):
        self.retriever = VectorStoreClient(host=host, port=port, timeout=timeout, *args, **kwargs)
        self.k = k
    def query(self, text:str):
        return self.retriever.query(text, k=self.k)
    __call__ = query

app = Flask(__name__)
app.secret_key = "supersecretkey"  # For session management

# Configuration
config = toml.load("../config.toml")
HF_TOKEN = config['HF_TOKEN']
GEMINI_API = config['GEMINI_API']
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize RAG Pipeline
def initialize_rag():
    retriever_client = RetrieverClient(host="127.0.0.1", port=8666, timeout=120)

    bge_m3_model = BGE_M3()
    colbert_model = colBERT()
    smol_lm_model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512, "return_full_text": False},
    )

    gemini_model = LLMAgent(google_api_key=GEMINI_API)
    thresolder_moe_model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.3, "max_length": 64, "max_new_tokens": 512, "return_full_text": False},
    )
    # Configure RAG Pipeline
    rag = RAG(vb=retriever_client, llm=gemini_model)

    rag.retrieval_agent_prep(
        q_model=smol_lm_model, parser=StrOutputParser(), reranker=colbert_model, mode="simple"
    )
    rag.retrieval_agent_prep(
        q_model=smol_lm_model, parser=StrOutputParser(), reranker=colbert_model, mode="intermediate"
    )
    rag.retrieval_agent_prep(
        q_model=smol_lm_model, parser=StrOutputParser(), reranker=colbert_model, mode="complex"
    )

    rag.reranker_prep(reranker=colbert_model, mode="simple")
    rag.reranker_prep(reranker=bge_m3_model, mode="intermediate")
    rag.reranker_prep(reranker=bge_m3_model, mode="complex")
    rag.moe_prep(thresolder_moe_model)
    rag.step_back_prompt_prep(model=smol_lm_model)
    rag.web_search_prep(model=smol_lm_model)
    rag.thresholder_prep(model=thresolder_moe_model)
    rag.set()

    rag.set()
    return rag

# Load models and pipeline at app startup
rag_pipeline = initialize_rag()

# Chat History Management
@app.before_request
def initialize_session():
    if "history" not in session:
        session["history"] = []

# Routes
@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        # Get user input
        question = request.form.get("question")
        retrieval_mode = request.form.get("retrieval_mode", "simple")
        reranker_mode = request.form.get("reranker_mode", "simple")

        # Process the question using RAG
        rag_pipeline.set_mode(retrieval=retrieval_mode, reranker=reranker_mode)
        answer = rag_pipeline.query(question)

        # Save to session history
        session["history"].append({"question": question, "answer": answer})
        session.modified = True

    return render_template("index.html", history=session["history"])

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
