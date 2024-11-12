# Agentic RAG System using Pathway

## Objective
Create an Agentic RAG (Retrieval-Augmented Generation) system using Pathway that autonomously retrieves, analyzes, and synthesizes information from multiple data sources. The system should dynamically decide the best approach for handling complex queries, utilizing techniques like corrective RAG and multi-agent collaboration to provide accurate responses.

---

## Setup the Environment
Please follow the README.md file present in the code link

---

## Demonstration
Please check the video present in the video folder

---

## Pipeline
The system pipeline is designed to handle diverse query complexities through a modular approach:

1. **Query Classification**:  
   The input query is passed through a Mixture of Experts (MoE) which categorizes it as simple, intermediate, or complex.

2. **Adaptive Retrieval and Re-Ranking**:  
   Depending on the query classification, the system uses one of three retrieval paths (simple, intermediate, or complex) to retrieve context differently. The retrieval paths rely on reasoning agents such as the Tree of Thought (ToT) or the Chain of Thought (CoT) and re-rankers.

3. **Thresholding Mechanism**:  
   The retrieved content is checked by a thresholding mechanism to assess its sufficiency and relevance before proceeding.

4. **LLM Processing and Output Generation**:  
   The retrieved content is fed to a Large Language Model (LLM) to generate an output based on the provided information. Web search is used only if the connection to the LLM is lost.

---

## Project Status
The individual modules are mostly ready, with integration and extensive testing remaining.

- **Chain of Thought (CoT)**, **Multi-step Chain of Thought (MCoT)**, and **Step-Back Prompting** are complete.
- **Tree of Thought (ToT)** is pending.
- **Rerankers** are mostly implemented, though not fully finalized:
  - The **LLM Reranker** still requires testing.
- **Thresholder** requires additional work to optimize performance.
- **Web Agent**, **LLM**, and **Vector Database** are functional at a basic level.

---

## System Requirements
1. Python 3.12
2. Linux (with Bash terminal)

### Library Requirements
- pathway[all]
- langchain
- langchain_google_genai
- pydantic
- spacy
- transformers
- torch
- langchain_core
- scikit-learn
- numpy
- langchain_community
- ragas
- langgraph
- backoff
- streamlit
- regex
- dspy_ai==2.4.9
- httpx
- langchain-text-splitters
- trafilatura
- google-api-python-client
- duckduckgo_search
- google-generativeai
- datasets