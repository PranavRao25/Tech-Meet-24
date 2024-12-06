# **Agentic RAG System using Pathway**

## **Objective**  
Create an Agentic RAG (Retrieval-Augmented Generation) system using Pathway that autonomously retrieves, analyzes, and synthesizes information from multiple data sources. The system dynamically decides the best approach for handling complex queries, utilizing techniques like corrective RAG and multi-agent collaboration to provide accurate responses.

---

## **Features**  
- Intelligent query classification (simple, intermediate, or complex).  
- Adaptive retrieval and re-ranking based on query complexity.  
- Thresholding mechanism to assess sufficiency and relevance of retrieved content.  
- Integrated web search for enhanced query resolution.  
- Dynamic Pathway Vector Store for efficient document indexing and embedding.  
- Guardrails for query filtering to handle invalid or inappropriate inputs.  
- Multi-agent reasoning with Chain of Thought (CoT) and Multi Chain of Thought (MCoT).

---

## **System Requirements**  
1. Python 3.12 (Ensure Python is installed and added to your system's PATH).  
2. Linux OS (with a Bash terminal for command execution).  
4. x86 architecture (The system works only on x86 architecture).

---

## **Setup Instructions**  

### 1. Create a Virtual Environment  
```bash
python -m venv my_env
```

### 2. Activate the environment
```bash
source my_env/bin/activate
```

### 2. Install Dependencies  
Run the following command:  
```bash
pip install -r requirements.txt
```

### 3. Set Up the Database  
Initialize the database by running:  
```bash
python DataBase/setup.py
```

### 4. Launch the User Interface  
1. Open a new terminal.  
2. Run the Streamlit UI using:
   ```bash
   cd User_Interface
   streamlit run Interface.py
   ```

---

### **UI Submission**
- **UI Video**: Check UIVideo in the Video folder.

### **Video Submission**  
- **Summary Video**: Check SummaryVideo in the Video folder. 

---

## Pipeline
The system pipeline is designed to handle diverse query complexities through a modular approach:

1. **Query Classification**:  
   The input query is passed through an-Itelligent Query Classifier which categorizes it as simple, intermediate, or complex.

2. **Adaptive Retrieval and Re-Ranking**:  
   Depending on the query classification, the system uses one of three retrieval paths (simple, intermediate, or complex) to retrieve context differently. The retrieval paths rely on reasoning agents such as the Chain of Thought (CoT) or the Multi Chain of Thought (MCoT) and re-rankers.

3. **Thresholding Mechanism**:
   The retrieved content is checked by a thresholding mechanism to assess its sufficiency and relevance before proceeding.

4. **Web Search**:
   If the thresholding mechanism deems the retrieved content insufficient, the system performs a web search to gather additional information. This ensures that the response is comprehensive and accurate.

5. **LLM Processing and Output Generation**:  
   The retrieved content is fed to a Large Language Model (LLM) to generate an output based on the provided information. Web search is used only if the connection to the LLM is lost.

6. **Guardrails**
   Guardrails are integrated at the query stage to filter out gibberish, empty, or inappropriate
   inputs using rigorously tested pre-trained models.

7. **Dynamic Pathway Vector Store**
   Our solution uses Pathway Vectorstore that enables building a document index on top of out documents without the complexity of ETL pipelines, managing different containers for storing, embedding, and serving.