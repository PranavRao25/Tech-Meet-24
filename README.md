# Agentic RAG System using Pathway

## Objective
Create an Agentic RAG (Retrieval-Augmented Generation) system using Pathway that autonomously retrieves, analyzes, and synthesizes information from multiple data sources. The system should dynamically decide the best approach for handling complex queries, utilizing techniques like corrective RAG and multi-agent collaboration to provide accurate responses.

### **System Requirements**
1. Python 3.12 (Ensure Python is installed and added to your system's PATH)
2. Linux OS (with a Bash terminal for command execution)

---

### **Setup Instructions**

#### 1. Create a new virtual environment
```bash
python -m venv my_env
```

#### 2. Install Necessary Dependencies
Run the following command to install all required Python packages:
```bash
pip install -r requirements.txt
```

#### 3. Set Up the Database
Initialize the database by executing the setup script:
```bash
python DataBase/setup.py
```

#### 4. Run the User Interface
1. Open a new terminal.
2. Launch the Streamlit user interface using this command:
   ```bash
   streamlit run User_Interface/Interface.py
   ```

---

### **User Interface and Summary Videos**
To learn more about the UI and see summary demonstrations, refer to the videos in the following folders:
- **UI Demonstration**: `UI_video` folder
- **Summary Demonstration**: `Summary_video` folder

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