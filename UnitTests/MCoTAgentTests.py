import sys
from pathlib import Path
from LanceDBSetup import TextDatabase  # Import the class from LanceDBSetup.py
import numpy as np
from sentence_transformers import SentenceTransformer  # Example embedding model
# from QueryAgent.MCoTAgent import *  # Import MCoTAgent class
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

sys.path.append(str(Path(__file__).resolve().parent.parent))
from QueryAgent.MCoTAgent import *  # Import MCoTAgent class
# Add the project folder to the Python path
from AutoWrapper import AutoWrapper
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load a sentence transformer model for embedding text
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Replace with your preferred embedding model
def load_smol_lm():
    return AutoWrapper("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    
# Assuming `mcot_agent` is your implemented agent class/function
def mcot_agent(query_vector, db_instance):
    """
    Retrieves relevant chunks using the query vector.
    """
    agent = MCoTAgent(db_instance, model_pair=(load_smol_lm(), StrOutputParser()))
    # Perform a query on the database
    results = [agent.query(query_vector)[0]]
    return results

if __name__ == "__main__":
    # Database setup
    table_name = "chunked_text_files"
    text_db = TextDatabase(table_name)

    # Load JSON data into LanceDB
    # json_file_path = "UnitTests/data.json"  # Ensure this file exists in the same directory
    # text_db.load_from_json(json_file_path)

    # Input the query string
    query_string = input("Enter your query: ")  # e.g., "Explain neural networks"

    try:
        # Embed the query string into a vector
        query_vector = embedding_model.encode(query_string)
        print(query_vector)
        # Call the mcot agent with the query vector
        response = mcot_agent(query_vector, text_db)
        
        print("\nRetrieved Chunks:")
        for chunk in response:
            print(chunk)
    except Exception as e:
        print(f"Error during retrieval: {e}")
