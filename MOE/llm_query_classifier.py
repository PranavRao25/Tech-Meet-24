import json
from typing import Any, Dict, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
# Take a model, a question as input and then classify it as whether it is a simple, intermediate or complex question.
class QueryClassifier:
    """
    A class to classify queries based on their complexity using an LLM.
    
    Supports three complexity levels:
    - simple: Straightforward, factual queries that can be answered directly
    - intermediate: Queries requiring some reasoning or multiple steps
    - complex: Queries needing in-depth analysis, multi-hop reasoning, or synthesis
    """
    
    def __init__(self, llm_model):
        """
        Initialize the QueryClassifier with an LLM model.
        
        :param llm_model: The language model to use for classification
        """
        self.llm_model = llm_model
    
    def _generate_classification_prompt(self, query: str) -> str:
        """
        Generate a carefully crafted prompt to classify query complexity.
        
        :param query: The input query to be classified
        :return: A detailed prompt for the LLM
        """
        template= """You are an expert query complexity classifier. 
            Your task is to classify the complexity of the following query into one of three levels:
            - simple
            - intermediate 
            - complex

            complexity Classification Guidelines:
            1. simple Query:
            - Direct, factual questions which requires minimal reasoning or context
            - Examples: "What is the capital of France?"

            2. intermediate Query:
            - Requires some reasoning or multi-step thinking and involves moderate level of analysis
            - Examples: "Explain the main causes of the Industrial Revolution"

            3. complex Query:
            - Requires in-depth analysis and synthesis
            - Examples: "Analyze the long-term geopolitical implications of climate change", 

            Query to Classify: "{query}"

            Please respond with a single word indicating the complexity level of the query:
            e.g.
            "complexity": "simple"

            """
        prompt = ChatPromptTemplate.from_template(template)
        
        return
    def classify(self, query: str) -> Dict[str, str]:
        """
        Classify the complexity of a given query.
        
        :param query: The input query to be classified
        :return: A dictionary with complexity and reasoning
        """
        # Validate input
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")
        
        # Generate prompt
        template= """You are an expert query complexity classifier. 
            Your task is to classify the complexity of the following query into one of three levels:
            - simple
            - intermediate 
            - complex

            complexity Classification Guidelines:
            1. simple Query:
            - Direct, factual questions which requires minimal reasoning or context
            - Examples: "What is the capital of France?"

            2. intermediate Query:
            - Requires some reasoning or multi-step thinking and involves moderate level of analysis
            - Examples: "Explain the main causes of the Industrial Revolution"

            3. complex Query:
            - Requires in-depth analysis and synthesis
            - Examples: "Analyze the long-term geopolitical implications of climate change", 

            Query to Classify: "{query}"

            Please respond with a single word indicating the complexity level of the query:
            e.g.
            "complexity": "simple"

            """
        prompt = ChatPromptTemplate.from_template(template)
        small_chain = {"query": RunnablePassthrough()} | prompt | self.llm_model #.invoke(prompt.format(question=question))
        llm_response = small_chain.invoke(query)[len(template):].split("complexity:")[-1].strip()
        
        llm_response = llm_response.strip()
        if "simple" in llm_response:
            return "simple"
        elif "intermediate" in llm_response:
            return "intermediate"
        elif "complex" in llm_response:
            return "complex"
        else:
            print("Error in classification")
            return "simple"