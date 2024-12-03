import json
from typing import Any, Dict, Optional
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
        return f"""You are an expert query complexity classifier. 
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

Please respond ONLY with a JSON object containing these keys:
- "complexity": "simple" or "intermediate" or "complex"
- "reasoning": A brief explanation of why you chose this complexity level

Response format:
{{
    "complexity": "...",
    "reasoning": "..."
}}
"""
    
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
        prompt = self._generate_classification_prompt(query)
        
        # Get LLM response
        try:
            llm_response = self.llm_model.generate(prompt)
            
            # Parse the JSON response
            classification = json.loads(llm_response)
            
            # Validate the classification
            valid_complexities = ["simple", "intermediate", "complex"]
            if classification.get("complexity") not in valid_complexities:
                raise ValueError(f"Invalid complexity level. Must be one of {valid_complexities}")
            print("complexity",classification["complexity"])
            print("reasoning",classification["reasoning"])
            return classification["complexity"].lower()
        
        except json.JSONDecodeError:
            # Fallback to a default classification if JSON parsing fails
            print("complexity","complex")
            print("reasoning","Default classification as JSON parsing failed")
            return "complex"
      