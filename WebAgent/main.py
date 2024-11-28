from .deepsearch import DeepSearch
from .midsearch import MidSearch
from .search import DuckDuckGoSearchRM, GoogleSearch, SerperRM, BraveRM, Retriever
from .models import OllamaClient, GoogleModel
from abc import ABC
from typing import List, Optional


import toml


config = toml.load("../config.toml")
GEMINI_API = config["GEMINI_API"]
SERPER_API = config["SERPER_API"]
GOOGLE_API = config["GOOGLE_API"]
BRAVE_API = config["BRAVE_API"]


# Function to handle hard queries
def hard(query: str, ground_truth_url: Optional[List[str]]) -> str:
    """
    Handles hard queries by breaking them into subtopics and performing multiple searches.

    Args:
        query (str): The query string.
        ground_truth_url (Optional[List[str]]): List of ground truth URLs.

    Returns:
        str: The result of the deep search query.
    """
    # Configuration for GoogleModel
    gemini_kwargs = {
        'api_key': GEMINI_API,
        'temperature': 1.0,
        'top_p': 1
    }

    # Initialize language models
    conv_simulator_lm = GoogleModel(model='models/gemini-1.5-flash', max_tokens=500, **gemini_kwargs)
    student_engine_lm = GoogleModel(model='models/gemini-1.5-flash', max_tokens=500, **gemini_kwargs)
    
    # Configuration for OllamaClient
    ollama_kwargs = {
        "model": 'qwen:0.5b',
        "port": '11434',
        "url": 'http://localhost',
        "stop": ('\n\n---',)
    }

    # Initialize retrievers
    rm2 = SerperRM(serper_search_api_key=SERPER_API, k=3)
    rm0 = GoogleSearch(google_search_api_key=GOOGLE_API, google_cse_id='d0b14c6884a6346d3', k=3, snippet_chunk_size=300)
    rm1 = DuckDuckGoSearchRM(k=3, safe_search='On', region='us-en')
    rm3 = BraveRM(brave_search_api_key=BRAVE_API, k=3, snippet_chunk_size=300)
    retriever = Retriever(available_retrievers=[rm3, rm0, rm1, rm2])

    # Initialize DeepSearch
    deepsearcher = DeepSearch(
        retriever=retriever,
        conv_simulator_lm=conv_simulator_lm,
        student_engine_lm=student_engine_lm,
        max_search_queries_per_turn=3,
        search_top_k=3,
        max_conv_turn=3,
    )
    
    # Perform the query
    return deepsearcher.query(query=query, ground_truth_url=ground_truth_url)


# Function to handle medium queries
def medium(query: str, ground_truth_url: List[str]) -> str:
    """
    Handles medium queries by performing multiple searches.

    Args:
        query (str): The query string.
        ground_truth_url (List[str]): List of ground truth URLs.

    Returns:
        str: The result of the mid search query.
    """
    # Configuration for GoogleModel
    gemini_kwargs = {
        'api_key': GEMINI_API,
        'temperature': 1.0,
        'top_p': 1
    }

    # Initialize language model
    student_engine_lm = GoogleModel(model='models/gemini-1.5-flash', max_tokens=500, **gemini_kwargs)
    
    # Configuration for OllamaClient
    ollama_kwargs = {
        "model": 'qwen:0.5b',
        "port": '11434',
        "url": 'http://localhost',
        "stop": ('\n\n---',)
    }

    # Initialize retrievers
    rm2 = SerperRM(serper_search_api_key=SERPER_API, k=3)
    rm0 = GoogleSearch(google_search_api_key=GOOGLE_API, google_cse_id='d0b14c6884a6346d3', k=3)
    rm1 = DuckDuckGoSearchRM(k=3, safe_search='On', region='us-en')
    rm3 = BraveRM(brave_search_api_key=BRAVE_API, k=3)
    retriever = Retriever(available_retrievers=[rm3, rm0, rm1, rm2])
    
    # Initialize MidSearch
    midesearch = MidSearch(
        retriever=retriever,
        student_engine_lm=student_engine_lm,
        search_top_k=3,
    )
    
    # Perform the query
    return midesearch.query(query=query, ground_truth_url=ground_truth_url)


# Function to handle easy queries
def easy(query: str, exclude_urls: Optional[List[str]]) -> str:
    """
    Handles easy queries by performing a single search.

    Args:
        query (str): The query string.
        exclude_urls (Optional[List[str]]): List of URLs to exclude from the search.

    Returns:
        str: The result of the search.
    """
    # Initialize retrievers
    rm2 = SerperRM(serper_search_api_key=SERPER_API, k=3)    
    rm0 = GoogleSearch(google_search_api_key=GOOGLE_API, google_cse_id='d0b14c6884a6346d3', k=3)
    rm1 = DuckDuckGoSearchRM(k=3, safe_search='On', region='us-en')
    rm3 = BraveRM(brave_search_api_key=BRAVE_API, k=3)
    retriever = Retriever(available_retrievers=[rm2, rm0, rm1, rm3])

    # Perform the search
    sources = retriever.forward(queries=[query], exclude_urls=exclude_urls)
    info = ""
    for source in sources:
        info += source['snippets'][0]
        
    return info


class WebAgent(ABC):
    """
    WebAgent class for generating conversational responses using various language models and search mechanisms.
    Attributes:
        conv_simulator_lm (OllamaClient): Language model client for conversation simulation.
        student_engine_lm (OllamaClient): Language model client for student engine.
        triage (OllamaClient): Language model client for triage.
        retriever (Retriever): Retriever client using DuckDuckGo search.
        deepsearcher (DeepSearch): Deep search client combining retriever and language models.
    Methods:
        query(query: str) -> str:
            Determines the complexity of the query and retrieves information using either a single search or multiple searches.
    """

    def __init__(self):
        # Configuration for GoogleModel
        gemini_kwargs = {
            'api_key': GEMINI_API,
            'temperature': 1.0,
            'top_p': 1
        }

        # Initialize triage model
        self.triage = GoogleModel(model='models/gemini-1.5-flash', max_tokens=500, **gemini_kwargs)
        
    def query(self, query: str) -> str:
        """
        Determines the complexity of the query and retrieves information using either a single search or multiple searches.

        Args:
            query (str): The query string.

        Returns:
            str: The result of the query based on its difficulty.
        """
        # Prompt to determine query difficulty
        prompt = f"You are given a query and you can only use Google search to answer it. \
            The query could be \
            -Hard: requires breaking the query into sub topics and search \
            -Medium: requires multiple searches \
            -Easy: A single search(top 5 results) is sufficient. \
            Answer if the query is 'Hard', 'Medium' or 'Easy' in one word. \
            The Query: {query}"
        
        # Determine the difficulty of the query
        difficulty = self.triage(prompt)[0]
        
        # print(difficulty)
        
        # Handle the query based on its difficulty
        if 'Hard' in difficulty:
            return hard(query, ground_truth_url=[''])
        elif 'Medium' in difficulty:
            return medium(query, ground_truth_url=[''])
        elif 'Easy' in difficulty:
            return easy(query, exclude_urls=[''])
        else:
            return "Unable to determine query difficulty."


if __name__ == '__main__':

    query = str(input())
    agent = WebAgent()
    print(agent.query(query=query))
