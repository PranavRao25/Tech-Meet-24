from abc import ABC, abstractmethod


class ContextAgent(ABC):
    """
    Base Class for Query Context Agents.
    This abstract class defines the structure and required methods for any context agent that retrieves
    relevant context for user queries.
    """

    def __init__(self, vb, model_pair, reranker=None):
        """
        Initializes the ContextAgent with a verbosity object, a model pair, and an optional reranker.

        Parameters:
        vb: Verbosity or logging object to facilitate debugging and monitoring.
        model_pair (tuple): A tuple containing a language model (LLM) and a parser model.
        reranker (optional): A model for reranking retrieved contexts based on relevance.
        """

        self.vb = vb
        self.q_model = model_pair[0]  # Language model (LLM) for generating or interpreting queries
        self.parser = model_pair[1]  # Parser model for processing or structuring query results
        self.reranker = reranker  # Optional reranker model for prioritizing contexts

    @abstractmethod
    def query(self, question:str) -> list[str]:
        """
        Abstract method to retrieve the total context for the user's main question.

        Parameters:
        question (str): The main question from the user.

        Returns:
        list[str]: A list of strings representing the relevant contexts for the main question.
        """
        raise NotImplementedError('Implement the query function')

    @abstractmethod
    def fetch(self, question:str) -> list[str]:
        """
        Abstract method to retrieve context for a specific sub-query by querying a vector database.

        Parameters:
        question (str): The sub-query for which context needs to be retrieved.

        Returns:
        list[str]: A list of strings representing the relevant contexts for the sub-query.
        """
        raise NotImplementedError('Implement the fetch function')
