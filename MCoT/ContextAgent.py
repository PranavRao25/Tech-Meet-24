from abc import ABC, abstractmethod


class ContextAgent(ABC):
    """
    Base Class for Query Context Agents
    """

    def __init__(self, vb, model_pair, reranker=None):
        self.vb = vb
        self.q_model = model_pair[0]  # llm
        self.parser = model_pair[1]  # parser
        self.reranker = reranker

    @abstractmethod
    def query(self, question)->list[str]:
        """
            Retrieves total context for the user query
        """
        raise NotImplementedError('Implement Query function')

    @abstractmethod
    def fetch(self, question)->list[str]:
        """
            Retrieves context for a subquery by making a call to the Vector Database
        """
        raise NotImplementedError('Implement Fetch function')
