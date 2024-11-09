from abc import ABC, abstractmethod


class ContextAgent(ABC):
    """
    Base Class for Query Context Agents
    """

    def __init__(self, vb, model_pair, reranker=None):
        self.vb_list = vb
        self.q_model = model_pair[0]
        self.parser = model_pair[1]
        self.cross_model = reranker

    @abstractmethod
    def query(self, question)->list[str]:
        raise NotImplementedError('Implement Query function')

    @abstractmethod
    def fetch(self, question)->list[str]:
        raise NotImplementedError('Implement Fetch function')
