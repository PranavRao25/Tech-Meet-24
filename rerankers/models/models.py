import numpy as np
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class BaseModelClass:
    def __init__(self):
        """Base model class to be used in reranker"""
    def embed_query(self, query:str)->np.ndarray:
        """
        Takes a query and returns a ndarray of embedding.
        If the embedding dimension is n. The shape of ndarray should be: (1, n)
        """
    def embed_docs(self, docs:list[str])->np.ndarray:
        """
        Takes a list of documents/passages and returns an ndarray of embedding
        If the embedding dimension is n and number of documents are m. The shape of ndarray should be: (m, n)
        """

class BaseLLMClass:
    def __init__(self):
        "Base model class for LLM"
    def invoke(self, prompt:str)->str:
        """Pass prompt to llm and get output"""


class colBERT(BaseModelClass):
    def __init__(self, device:str='cpu'):
        self.model = HuggingFaceEmbeddings(model_name="colbert-ir/colbertv2.0", model_kwargs={"device":device})
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([self.model.embed_query(query)])
    def embed_docs(self, docs: list[str]) -> np.ndarray:
        return np.array(self.model.embed_documents(docs))
    
class BGE_M3(BaseModelClass):
    def __init__(self, device:str='cpu'):
        self.model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device":device})
    def embed_query(self, query: str) -> np.ndarray:
        return np.array([self.model.embed_query(query)])
    def embed_docs(self, docs: list[str]) -> np.ndarray:
        return np.array(self.model.embed_documents(docs))
