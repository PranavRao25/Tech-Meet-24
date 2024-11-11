from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from rerankers.models.models import BaseModelClass, BaseLLMClass, colBERT, BGE_M3

class Reranker: # simple and intermediate reranker
    def __init__(self, model:BaseModelClass, k:Optional[int] = None):
        self.model = model
        self.k = k
    def rerank(self, query:str, docs:list[str]):
        # rerank function
        query_embedding = self.model.embed_query(query)
        docs_embeddings = self.model.embed_docs(docs)
        scores:np.ndarray = cosine_similarity(query_embedding, docs_embeddings)[0]
        if self.k is None:
            top_k = [docs[i] for i in (-scores).argsort()]
        else:
            top_k = [docs[i] for i in (-scores).argsort()[:self.k]]
        return top_k
    __call__ = rerank
    
class LLMReranker: # complex reranker
    def __init__(self, llm:BaseLLMClass, k:Optional[int]=None):
        self.llm = llm
        self.k = k
    def rerank(self, query:str, docs:list[str]):
        # Create prompt for reranking
        prompt = f"Given the query:\n'{query}'\n\nRank the following document chunks by relevance:\n"
        for i, doc in enumerate(docs):
            prompt += f"\nChunk {i + 1}:\n{doc}\n"
        prompt += "\n\nReturn the chunks ranked by relevance with a score between 0 (not relevant) to 10 (highly relevant)."
        self.llm.invoke(prompt)
        raise NotImplementedError()
