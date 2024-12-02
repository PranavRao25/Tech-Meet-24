from __future__ import annotations
import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import cosine_similarity
from rerankers.models.models import BaseModelClass, BaseLLMClass, colBERT, BGE_M3
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Reranker:
    """
    Simple and intermediate reranker class that reranks documents based on their cosine similarity to the query.
    """

    def __init__(self, model: BaseModelClass, k: Optional[int] = None):
        """
        Initializes the Reranker with a model and an optional value for the number of top documents to return.

        Parameters:
        model (BaseModelClass): The embedding model used for computing document similarity.
        k (Optional[int]): The number of top documents to return after reranking. If None, all documents are returned.
        """
        self.model = model
        self.k = k

    def rerank(self, query: str, docs: list[str]) -> list[str]:
        """
        Reranks documents based on their cosine similarity to the query.

        Parameters:
        query (str): The query text.
        docs (list[str]): List of documents to be reranked.

        Returns:
        list[str]: A list of documents ordered by relevance to the query.
        """
        if len(docs) == 0 or len(query) == 0:
            return []
        # Embed query and documents
        query_embedding = self.model.embed_query(query)
        docs_embeddings = self.model.embed_docs(docs)

        # Compute cosine similarity between the query and each document
        # print(query_embedding.shape, docs_embeddings.shape)
        # print('--------------------------------------------------')
        # print("Q: ", query_embedding)
        # print('--------------------------------------------------')
        # print("D: ", docs_embeddings)
        # print('--------------------------------------------------')
        scores: np.ndarray = cosine_similarity(query_embedding, docs_embeddings)[0]

        # Rank documents by descending similarity score
        if self.k is None:
            top_k = [docs[i] for i in (-scores).argsort()]
        else:
            top_k = [docs[i] for i in (-scores).argsort()[:self.k]]
        logger.info(f"top_k: {top_k}")
        return top_k

    __call__ = rerank  # Enable the reranker to be called directly


class LLMReranker:
    """
    Complex reranker class that uses a language model (LLM) to rerank documents based on relevance to the query.
    """

    def __init__(self, llm: BaseLLMClass, k: Optional[int] = None):
        """
        Initializes the LLMReranker with a language model and an optional value for the number of top documents to return.

        Parameters:
        llm (BaseLLMClass): The language model used for interpreting and ranking document relevance.
        k (Optional[int]): The number of top documents to return after reranking. If None, all documents are returned.
        """
        self.llm = llm
        self.k = k

    def rerank(self, query: str, docs: list[str]) -> None:
        """
        Reranks documents by prompting the language model to assess their relevance to the query.

        Parameters:
        query (str): The query text.
        docs (list[str]): List of document chunks to be reranked.

        Raises:
        NotImplementedError: The rerank function for LLMReranker requires a model-specific implementation.
        """

        # Construct a prompt for the LLM to rank document chunks by relevance
        prompt = f"Given the query:\n'{query}'\n\nRank the following document chunks by relevance:\n"
        for i, doc in enumerate(docs):
            prompt += f"\nChunk {i + 1}:\n{doc}\n"
        prompt += "\n\nReturn the chunks ranked by relevance with a score between 0 (not relevant) to 10 (highly relevant)."

        # Invoke the language model with the constructed prompt
        self.llm(prompt)

        # The rerank logic should be implemented specifically for the LLM
        raise NotImplementedError()
