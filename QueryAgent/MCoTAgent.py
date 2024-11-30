import sys
sys.path.insert(0, '/Users/rachitsandeepjain/Tech-Meet-24/QueryAgent')
from AlternateQueryAgent import *
from SubQueryAgent import *
from langchain_core.runnables import RunnableLambda


class MCoTAgent:
    """
    Implementation of Multiple Chains of Thought (MCoT) Query Agent.
    This agent is used for retrieving intermediate contexts by generating multiple alternate
    and sub-queries for a given question.
    """


    def __init__(self, vb, model_pair: tuple, reranker=None, best=3):
        """
        Initializes the MCoTAgent with a verbosity level, a pair of models, and a reranker.

        Parameters:
        vb: Verbosity level for logging and debugging.
        model_pair (tuple): A tuple of models to be used for alternate and sub-query generation.
        reranker: A reranking model used to rank the contexts based on relevance.
        """

        self._reranker = reranker
        # Initialize alternate query generation using AlternateQueryAgent
        self._alt_q = RunnableLambda(AlternateQueryAgent(model_pair).multiple_question_generation)
        # Initialize sub-query generation using SubQueryAgent
        self._sub_q = RunnableLambda(SubQueryAgent(vb, model_pair).query)
        self._best = best  # Number of top contexts to return after reranking
        self.output_file = "../mcotagentlogs"

    def query(self, question: str) -> list[str]:
        """
        Takes a user query and returns the consolidated context after generating
        alternate and sub-queries.

        Parameters:
        question (str): The user's question or query.

        Returns:
        list[str]: A list of the best contexts consolidated from alternate and sub-queries.
        """

        # Generate alternate questions based on the input question
        alt_qs = self._alt_q.invoke(question)
        alternate_context = []

        # For each alternate question, generate sub-queries and collect their contexts
        for q in alt_qs:
            contexts = self._sub_q.invoke(q)
            alternate_context.append("\n".join(contexts))
        # Log to file or console
        self._log_output("Alternate Contexts:", alternate_context)
        # Clean and rerank the contexts based on relevance
        final_context = self._clean(question, alternate_context)
        return final_context

    def _clean(self, question: str, alternate_context: list[str]) -> list[str]:
        """
        Cleans the retrieved contexts by reranking and returning only the best contexts.

        Parameters:
        question (str): The original user question for reference during reranking.
        alternate_context (list[str]): List of contexts generated from alternate queries.

        Returns:
        list[str]: A list of the top contexts after reranking.
        """

        # Rerank contexts and select the top 'best' number of contexts
        context = self._reranker.rerank(
            query=question,
            documents=alternate_context,
            return_documents=True
        )[:len(alternate_context) - self._best + 1]

        # Return the text of the top contexts
        return [c['text'] for c in context]
    def _default_reranker(self):
        """
        Provides a default reranker that returns all contexts without modification.
        """
        class DefaultReranker:
            def rerank(self, query, documents, return_documents=False):
                return [{"text": doc} for doc in documents]

        return DefaultReranker()
    def _log_output(self, title: str, content: list[str]):
        """
        Logs or writes output to a file.

        Parameters:
        title (str): Title or header for the content.
        content (list[str]): The content to write or log.
        """
        output = f"{title}\n" + "\n".join(content) + "\n\n"
        if self.output_file:
            with open(self.output_file, "a") as f:
                f.write(output)
        else:
            print(output)
