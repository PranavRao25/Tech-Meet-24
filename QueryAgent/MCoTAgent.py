import sys
# sys.path.insert(0, '/Users/rachitsandeepjain/Tech-Meet-24/QueryAgent')
from QueryAgent.AlternateQueryAgent import *
from QueryAgent.SubQueryAgent import *
import logging
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCoTAgent:
    """
    Implementation of Multiple Chains of Thought (MCoT) Query Agent.
    This agent is used for retrieving intermediate contexts by generating multiple alternate
    and sub-queries for a given question.
    """

    def __init__(self, vb, model_pair: tuple, best=3):
        """
        Initializes the MCoTAgent with a verbosity level, a pair of models, and a reranker.

        Parameters:
        vb: Verbosity level for logging and debugging.
        model_pair (tuple): A tuple of models to be used for alternate and sub-query generation.
        reranker: A reranking model used to rank the contexts based on relevance.
        """

        # Initialize alternate query generation using AlternateQueryAgent
        self._alt_q = RunnableLambda(AlternateQueryAgent(model_pair).multiple_question_generation)
        # Initialize sub-query generation using SubQueryAgent
        self._sub_q = RunnableLambda(SubQueryAgent(vb, model_pair).query)
        self._best = best  # Number of top contexts to return after reranking
        self.output_file = "../mcotagentlogs"
        logger.info("MCoT Agent has been set up")

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
        logger.info(f"Input question: {question}")
        alt_qs = self._alt_q.invoke(question)
        alternate_context = []

        # For each alternate question, generate sub-queries and collect their contexts
        for q in alt_qs:
            logger.info(f"alt q: {q}")
            contexts = self._sub_q.invoke(q)
            alternate_context.append("\n".join(contexts))
        # Log to file or console
        self._log_output("Alternate Contexts:", alternate_context)
        # Clean and rerank the contexts based on relevance
        final_context = self._clean(alternate_context)
        return final_context

    def _clean(self, alternate_context: list[str]) -> list[str]:
        """
        Cleans the retrieved contexts by picking only the unique contexts

        Parameters:
        alternate_context (list[str]): List of contexts generated from alternate queries.

        Returns:
        list[str]: A list of the top contexts after reranking.
        """

        unique_context = []
        for context in alternate_context:
            if context not in unique_context:
                unique_context.append(context)

        # Return the text of the top contexts
        return unique_context

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
