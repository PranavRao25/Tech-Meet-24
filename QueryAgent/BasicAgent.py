from QueryAgent.ContextAgent import ContextAgent
import logging

class BasicAgent(ContextAgent):
    """
    BasicAgent class that inherits from ContextAgent.
    This agent decomposes complex questions into structured sub-questions and retrieves relevant
    contexts for each sub-question.
    """

    def query(self, question: str) -> list[str]:
        """
        Takes a user question, retrieves contexts for it, and consolidates the answers.

        Parameters:
        question (str): The main question provided by the user.

        Returns:
        list[str]: A list of consolidated answers based on the contexts retrieved.
        """

        logging.info("Basic Agent Started...")
        return self._fetch(question)

    def _fetch(self, question: str) -> list[str]:
        """
        Fetches relevant documents based on the question and consolidates their text content.

        Parameters:
        question (str): The question for which to retrieve documents.

        Returns:
        list[str]: A list of unique text contents from retrieved documents.
        """

        # Retrieve documents based on the question
        docs = self._vb.query(question)
        answer = set()

        # Consolidate the text from each document into a set to ensure uniqueness
        for doc in docs:
            answer.add(doc['text'])

        return list(answer)