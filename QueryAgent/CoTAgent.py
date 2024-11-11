from ContextAgent import *


class CoTAgent(ContextAgent):
    """
    Chain of Thought (CoT) Agent class that inherits from ContextAgent.
    This agent decomposes complex questions into structured sub-questions and retrieves relevant
    contexts for each sub-question.
    """

    def query(self, question:str)->list[str]:
        """
        Takes a user question, decomposes it into three sub-questions, retrieves contexts for each,
        and consolidates the answers.

        Parameters:
        question (str): The main question provided by the user.

        Returns:
        str: A consolidated answer based on the contexts retrieved for each sub-question.
        """

        # Define system and human messages for question decomposition
        messages = [
            (
                "system",
                "You are a structured assistant who decomposes complex questions into specific, three distinct sub-questions. "
                "Your task is to identify each part needed to answer the main question thoroughly. "
                "Provide each sub-question in a comma-separated list, without numbering or extra formatting, "
                "to facilitate retrieval-augmented generation (RAG) processing.\n\n"
                "Input format:\nMain question provided by the user.\n\n"
                "Output format:\nA list of three sub-questions separated by commas, in a single line.\n\n"
                "Example:\nInput: 'What is the history of artificial intelligence and its applications today?'\n"
                "Output: 'What is the origin of artificial intelligence?, How did AI develop over the years?, What are the current applications of AI?'"
            ),
            ("human", question)
        ]

        # Fetch initial context based on the main question
        answer = self.fetch(question=question)

        # Generate sub-questions using the q_model
        subqueries = self.q_model.invoke(messages)

        # Retrieve and accumulate answers for each sub-question
        for subquery in subqueries.content.split(','):
            subquery = str(subquery).strip()  # Clean and format subquery
            answer += self.fetch(question=subquery)

        return answer

    def fetch(self, question:str)->str:
        """
        Fetches relevant documents based on the question and consolidates their text content.

        Parameters:
        question (str): The question or sub-question for which to retrieve documents.

        Returns:
        str: Concatenated text content from retrieved documents.
        """

        # Retrieve documents based on the question
        docs = self.vb(question)
        answer = ""

        # Consolidate the text from each document into a single string
        for doc in docs:
            answer += doc['text']

        return answer
