from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from .ContextAgent import *
from langchain_core.runnables import RunnableLambda


class SubQueryGenAgent:
    """
    SubQuery Generation Agent class responsible for generating sub-questions from a main question.
    This agent can operate in two modes:
    1. Mode with question only.
    2. Mode with both question and context for more targeted sub-question generation.
    """

    def __init__(self, agent_model, mode=True):
        """
        Initializes the SubQueryGenAgent with a model and operational mode.

        Parameters:
        agent_model (tuple): A tuple with the question model and parser model.
        mode (bool): Determines if the agent uses only the question (True) or both question and context (False).
        """

        self.q_model = agent_model[0]
        self.parser = agent_model[1]
        self.mode = mode

        if self.mode:
            # Define prompt for mode with question only
            self.prompt = ChatPromptTemplate.from_template(
                """You are given a main Question {question}. You must generate a subquestion for the same. Output should in the format: sub-question : <sub_question>"""
            )
            self.chain = {
                             "question": RunnablePassthrough()
                         } | self.prompt | self.q_model | self.parser

        else:
            # Define prompt for mode with question and context
            self.context = ""
            self.prompt = ChatPromptTemplate.from_template(
                """You are given a main Question {question} and a context {context}. You must generate a subquestion for the same. Output should in the format: sub-question : <sub_question>"""
            )
            self.chain = {
                             "question": RunnablePassthrough(),
                             "context": RunnableLambda(lambda x: self.context)
                         } | self.prompt | self.q_model | self.parser

    def sub_questions_gen(self, question:str)->str:
        """
        Generates sub-questions based on the provided question and context if available.

        Parameters:
        question (str): The main question from which to generate sub-questions.

        Returns:
        str: Generated sub-question.
        """
        return self.chain.invoke(question)


class SubQueryAgent(ContextAgent):
    """
    SubQuery Agent class that generates a set of sub-questions for a main question.
    It iteratively generates sub-questions and retrieves relevant contexts.
    """

    turns = 3  # Default number of turns for generating progressive sub-questions

    def __init__(self, vb, model_pair, reranker=None, no_q=3):
        """
        Initializes the SubQueryAgent with verbosity, model pair, reranker, and number of queries.

        Parameters:
        vb: Verbosity level or logging parameter.
        model_pair (tuple): Pair of models to be used for sub-query generation and parsing.
        reranker (optional): Model for reranking contexts if applicable.
        no_q (int): Number of sub-questions to generate in progressive querying.
        """

        super().__init__(vb, model_pair, reranker)
        self.prompt = ChatPromptTemplate.from_template("")
        self.sub_q_gen1 = SubQueryGenAgent(model_pair, mode=True)  # Initial sub-question generation mode
        self.sub_q_gen2 = SubQueryGenAgent(model_pair, mode=False)  # Progressive sub-question generation with context
        self.turns = no_q

    def query(self, question:str) -> list[str]:
        """
        Takes a main question and generates a series of sub-questions to gather related contexts.

        Parameters:
        question (str): Main question provided by the user.

        Returns:
        list[str]: List of contexts retrieved based on generated sub-questions.
        """

        # Generate initial sub-question and retrieve initial context
        sub_q = self.sub_q_gen1.sub_questions_gen(question)
        initial_context = self.fetch(sub_q)
        total_contexts = [initial_context]

        # Iteratively generate and fetch contexts for progressive sub-questions
        context, query = initial_context, sub_q
        for _ in range(self.turns):
            self.sub_q_gen2.context = context  # Update context for the next sub-question generation
            query = self.sub_q_gen2.sub_questions_gen(query)
            context = self.fetch(query)
            total_contexts.append(context)

        return total_contexts

    def fetch(self, question:str) -> str:
        """
        Fetches and retrieves documents based on the sub-question. Can optionally use reranker.

        Parameters:
        question (str): Sub-question for which to retrieve documents.

        Returns:
        str: Concatenated text content from retrieved documents.
        """

        return self.vb.retrieve(question)
