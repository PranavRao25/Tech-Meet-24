import sys
# sys.path.insert(0, '/Users/rachitsandeepjain/Tech-Meet-24/QueryAgent')
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from QueryAgent.ContextAgent import *
import logging
from langchain_core.runnables import RunnableLambda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        self._q_model = agent_model[0]
        self._parser = agent_model[1]
        self._mode = mode
        if self._mode:
            # Define prompt for mode with question only
            self._template = """You are given a main Question {question}. You must generate ONE subquestion for the same. Output should in the format: sub-question : <sub_question>. Do not answer the question"""
            self._prompt = ChatPromptTemplate.from_template(
                self._template
            )
            self._chain = {
                             "question": RunnablePassthrough()
                         } | self._prompt | self._q_model | self._parser
            logger.info("Sub Query with question only")
        else:
            self._context = "" 
            # Define prompt for mode with question and context
            self._template = """You are given a main Question {question} and a context {context}. You must generate a subquestion for the same. Output should in the format: sub-question : <sub_question>. Do not answer the question"""
            self._prompt = ChatPromptTemplate.from_template(
                self._template
            )
            self._chain = {
                             "question": RunnablePassthrough(),
                             "context": RunnableLambda(lambda x: self._context)
                         } | self._prompt | self._q_model | self._parser
            logger.info("Sub Query with question and context")

        logger.info("Sub Query Gen Agent set")

    def sub_questions_gen(self, question:str)->str:
        """
        Generates sub-questions based on the provided question and context if available.

        Parameters:
        question (str): The main question from which to generate sub-questions.

        Returns:
        str: Generated sub-question.
        """
        logger.info("Sub queries being generated")
        return self._chain.invoke(question).strip()


class SubQueryAgent(ContextAgent):
    """
    SubQuery Agent class that generates a set of sub-questions for a main question.
    It iteratively generates sub-questions and retrieves relevant contexts.
    """


    def __init__(self, vb, model_pair, reranker=None, no_q=1):
        """
        Initializes the SubQueryAgent with verbosity, model pair, reranker, and number of queries.

        Parameters:
        vb: Vector Database
        model_pair (tuple): Pair of models to be used for sub-query generation and parsing.
        reranker (optional): Model for reranking contexts if applicable.
        no_q (int): Number of sub-questions to generate in progressive querying.
        """

        super().__init__(vb, model_pair, reranker)
        self._prompt = ChatPromptTemplate.from_template("")
        self._sub_q_gen1 = SubQueryGenAgent(model_pair, mode=True)  # Initial sub-question generation mode
        self._sub_q_gen2 = SubQueryGenAgent(model_pair, mode=False)  # Progressive sub-question generation with context
        self._turns = no_q  # Default number of turns for generating progressive sub-questions
        logger.info("Sub Query Agent set")

    def query(self, question:str) -> list[str]:
        """
        Takes a main question and generates a series of sub-questions to gather related contexts.

        Parameters:
        question (str): Main question provided by the user.

        Returns:
        list[str]: List of contexts retrieved based on generated sub-questions.
        """

        # Generate initial sub-question and retrieve initial context
        sub_q = self._sub_q_gen1.sub_questions_gen(question)
        logger.info(f"sub question:- {sub_q}")
        initial_context = self._fetch(sub_q)
        total_contexts = set(cont["text"] for cont in initial_context)

        # Iteratively generate and fetch contexts for progressive sub-questions
        context, query = initial_context, sub_q
        for _ in range(self._turns):
            self._sub_q_gen2.context = context  # Update context for the next sub-question generation
            query = self._sub_q_gen2.sub_questions_gen(query)
            logger.info(f"sub questions:- {query}")
            context = self._fetch(query)
            for cont in context:
                total_contexts.add(cont["text"])
        return list(total_contexts)

    def _fetch(self, question:str) -> str:
        """
        Fetches and retrieves documents based on the sub-question. Can optionally use reranker.

        Parameters:
        question (str): Sub-question for which to retrieve documents.

        Returns:
        str: Concatenated text content from retrieved documents.
        """
        logger.info("fetch from database")
        return self._vb(question)
