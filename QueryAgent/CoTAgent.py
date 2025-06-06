from QueryAgent.ContextAgent import *
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import logging

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
        template = f"""You are provided with a main question: {question}?
                Your task is to generate three relevant sub-questions based on this main question, avoiding any repetition of the main question itself.

                Guidelines:

                    Ensure each sub-question directly relates to and explores aspects of the main question.
                    Do not include any unrelated information.
                    Do not provide answers—only generate sub-questions.

                Output Format:

                    sub-question: <sub_question> \n <sub_question> \n <sub_question>........
            """
        prompt = ChatPromptTemplate.from_template(template)
        # Fetch initial context based on the main question
        answer = set()
        
        # Generate sub-questions using the q_model
        small_chain = {"question": RunnablePassthrough()} | prompt | self._q_model #.invoke(prompt.format(question=question))
        logging.info(f"COT AGENT SUBQUERIES GENERATING...")
        subqueries = small_chain.invoke(question).strip()#[len(template):]
        logging.info(f"COT AGENT SUBQUERIES GENERATION FINISHED")
        # Retrieve and accumulate answers for each sub-question
        for subquery in [question, *subqueries.split('?')]:
            if subquery == "":
                continue
            subquery = str(subquery).strip()  # Clean and format subquery
            for fetch in self._fetch(question=subquery):
                print("fetching...")
                answer.add(fetch)
        answer = list(answer)
        logging.info(f"CoTsubqueries: {subqueries} \n answer_element_type: {type(answer[0])}")
        return answer

    def _fetch(self, question:str)->list[str]:
        """
        Fetches relevant documents based on the question and consolidates their text content.

        Parameters:
        question (str): The question or sub-question for which to retrieve documents.

        Returns:
        str: Concatenated text content from retrieved documents.
        """

        # Retrieve documents based on the question
        docs = self._vb.query(question)
        answer = set()

        # Consolidate the text from each document into a single string
        for doc in docs:
            answer.add(doc['text'])

        return list(answer)
