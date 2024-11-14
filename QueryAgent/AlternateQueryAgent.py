from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class AlternateQueryAgent:
    """
    Alternate Query Agent class that generates multiple alternate questions based on a given input question.
    """

    def __init__(self, model_pair, no_q=3):
        """
        Initializes the AlternateQueryAgent with a model pair and the number of alternate questions to generate.

        Parameters:
        model_pair (tuple): A tuple containing a question model and a parser model.
        no_q (int): The number of alternate questions to generate. Default is 3.
        """

        self._q_model = model_pair[0]
        self._parser = model_pair[1]
        self._turn = no_q
        self._prompt = ChatPromptTemplate.from_template(
            template="""You are given a question {question}.
                  Generate """ + str(no_q) + """ alternate questions based on it. They should be numbered and separated by newlines.
                  Do not answer the questions.""".strip()
        )
        # Define a chain for generating alternate questions
        self._chain = {"question": RunnablePassthrough()} | self._prompt | self._q_model | self._parser

    def multiple_question_generation(self, question) -> list[str]:
        """
        Generates multiple alternate questions based on the given question.

        Parameters:
        question (str): The original question for which alternate versions are generated.

        Returns:
        list[str]: A list of alternate questions, including the original question as the last element.
        """

        # Generate alternate questions and include the original question in the list
        mul_qs = (self._chain.invoke(question).split('\n')).append(question)
        return mul_qs
