from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import logging
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        template="""You are given a question {question}.
                  Generate """ + str(no_q) + """ alternate questions based on it. They should be numbered and separated by newlines.
                  Do not answer the questions.""".strip()
        self._prompt = ChatPromptTemplate.from_template(template)
        # Define a chain for generating alternate questions
        
        self._chain = {"question": RunnablePassthrough()} | self._prompt | self._q_model | self._parser

        logger.info("Alternate Query Agent set")

    def multiple_question_generation(self, question: str) -> list[str]:
        """
        Generates multiple alternate questions based on the given question.

        Parameters:
        question (str): The original question for which alternate versions are generated.

        Returns:
        list[str]: A list of alternate questions, including the original question as the last element.
        """

        # Generate alternate questions and include the original question in the list
        mul_qs = self._chain.invoke(question)#.split('\n')

        logger.info("multiple questions generated")
        return mul_qs

# if __name__ == '__main__':
#
#
#     model = HuggingFaceHub(
#         repo_id="mistralai/Mistral-7B-Instruct-v0.3",
#         model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
#     )
#     parser = RunnableLambda(MistralParser().invoke)
#     alt_q = AlternateQueryAgent((self._model, self._parser))
#     alt_q_agent = AlternateQueryAgent(model_pair)
#
#     # Define the input question for alternate question generation
#     question = "What are the benefits of eating fruits?"
#
#     # Generate multiple alternate questions based on the input question
#     alternate_questions = alt_q_agent.multiple_question_generation(question)
#     print(alternate_questions)