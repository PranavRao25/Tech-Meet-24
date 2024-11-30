from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# from ..QueryAgent.AlternateQueryAgent import *
# from ..QueryAgent.AlternateQueryAgent import *
from langchain_community.llms import HuggingFaceHub
import os

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
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
        # mul_qs = ().append(question)
        return mul_qs
class MistralParser:
  """
    Wrapper Class for StrOutputParser Class
    Custom made for Mistral models
  """

  def __init__(self, stopword='Answer:'):
    """
      Initializes a StrOutputParser as the base parser
    """
    self.parser = StrOutputParser()
    self.stopword = stopword

  def invoke(self, query):
    """
      Invokes the parser and finds the Model response
    """
    ans = self.parser.invoke(query)
    return ans[ans.find(self.stopword)+len(self.stopword):].strip()


class AlternateAgentTest:
    def __init__(self):
        # self._model = HuggingFaceHub(
        #     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        #     model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
        # )
        # self._model = pipeline("text2text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
        print(self._model("What are the benefits of eating fruits?"))
        self._parser = RunnableLambda(MistralParser().invoke)
        self._alt_q = AlternateQueryAgent((self._model, self._parser))

    def test(self, query):
        self._alt_q = AlternateQueryAgent((self._model, self._parser))
        return self._alt_q.multiple_question_generation(query)


if __name__ == '__main__':
    os.environ['HF_TOKEN'] = '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'
    aat = AlternateAgentTest()
    try:
        while True:
            question = input()
            qs = aat.test(question)
            print("qs".upper())
            for q in qs:
                print(q, end='')
            # print('\n\n')
    except Exception as e:
        print(f"Exception {e}")
    finally:
        print("Testing done".upper())
