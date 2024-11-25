from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from QueryAgent.AlternateQueryAgent import *
from langchain_community.llms import HuggingFaceHub
import os

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
        self._model = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
        )
        self._parser = RunnableLambda(MistralParser().invoke)
        self._alt_q = AlternateQueryAgent((self._model, self._parser))

    def test(self, query):
        self._alt_q = AlternateQueryAgent((self._model, self._parser))
        return self._alt_q.multiple_question_generation(query)


if __name__ == '__main__':
    os.environ['huggingfacehub_api_token'.upper()] = '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'
    aat = AlternateAgentTest()
    try:
        while True:
            question = input()
            qs = aat.test(question)
            print("qs".upper())
            for q in qs:
                print(q)
            print('\n\n')
    except Exception as e:
        print(f"Exception {e}")
    finally:
        print("Testing done".upper())
