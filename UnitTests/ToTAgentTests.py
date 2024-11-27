from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder

from QueryAgent.ToTAgent import ToTAgent
from UnitTests.AlternateAgentTests import MistralParser
from UnitTests.LanceDBSetup import TextDatabase

class ToTAgentTest:
    def __init__(self, data):
        self.db = TextDatabase('tot_test')
        # assume this are chunks
        self.db.upsert(data)
        self._model = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
        )
        self._reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")
        self._parser = RunnableLambda(MistralParser().invoke)
        self._tot_q = ToTAgent(self.db, (self._model, self._parser), self._reranker)

    def test(self, query):
        return self._tot_q.query(query)
