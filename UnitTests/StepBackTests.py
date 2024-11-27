from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder

from QueryAgent.ToTAgent import ToTAgent
from UnitTests.AlternateAgentTests import MistralParser
from UnitTests.LanceDBSetup import TextDatabase

class StepBackTest:
    def __init__(self, data):
        self.db = TextDatabase('step_back')
        self.db.upsert(data)

        self._model = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
        )
