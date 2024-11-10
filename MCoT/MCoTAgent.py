from AlternateQueryAgent import *
from SubQueryAgent import *
from langchain_core.runnables import RunnableLambda


class MCoTAgent:
    best = 3

    def __init__(self, vb, model_pair:tuple, reranker):
        super().__init__(vb, model_pair, reranker)
        self.reranker = reranker
        self.alt_q = RunnableLambda(AlternateQueryAgent(model_pair).multiple_question_generation)
        self.sub_q = RunnableLambda(SubQueryAgent(vb, model_pair).query)

    def query(self, question:str)->list[str]:
        alt_qs = self.alt_q.invoke(question)  # alternate questions
        alternate_context = []
        for q in alt_qs: # generate sub queries for q
            contexts = self.sub_q.invoke(q)
            alternate_context.append("\n".join(contexts))  # contexts is a list[str]
        final_context = self.clean(question, alternate_context)
        return final_context

    def clean(self, question:str, alternate_context:list[str])->list[str]:
        """
            Cleaning of contexts
        """

        context = self.reranker.rank(
            query=question,
            documents=alternate_context,
            return_documents=True
        )[:len(alternate_context) - self.best + 1]
        return [c['text'] for c in context]

