from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from ContextAgent import *
from langchain_core.runnables import RunnableLambda


class SubQueryGenAgent:
    def __init__(self, agent_model, mode=True):
        self.q_model = agent_model[0]
        self.parser = agent_model[1]
        self.mode = mode
        if self.mode:
            self.prompt = ChatPromptTemplate.from_template("""You are given a main Question {question}. You must generate a subquestion for the same. Output should in the format: sub-question : <sub_question>""".strip())
            self.chain = {"question": RunnablePassthrough()} | self.prompt | self.q_model | self.parser
        else:
            self.context = ""
            self.prompt = ChatPromptTemplate.from_template(
                """You are given a main Question {question} and a context {context}. You must generate a subquestion for the same. Output should in the format: sub-question : <sub_question>""".strip())
            self.chain = {"question": RunnablePassthrough(), "context": RunnableLambda(lambda x: self.context)} | self.prompt | self.q_model | self.parser

    def sub_questions_gen(self, question):
        return self.chain.invoke(question)


class SubQueryAgent(ContextAgent):
    """
        Generate a set of sub questions for the main question
    """

    turns = 3
    def __init__(self, vb, model_pair, reranker=None, no_q=3):
        super().__init__(vb, model_pair, reranker)
        self.prompt = ChatPromptTemplate.from_template("""""".strip())
        self.sub_q_gen1 = SubQueryGenAgent(model_pair,mode=True) # RunnableLambda(.sub_questions_gen)
        self.sub_q_gen2 = SubQueryGenAgent(model_pair, mode=False) # RunnableLambda(.sub_questions_gen)
        self.turns = no_q

    def query(self, question)->list[str]:
        # initial sub question
        sub_q = self.sub_q_gen1.sub_questions_gen(question)
        initial_context = self.fetch(sub_q)
        total_contexts = [initial_context]

        # progressive sub questions
        context, query = initial_context, sub_q
        for _ in range(self.turns):
            self.sub_q_gen2.context = context
            query = self.sub_q_gen2.sub_questions_gen(query)
            context = self.fetch(query)
            total_contexts.append(context)
        return total_contexts

    def fetch(self, question)->str:
        # can add reranker here
        return self.vb.retrieve(question)