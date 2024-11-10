from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class AlternateQueryAgent:
    def __init__(self, model_pair, no_q=3):
        self.q_model = model_pair[0]
        self.parser = model_pair[1]
        self.turn = no_q
        self.prompt = ChatPromptTemplate.from_template(
            template="""You are given a question {question}.
                  Generate """ + str(no_q) + """alternate questions based on it. They should be numbered and separated by newlines.
                  Do not answer the questions.""".strip()
        )
        self.chain = {"question": RunnablePassthrough()} | self.prompt | self.q_model | self.parser

    def multiple_question_generation(self, question)->list[str]:
        mul_qs = ((self.chain.invoke(question)).split('\n')).append(question)
        return mul_qs