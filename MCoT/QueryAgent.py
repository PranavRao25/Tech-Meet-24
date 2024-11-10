from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import re
from langchain_core.runnables import RunnableLambda
import ContextAgent


class AlternateQuestionAgent(ContextAgent):
    """
      Prepares some alternate questions for given question and returns the cumulative context
    """

    best = 2

    def __init__(self, vb, agent_pair, reranker, no_q=2):
        super().__init__(vb, agent_pair, reranker)
        self.prompt = ChatPromptTemplate.from_template(
            template="""You are given a question {question}.
          Generate """ + str(no_q) + """alternate questions based on it. They should be numbered and separated by newlines.
          Do not answer the questions. Header of the output should be 'alternate-questions :'
          """.strip()
        )
        self.chain = {"question": RunnablePassthrough()} | self.prompt | self.q_model | self.parser

    def mul_qs(self, question):
        """
          Prepares multiple questions for the given question
        """

        qs = [i for i in (self.chain.invoke(question)).split('\n')] + [question]
        if '' in qs:
            qs.remove('')
        uni_q = []
        for q in qs:
            if q not in uni_q:
                uni_q.append(q)
        return uni_q  # assuming the questions are labelled as 1. q1 \n 2. q2

    def query(self, question):
        """
          Returns the cumulative context for the given question
        """

        questions = self.mul_qs(question)
        for q in questions:
            print(q)
        return self.fetch(questions)

    def retrieve(self, question):
        """
          Returns the context for the given question
        """

        prior_context = self.vb.query(question)
        cont = ["".join(i) for i in prior_context]

        c = self.cross_model.rank(
            query=question,
            documents=cont,
            return_documents=True
        )[:len(prior_context) - self.best + 1]
        return [i['text'] for i in c]  # list of text

    def fetch(self, questions):
        """
          Fetches contexts from the Vector Databases
        """

        contexts = [self.retrieve(q) for q in questions]
        uni_contexts = []
        for i in contexts:
            for j in i:
                if j not in uni_contexts:
                    uni_contexts.append(j)
        u = []
        for i in uni_contexts:
            k = re.split("(\.|\?|!)\n", i)
            for j in k:
                if j in '.?!':
                    continue
                if j not in u:
                    u.append(j)
        uni_contexts = []
        for i in range(len(u)):
            for j in range(len(u)):
                if j != i and u[i] in u[j]:
                    break
            else:
                uni_contexts.append(u[i])
        contexts = "@@".join(uni_contexts)
        return contexts


class SubQueryAgent(ContextAgent):
    """
        Prepares sub-questions based on question and context provided and returns cumulative contexts
    """

    best = 2

    class _QueryGen:
        """
        Prepares question based on provided question and context
        Subclass used by SubQueryAgent
        """

        def __init__(self, q_model, parser=RunnableLambda(lambda x: x),prompt="""
                    You will be given a pair of question and its context as an input.You must form a question contextually related to both of them.
                    Question : {Question}\nContext: {Context}
                    Output should in the format: sub-question : <sub_question>
                    """):
            self.context = ""
            self.prompt = ChatPromptTemplate.from_template(prompt.strip())
            self.chain = {"Question": RunnablePassthrough(),
                          "Context": RunnableLambda(lambda c: self.context)} | self.prompt | q_model | parser

        def __call__(self, question, context=""):
            self.context = context
            return self.chain.invoke(question)

    def __init__(self, vb, agent_model, reranker, no_q=3):
        super().__init__(vb, agent_model, reranker)
        self.turns = no_q

    def fetch(self, question):
        prior_context = self.vb.query(question)
        cont = ["".join(i) for i in prior_context]

        c = self.reranker.rank(
            query=question,
            documents=cont,
            return_documents=True
        )[:len(prior_context) - self.best + 1]
        return [i['text'] for i in c]  # list of text

    def query(self, question):
        question = question
        all_sub_qs = []
        agent = self._QueryGen(self.q_model, self.parser)
        sub_q = agent(question)
        print(f"First Sub question: {sub_q}\n")
        all_sub_qs.append(sub_q)
        contexts = []
        prompt = f"""
        You are given a main Question {question} and a pair of its subquestion and related sub context.
    You must generate a question based on the main question, and all of the sub-question and sub-contexts pairs.
    Output should in the format: sub-question : <sub_question>        
        """
        for i in range(self.turns - 1):
            print(f"ITERATION NO: {i + 1}")
            context = self.fetch(sub_q)
            contexts += context
            total_context = "\n".join(contexts)
            agent = self._QueryGen(self.q_model, self.parser,prompt=prompt + "\nsub-question : {Question}\nsub-context: {Context}")
            prompt += f"\nsub-question : {sub_q}\nsub-context: {total_context}"
            sub_q = agent(sub_q, total_context)
            print(f"{i + 2}th Sub question: {sub_q}\n")
        uni = []
        for c in contexts:
            if c not in uni:
                uni.append(c)
        return uni


class MCoTAgent(ContextAgent):
    """
      Forms multiple questions for a given question
      Prepares some serial subquestion for each of the alternate question
      Fetches contexts for each subquestion and returns the sum of them
    """

    def __init__(self, vb_list, model, cross_model, parser=(RunnableLambda(lambda x: x), RunnableLambda(lambda x: x))):
        super().__init__(vb_list, model, cross_model, parser)
        self.alt_agent = RunnableLambda(AlternateQuestionAgent(vb_list, model, cross_model, parser[0]).mul_qs)
        self.sub_agent = RunnableLambda(SubQueryAgent(vb_list, model, cross_model, parser[1]).query)

    def query(self, question):
        """
          Returns the cumulative context for the given question
        """

        contexts = []
        for q in self.alt_agent.invoke(question):  # multiple alternate questions
            print(f"Question: {q}")
            contexts.append(self.sub_agent.invoke(q))  # context retrieved for each multiple question
        return self.fetch(contexts)

    def fetch(self, contexts):
        """
          Returns the context after cleaning
        """

        uni_contexts = []
        for i in contexts:
            if i not in uni_contexts:
                uni_contexts.append(i)
        u = []
        for i in uni_contexts:
            k = re.split("(\.|\?|!)\n", i)
            for j in k:
                if j in '.?!':
                    continue
                if j not in u:
                    u.append(j)
        uni_contexts = []
        for i in range(len(u)):
            for j in range(len(u)):
                if j != i and u[i] in u[j]:
                    break
            else:
                uni_contexts.append(u[i])
        # uni_contexts = u
        return "@@".join(uni_contexts)
