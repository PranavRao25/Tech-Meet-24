from datasets import Dataset
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from ragas import evaluate
from typing_extensions import TypedDict
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from langgraph.graph import END, StateGraph
from QueryAgent.MCoTAgent import MCoTAgent
from QueryAgent.CoTAgent import CoTAgent
from rerankers.rerankers.reranker import *

class Pipeline:
    def __init__(self, retrieval_agent, reranker):
        self.simple_retrieval_agent = retrieval_agent
        self.simple_reranker = reranker

    def retrieve(self, question):
        context = self.simple_retrieval_agent.query(question)
        new_context = self.simple_reranker.rerank(question, context)
        return new_context


class RAG:
    """
        Standard Operating Procedure:
        1. __init__
        2. retrieval_agent_prep (simple, intermediate, complex)
        3. reranker_prep (simple, intermediate, complex)
        4. moe_prep
        7. thresholder_prep
        8. web_search_prep
        9. step_back_prompt_prep
        10. set
    """

    def __init__(self, vb, llm):
        self.vb = vb
        self.llm = llm

    def retrieval_agent_prep(self, q_model, parser, reranker, mode):
        if mode == "simple":
            self.cot_agent = CoTAgent(self.vb, (q_model, parser), reranker)
        elif mode == "intermediate":
            self.mcot_agent = MCoTAgent(self.vb, (q_model, parser), reranker)
        elif mode == "complex":
            self.tot_agent = None
        else:
            raise ValueError("Incorrect mode")

    def ground_truth_prep(self):
        self.ground_truth = None

    def reranker_prep(self, reranker, mode):
        if mode == "simple":
            self.simple_reranker = Reranker(reranker)
        elif mode == "intermediate":
            self.intermediate_reranker = Reranker(reranker)
        elif mode == "complex":
            self.complex_reranker = LLMReranker(reranker)
        else:
            raise ValueError("Incorrect mode")

    def moe_prep(self, model):
        self.moe = model

    def thresholder_prep(self, model):
        self.thresholder = model

    def web_search_prep(self, model):
        self.web_search_agent = model

    def step_back_prompt_prep(self, model):
        self.step_back_agent = model

    def pipeline_setup(self):
        self.simple_pipeline = RunnableLambda(Pipeline(self.cot_agent, self.simple_reranker).retrieve)
        self.intermediate_pipeline = RunnableLambda(Pipeline(self.mcot_agent, self.intermediate_reranker).retrieve)
        self.complex_pipeline = RunnableLambda(Pipeline(self.tot_agent, self.complex_reranker).retrieve)

    def _context_prep(self):
        self.context = ""

    def _rag_graph(self):
        class GraphState(TypedDict):
            """
            Represents the state of our graph.

            Attributes:
                question: question
                context: context
                answer: answer
            """
            question: str
            context: str
            answer: str


        def simple_pipeline(state):
            context = self.simple_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}

        def fetch(state):
            self._context_prep()
            return {"question": state["question"], "context": self.context, "answer": state["answer"]}

        def answer(state):
            return {"question": state["question"], "context": self.context, "answer": answer}

        # def classify(state):
        #     # call the MoE
        #     decision = None
        #     if decision in ["simple", "intermediate", "complex"]:
        #         return decision
        #     else:
        #         raise Exception("MoE Error")

        # def intermediate_pipeline(state):
        #     context = self.intermediate_pipeline.invoke(state["question"])
        #     return {"question": state["question"], "context": context, "answer": state["answer"]}

        # def complex_pipeline(state):
        #     context = self.complex_pipeline.invoke(state["question"])
        #     return {"question": state["question"], "context": context, "answer": state["answer"]}

        # def web_search(state):

        # def threshold(state):

        # def redirect(state):

        # self.RAGraph = StateGraph(GraphState)
        # self.RAGraph.set_entry_point("entry")
        # self.RAGraph.add_node("entry", RunnablePassthrough)
        # self.RAGraph.add_node("simple pipeline", simple_pipeline)
        # self.RAGraph.add_node("intermediate pipeline", intermediate_pipeline)
        # self.RAGraph.add_node("complex pipeline", complex_pipeline)
        # self.RAGraph.add_node("thresholder", threshold)
        # self.RAGraph.add_node("llm", answer)
        # self.RAGraph.add_node("web searcher", web_search)
        #
        # self.RAGraph.add_edge("entry", "moe")
        # self.RAGraph.add_conditional_edges(
        #     "entry",
        #     classify,
        #     {"simple":"simple pipeline", "intermediate":"intermediate pipeline", "complex":"complex pipeline"}
        # )
        # self.RAGraph.add_edge("simple pipeline", "thresholder")
        # self.RAGraph.add_edge("intermediate pipeline", "thresholder")
        # self.RAGraph.add_edge("complex pipeline", "thresholder")
        # self.RAGraph.add_conditional_edges(
        #     "thresholder",
        #     redirect,
        #     {"relevant":"llm", "irrelevant":"web searcher"}
        # )
        # self.RAGraph.add_edge("llm", END)
        # self.RAGraph.add_edge("web searcher", END)

        self.RAGraph = StateGraph(GraphState)
        self.RAGraph.set_entry_point("entry")
        self.RAGraph.add_node("entry", RunnablePassthrough)
        self.RAGraph.add_node("simple pipeline", simple_pipeline)
        self.RAGraph.add_node("llm", answer)
        self.RAGraph.add_node("fetch", fetch)
        self.RAGraph.add_edge("entry", "simple pipeline")
        self.RAGraph.add_edge("simple pipeline", "llm")
        # self.RAGraph.add_edge("llm", END)
        self.RAGraph.set_finish_point("llm")
        self.ragchain = self.RAGraph.compile()

    def set(self):
        self._rag_graph()

    def query(self, question):
        self.question = question
        state = {"question": self.question, "context": "", "answer": ""}
        answer_state = self.ragchain.invoke(state)
        self.answer = answer_state["answer"]
        return self.answer

    def ragas_evaluate(self, raise_exceptions=False):
        """
          Runs RAGAS evaluation on the RAG output
        """

        data = {
            "question": [self.question],
            "answer": [self.answer],
            "contexts": [[self.context]],
            "ground_truth": [self.ground_truth]
        }
        dataset = Dataset.from_dict(data)
        print(dataset['question'])
        print(dataset['answer'])
        print(dataset['contexts'])
        print(dataset['ground_truth'])
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ],
            raise_exceptions=raise_exceptions
        )
        df = result.to_pandas()
        return df
