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
# from MOE.Query_Classifier import
from QueryAgent.MCoTAgent import MCoTAgent
from QueryAgent.CoTAgent import CoTAgent
from rerankers.rerankers.reranker import *

class Pipeline:
    """
    Factory class for building retrieval pipelines that link retrievers with rerankers.
    Utilizes LangChain's RunnableLambda for modularity and reusability.
    """

    def __init__(self, retrieval_agent, reranker):
        """
        Initializes the Pipeline with a retrieval agent and a reranker.

        Parameters:
        retrieval_agent: The retrieval agent responsible for querying context.
        reranker: The reranker used to rank the retrieved context based on relevance.
        """
        self.simple_retrieval_agent = retrieval_agent
        self.simple_reranker = reranker

    def retrieve(self, question):
        """
        Retrieves and reranks the context for a given question.

        Parameters:
        question (str): The input question.

        Returns:
        list[str]: The ranked context.
        """
        context = self.simple_retrieval_agent.query(question)
        new_context = self.simple_reranker.rerank(question, context)
        return new_context


class RAG:
    """
    Framework for building and managing Agentic Retrieval-Augmented Generation (RAG) pipelines.

    Standard Operating Procedure:
    1. __init__
    2. retrieval_agent_prep (simple, intermediate, complex)
    3. reranker_prep (simple, intermediate, complex)
    4. moe_prep
    5. thresholder_prep
    6. web_search_prep
    7. step_back_prompt_prep
    8. set
    9. query
    """

    def __init__(self, vb, llm):
        """
        Initializes the RAG instance with a vector database and language model.

        Parameters:
        vb: The vector database used for retrieval.
        llm: The language model used for generation and evaluation.
        """
        self.vb = vb
        self.llm = llm

    def retrieval_agent_prep(self, q_model, parser, reranker, mode):
        """
        Sets up a retriever agent based on the specified mode.

        Parameters:
        q_model: The question model.
        parser: The parser model.
        reranker: The reranker.
        mode (str): The mode of the retriever (simple, intermediate, complex).
        """
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
        """
        Sets up a reranker agent based on the specified mode.

        Parameters:
        reranker: The reranker model.
        mode (str): The mode of the reranker (simple, intermediate, complex).
        """
        if mode == "simple":
            self.simple_reranker = Reranker(reranker)
        elif mode == "intermediate":
            self.intermediate_reranker = Reranker(reranker)
        elif mode == "complex":
            self.complex_reranker = LLMReranker(reranker)
        else:
            raise ValueError("Incorrect mode")

    def moe_prep(self, model):
        """
        Sets up a Mixture of Experts (MoE) agent.

        Parameters:
        model: The MoE model.
        """
        self.moe = model

    def thresholder_prep(self, model):
        """
        Sets up a Thresholder Agent.

        Parameters:
        model: The thresholder model.
        """
        self.thresholder = model

    def web_search_prep(self, model):
        """
        Sets up a Web Search Agent.

        Parameters:
        model: The web search model.
        """
        self.web_search_agent = model

    def step_back_prompt_prep(self, model):
        """
        Sets up a Step-Back Prompt Agent.

        Parameters:
        model: The step-back prompt model.
        """
        self.step_back_agent = model

    def pipeline_setup(self):
        """
        Configures the retrieval pipelines.
        """
        self.simple_pipeline = RunnableLambda(Pipeline(self.cot_agent, self.simple_reranker).retrieve)
        # self.intermediate_pipeline = RunnableLambda(Pipeline(self.mcot_agent, self.intermediate_reranker).retrieve)
        # self.complex_pipeline = RunnableLambda(Pipeline(self.tot_agent, self.complex_reranker).retrieve)

    def _context_prep(self):
        """
        Initializes an empty context.
        """
        self.context = ""

    def _rag_graph(self):
        """
        Builds the main RAG state graph.
        """

        class GraphState(TypedDict):
            """
            Represents the state of the RAG graph, including question, context, and answer.
            """
            question: str
            context: str
            answer: str

        def simple_pipeline(state):
            """
            Executes the simple pipeline for a given state.
            """
            context = self.simple_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}

        def fetch(state):
            """
            Fetches the context and updates the state.
            """
            self._context_prep()
            return {"question": state["question"], "context": self.context, "answer": state["answer"]}

        def answer(state):
            """
            Generates an answer based on the updated state.
            """
            return {"question": state["question"], "context": self.context, "answer": answer}

        self.pipeline_setup()
        self.RAGraph = StateGraph(GraphState)
        self.RAGraph.set_entry_point("entry")
        self.RAGraph.add_node("entry", RunnablePassthrough)
        self.RAGraph.add_node("simple pipeline", simple_pipeline)
        self.RAGraph.add_node("llm", answer)
        self.RAGraph.add_node("fetch", fetch)
        self.RAGraph.add_edge("entry", "simple pipeline")
        self.RAGraph.add_edge("simple pipeline", "llm")
        self.RAGraph.set_finish_point("llm")
        self.ragchain = self.RAGraph.compile()

    def set(self):
        """
        Finalizes the RAG graph setup.
        """
        self._rag_graph()

    def query(self, question):
        """
        Queries the RAG graph with a question.

        Parameters:
        question (str): The input question.

        Returns:
        str: The generated answer.
        """
        self.question = question
        state = {"question": self.question, "context": "", "answer": ""}
        answer_state = self.ragchain.invoke(state)
        self.answer = answer_state["answer"]
        return self.answer

    def ragas_evaluate(self, raise_exceptions=False):
        """
        Evaluates the RAG output using the RAGAS framework.

        Parameters:
        raise_exceptions (bool): Whether to raise exceptions during evaluation.

        Returns:
        DataFrame: Evaluation results as a DataFrame.
        """
        data = {
            "question": [self.question],
            "answer": [self.answer],
            "contexts": [[self.context]],
            "ground_truth": [self.ground_truth]
        }
        dataset = Dataset.from_dict(data)
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
