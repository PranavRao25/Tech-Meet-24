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
from MOE.Query_Classifier import QueryClassifier
from QueryAgent.MCoTAgent import MCoTAgent
from QueryAgent.CoTAgent import CoTAgent
from WebAgent.main import WebAgent
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
        self._vb = vb
        self._llm = llm

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
            self._cot_agent = CoTAgent(self._vb, (q_model, parser), reranker)
        elif mode == "intermediate":
            self._mcot_agent = MCoTAgent(self._vb, (q_model, parser), reranker)
        elif mode == "complex":
            self._tot_agent = None
        else:
            raise ValueError("Incorrect mode")

    def ground_truth_prep(self):
        self._ground_truth = None

    def reranker_prep(self, reranker, mode):
        """
        Sets up a reranker agent based on the specified mode.

        Parameters:
        reranker: The reranker model.
        mode (str): The mode of the reranker (simple, intermediate, complex).
        """

        if mode == "simple":
            self._simple_reranker = Reranker(reranker)
        elif mode == "intermediate":
            self._intermediate_reranker = Reranker(reranker)
        elif mode == "complex":
            self._complex_reranker = LLMReranker(reranker)
        else:
            raise ValueError("Incorrect mode")

    def moe_prep(self, model):
        """
        Sets up a Mixture of Experts (MoE) agent.

        Parameters:
        model: The MoE model.
        """

        self._moe_agent = RunnableLambda(QueryClassifier(model=model).classify)

    def thresholder_prep(self, model):
        """
        Sets up a Thresholder Agent.

        Parameters:
        model: The thresholder model.
        """

        self._thresholder = model

    def web_search_prep(self, model):
        """
        Sets up a Web Search Agent.

        Parameters:
        model: The web search model.
        """

        self._web_search_agent = RunnableLambda(WebAgent().query)

    def step_back_prompt_prep(self, model):
        """
        Sets up a Step-Back Prompt Agent.

        Parameters:
        model: The step-back prompt model.
        """

        self._step_back_agent = model

    def _pipeline_setup(self):
        """
        Configures the retrieval pipelines.
        """

        self._simple_pipeline = RunnableLambda(Pipeline(self._cot_agent, self._simple_reranker).retrieve)
        self._intermediate_pipeline = RunnableLambda(Pipeline(self._mcot_agent, self._intermediate_reranker).retrieve)
        # self.complex_pipeline = RunnableLambda(Pipeline(self._tot_agent, self._complex_reranker).retrieve)

    def _context_prep(self, question:str):
        """
        Initializes an empty context.
        """

        self._context = self._vb(question)

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

        def _classify_query(state):
            # "simple"
            # "intermediate"
            # "complex"
            return self._moe_agent.invoke(state['question'])

        def _simple_pipeline(state):
            """
            Executes the simple pipeline for a given state.
            """

            context = self._simple_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}

        def _intermediate_pipeline(state):
            """
            Executes the simple pipeline for a given state.
            """
            context = self._intermediate_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}

        def _threshold(state):
            return {"question": state["question"], "context": state["context"], "answer": state["answer"]}

        def _classify_answer(state):
            q = state['question']
            return 'llm' if q == True else 'web'

        def _answer(state):
            """
            Generates an answer based on the updated state.
            """
            bot_answer = self._llm.process_query(state["question"], state["context"])
            return {"question": state["question"], "context": state["context"], "answer": bot_answer}

        def _search(state):
            answer = self._web_search_agent.invoke(state['question'])
            return {"question": state["question"], "context": state["context"], "answer": answer}

        self._pipeline_setup()
        self._RAGraph = StateGraph(GraphState)
        self._RAGraph.set_entry_point("entry")
        self._RAGraph.add_node("entry", RunnablePassthrough())
        self._RAGraph.add_node("simple pipeline", _simple_pipeline)
        self._RAGraph.add_node("intermediate pipeline", _intermediate_pipeline)
        self._RAGraph.add_node("thresholder", _threshold)
        self._RAGraph.add_node("llm", _answer)
        self._RAGraph.add_node("web", _search)
        self._RAGraph.add_conditional_edges(
            "entry",
            _classify_query,
            {
                "simple": "simple pipeline",
                "intermediate": "intermediate pipeline",
                "complex": "complex pipeline"
            }
        )
        self._RAGraph.add_edge("simple pipeline", "thresholder")
        self._RAGraph.add_edge("intermediate pipeline", "thresholder")
        self._RAGraph.add_edge("complex pipeline", "thresholder")
        self._RAGraph.add_conditional_edges(
            "thresholder",
            _classify_answer,
            {
                "llm": "llm",
                "web": "web"
            }
        )
        self._RAGraph.add_edge("llm", END)
        self._RAGraph.add_edge("web", END)
        self._ragchain = self._RAGraph.compile()

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
        
        self._question = question
        state = {"question": self._question, "context": "", "answer": ""}
        self._rag_graph()
        answer_state = self.ragchain.invoke(state)
        self._answer = answer_state["answer"]
        return self._answer

    def ragas_evaluate(self, raise_exceptions=False):
        """
        Evaluates the RAG output using the RAGAS framework.

        Parameters:
        raise_exceptions (bool): Whether to raise exceptions during evaluation.

        Returns:
        DataFrame: Evaluation results as a DataFrame.
        """
        data = {
            "question": [self._question],
            "answer": [self._answer],
            "contexts": [[self._context]],
            "ground_truth": [self._ground_truth]
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
