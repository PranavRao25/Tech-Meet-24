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
# from MOE.Query_Classifier import QueryClassifier
from MOE.llm_query_classifier import QueryClassifier
from QueryAgent.MCoTAgent import MCoTAgent
from QueryAgent.CoTAgent import CoTAgent
from Step_back.stepback import QuestionGen
from langchain.schema.runnable import RunnableLambda
from transformers import pipeline
from QueryAgent.ToTAgent import ToTAgent
from WebAgent.main import WebAgent
from rerankers.rerankers.reranker import *
from Thresholder.Thresholder import Thresholder

class Pipeline:
    """
    Factory class for building retrieval pipelines that link retrievers with rerankers.
    Utilizes LangChain's RunnableLambda for modularity and reusability.
    """

    def __init__(self, retrieval_agent, reranker, step_back_agent=None):
        """
        Initializes the Pipeline with a retrieval agent and a reranker.

        Parameters:
        retrieval_agent: The retrieval agent responsible for querying context.
        reranker: The reranker used to rank the retrieved context based on relevance.
        """

        self.simple_retrieval_agent = retrieval_agent
        self.simple_reranker = reranker
        self.step_back_agent = step_back_agent  # takes a str and returns a list of str

    def retrieve(self, question):
        """
        Retrieves and reranks the context for a given question.

        Parameters:
        question (str): The input question.

        Returns:
        list[str]: The ranked context.
        """

        if self.step_back_agent is not None:
            questions = self.step_back_agent(question)
            contexts = []
            for question in questions:
                contexts += self.simple_retrieval_agent.invoke(question)
        else:
            contexts = self.simple_retrieval_agent.invoke(question)
        new_context = self.simple_reranker.rerank(question, contexts)

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
            self._cot_agent = RunnableLambda(CoTAgent(self._vb, (q_model, parser), reranker).query)
        elif mode == "intermediate":
            self._mcot_agent = RunnableLambda(MCoTAgent(self._vb, (q_model, parser), reranker).query)
        elif mode == "complex":
            self._tot_agent = RunnableLambda(ToTAgent(self._vb, (q_model, parser), reranker).query)
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
            self._complex_reranker = Reranker(reranker)
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

        self._thresholder = Thresholder(model=model) # make it runnable please

    def web_search_prep(self):
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
        tokenizer: The step-back prompt tokenizer.
        """

        self._step_back_agent = QuestionGen(model)

    def _pipeline_setup(self):
        """
        Configures the retrieval pipelines.
        """

        self._simple_pipeline = RunnableLambda(Pipeline(self._cot_agent, self._simple_reranker).retrieve)
        self._intermediate_pipeline = RunnableLambda(Pipeline(self._mcot_agent, self._intermediate_reranker, step_back_agent=self._step_back_agent).retrieve)
        self._complex_pipeline = RunnableLambda(Pipeline(self._tot_agent, self._complex_reranker, step_back_agent=self._step_back_agent).retrieve)

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
            return "simple"
            import numpy
            return numpy.random.choice(["simple", "intermediate", "complex"])
            return "simple"
            return self._moe_agent.invoke(state['question'])

        def _simple_pipeline(state):
            """
            Executes the simple pipeline for a given state.
            """
            print("simple pipeline has been chosen\n")
            context = self._simple_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}

        def _intermediate_pipeline(state):
            """
            Executes the intermediate pipeline for a given state.
            """
            print("intermediate pipeline has been chosen\n")
            context = self._intermediate_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}
        
        def _complex_pipeline(state):
            """
            Executes the complex pipeline for a given state.
            """
            print("complex pipeline has been chosen\n")
            context = self._complex_pipeline.invoke(state["question"])
            return {"question": state["question"], "context": context, "answer": state["answer"]}
        
        def _threshold(state):
            return {"question": state["question"], "context": state["context"], "answer": state["answer"]}

        def _classify_answer(state):
            
            grades = self._thresholder.grade(state['question'], state['context'])
            print(grades)
            thres = sum(grades) / len(state['context'])
            print(thres)
            if thres >= 0: #0.4
                return 'llm'
            else:
                return 'web'

        def _answer(state):
            """
            Generates an answer based on the updated state.
            """

            bot_answer = self._llm.process_query(state["question"], state["context"])
            return {"question": state["question"], "context": state["context"], "answer": bot_answer}

        def _search(state):
            answer = self._web_search_agent.invoke(state['question'])
            bot_answer = _answer({"question": state["question"], "context": answer, "answer": state["answer"]})
            return {"question": state["question"], "context": state["context"], "answer": bot_answer}

        self._pipeline_setup()
        self._RAGraph = StateGraph(GraphState)
        self._RAGraph.set_entry_point("entry")
        self._RAGraph.add_node("entry", RunnablePassthrough())
        self._RAGraph.add_node("simple pipeline", _simple_pipeline)
        self._RAGraph.add_node("intermediate pipeline", _intermediate_pipeline)
        self._RAGraph.add_node("complex pipeline", _complex_pipeline)
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
        answer_state = self._ragchain.invoke(state)
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
