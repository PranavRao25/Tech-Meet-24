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
from MOE.llm_query_classifier import QueryClassifier
from QueryAgent.MCoTAgent import MCoTAgent
from QueryAgent.CoTAgent import CoTAgent
from Step_back.stepback import QuestionGen
from langchain.schema.runnable import RunnableLambda
from transformers import pipeline
from QueryAgent.ToTAgent import ToTAgent
from QueryAgent.BasicAgent import BasicAgent
from WebAgent.main import WebAgent
from rerankers.rerankers.reranker import *
from Thresholder.Thresholder import Thresholder
from concurrent.futures import ThreadPoolExecutor
from GuardRails import validate_query
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
        logging.info("Retriever Started")
        if self.step_back_agent is not None:
            questions = self.step_back_agent(question)
            questions.append(question)
            contexts = []
            for question in questions:
                contexts += self.simple_retrieval_agent.invoke(question)
        else:
            contexts = self.simple_retrieval_agent.invoke(question)
        new_context = self.simple_reranker.rerank(question, contexts)
        logging.info("Retriever Finished")
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
        self.web_results = None

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
            self._basic_agent = RunnableLambda(BasicAgent(self._vb, (q_model, parser), reranker).query)
        elif mode == "intermediate":
            self._cot_agent = RunnableLambda(CoTAgent(self._vb, (q_model, parser), reranker).query)
        elif mode == "complex":
            self._mcot_agent = RunnableLambda(MCoTAgent(self._vb, (q_model, parser), reranker).query)
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

        self._moe_agent = RunnableLambda(QueryClassifier(model).classify)

    def thresholder_prep(self, model):
        """
        Sets up a Thresholder Agent.

        Parameters:
        model: The thresholder model.
        """

        self._thresholder = Thresholder(model=model) # make it runnable please

    def web_search_prep(self, model):
        """
        Sets up a Web Search Agent.

        Parameters:
        model: The web search model.
        """

        self._web_search_agent = RunnableLambda(WebAgent(model=model).query)

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

        self._simple_pipeline = RunnableLambda(Pipeline(self._basic_agent, self._simple_reranker).retrieve)
        self._intermediate_pipeline = RunnableLambda(Pipeline(self._cot_agent, self._simple_reranker).retrieve)
        self._complex_pipeline = RunnableLambda(Pipeline(self._mcot_agent, self._intermediate_reranker, step_back_agent=self._step_back_agent).retrieve)

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
            # return "intermediate"
            _answer =  self._moe_agent.invoke(state['question'])
            print(_answer)
            return _answer

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
            with ThreadPoolExecutor() as executor: # noob
                future_web_results = executor.submit(self._web_search_agent.invoke, state["question"])
                future_context = executor.submit(self._intermediate_pipeline.invoke, state["question"])

                self.web_results = future_web_results.result()
                context = future_context.result()
            return {"question": state["question"], "context": context, "answer": state["answer"]}
        
        def _complex_pipeline(state):
            """
            Executes the complex pipeline for a given state.
            """
            print("complex pipeline has been chosen\n")
            with ThreadPoolExecutor() as executor: # noob
                future_web_results = executor.submit(self._web_search_agent.invoke, state["question"])
                future_context = executor.submit(self._complex_pipeline.invoke, state["question"])
    
                self.web_results = future_web_results.result()
                context = future_context.result()
            return {"question": state["question"], "context": context, "answer": state["answer"]}
        
        def _threshold(state):
            return {"question": state["question"], "context": state["context"], "answer": state["answer"]}
        
        def _OOD(state):
            return {"question": state["question"], "context": state["context"], "answer": state["answer"]}

        def _classify_answer(state):
            
            grades = self._thresholder.grade(state['question'], state['context'])
            print(grades)
            
            if grades.count(1) / len(grades) >= 0.2:
                return "llm"
            elif grades.count(0) / len(grades) >= 0.4:
                return "web_llm"
            else:
                return "web"

        def _answer(state):
            """
            Generates an answer based on the updated state.
            """
            logger.info("Documents are relevant")
            bot_answer = self._llm.process_query(state["question"], state["context"])
            return {"question": state["question"], "context": state["context"], "answer": bot_answer}

        def _search(state):
            logger.info("Documents are Irrelevant")
            
            if self.web_results is None:
                self.web_results = self._web_search_agent.invoke(state['question'])
            
            web_result = self.web_results
            self.web_results = None
            bot_answer = self._llm.process_query(state["question"], web_result)
            return {"question": state["question"], "context": state["context"], "answer": bot_answer}

        def _ambiguous(state):
            logger.info("Documents are Ambigious")
            
            if self.web_results is None:
                self.web_results = self._web_search_agent.invoke(state['question'])
            
            web_result = self.web_results
            self.web_results = None
            context = ["retrieved context:\n"]
            context.extend(state["context"])
            context.extend(["web results:\n"])
            context.extend(web_result)
            bot_answer = self._llm.process_query(state["question"], context)
            return {"question": state["question"], "context": state["context"], "answer": bot_answer}

        self._pipeline_setup()
        self._RAGraph = StateGraph(GraphState)
        self._RAGraph.set_entry_point("entry")
        self._RAGraph.add_node("entry", RunnablePassthrough())
        self._RAGraph.add_node("simple pipeline", _simple_pipeline)
        self._RAGraph.add_node("intermediate pipeline", _intermediate_pipeline)
        self._RAGraph.add_node("complex pipeline", _complex_pipeline)
        self._RAGraph.add_node("OOD", _OOD)
        self._RAGraph.add_node("thresholder", _threshold)
        self._RAGraph.add_node("llm", _answer)
        self._RAGraph.add_node("web", _search)
        self._RAGraph.add_node("web_llm", _ambiguous)
        self._RAGraph.add_conditional_edges(
            "entry",
            _classify_query,
            {
                "simple": "simple pipeline",
                "intermediate": "intermediate pipeline",
                "complex": "complex pipeline",
                "trivial": "OOD"
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
                "web": "web",
                "web_llm": "web_llm"
            }
        )
        self._RAGraph.add_edge("OOD", "llm")
        self._RAGraph.add_edge("llm", END)
        self._RAGraph.add_edge("web", END)
        self._RAGraph.add_edge("web_llm", END)
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
        
        val = validate_query(question)
        if val is not None:
            return val
        
        self._question = question
        state = {"question": self._question, "context": "", "answer": ""}
        self._rag_graph()
        answer_state = self._ragchain.invoke(state)
        self._answer = answer_state["answer"]
        return self._answer

    def ragas_evaluate(self, questions: list[str], ground_truths:list[str], raise_exceptions=False):
        """
        Evaluates the RAG output using the RAGAS framework.

        Parameters:
        raise_exceptions (bool): Whether to raise exceptions during evaluation.

        Returns:
        DataFrame: Evaluation results as a DataFrame.
        """
        
        answers = []
        contexts = []
        for question in questions:
            answers.append(self.query(question))
            contexts.append(self._context)
         
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": [contexts],
            "ground_truth": ground_truths
        })
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
