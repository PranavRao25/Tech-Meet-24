import dspy
from typing import Union, List
from .search import Retriever
from .utils import QuestionToQuery, Information, limit_word_count
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

class AlternateQueryAgent:
    """
    Alternate Query Agent class that generates multiple alternate questions based on a given input question.
    """

    def __init__(self, model_pair, no_q=3):
        """
        Initializes the AlternateQueryAgent with a model pair and the number of alternate questions to generate.

        Parameters:
        model_pair (tuple): A tuple containing a question model and a parser model.
        no_q (int): The number of alternate questions to generate. Default is 3.
        """

        self._q_model = model_pair[0]
        self._parser = model_pair[1]
        self._turn = no_q
        template="""You are given a question {question}.
                  Generate """ + str(no_q) + """ alternate questions based on it. They should be numbered and separated by newlines.
                  Do not answer the questions.""".strip()
        self._prompt = ChatPromptTemplate.from_template(template)
        # Define a chain for generating alternate questions
        
        self._chain = {"question": RunnablePassthrough()} | self._prompt | self._q_model | self._parser

    def multiple_question_generation(self, question: str) -> list[str]:
        """
        Generates multiple alternate questions based on the given question.

        Parameters:
        question (str): The original question for which alternate versions are generated.

        Returns:
        list[str]: A list of alternate questions, including the original question as the last element.
        """

        # Generate alternate questions and include the original question in the list
        mul_qs = self._chain.invoke(question)#.split('\n')
        # mul_qs = ().append(question)
        return mul_qs

class Student(dspy.Module):
    """
    Answer questions using search-based retrieval and answer generation. This module conducts the following steps:
    1. Generate queries from the question.
    2. Search for information using the queries.
    3. Filter out unreliable sources.
    4. Generate an answer using the retrieved information.
    """

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
    ):
        """
        Initialize the Student module.

        Args:
            engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model or HF model to use.
            max_search_queries (int): The maximum number of search queries to generate.
            search_top_k (int): The number of top search results to consider.
            retriever (Retriever): The retriever object to use for searching information.
        """
        super().__init__()
        
        self.generate_queries = AlternateQueryAgent(model_pair=(engine, StrOutputParser()))
        self.retriever = retriever
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, question: str, ground_truth_url: str) -> dspy.Prediction:
        """
        Process the question to generate an answer.

        Args:
            question (str): The question to answer.
            ground_truth_url (str): The URL to exclude from search results.

        Returns:
            dspy.Prediction: The prediction containing queries, searched results, and the generated answer.
        """

        queries = self.generate_queries.multiple_question_generation(question=question)
        print(f"Queries: {queries}")
        queries = [
            q.replace("-", "").strip().strip('"').strip('"').strip()
            for q in queries.split("\n")
        ]
        queries = queries[: self.max_search_queries]
        
        # Search
        searched_results: List[Information] = self.retriever.forward(
            list(set(queries)), exclude_urls=[ground_truth_url]
        )
        
        if len(searched_results) > 0:
            # Evaluate: Simplify this part by directly using the top 1 snippet.
            info = ""
            for n, r in enumerate(searched_results):
                if len(r['snippets']) == 0:
                    continue
                info += f"[{n + 1}]: {r['snippets'][0]}\n"

            info = limit_word_count(info, 1000)
                            
        else:
            # When no information is found, the expert shouldn't hallucinate.
            info = "No info here. Please ask another question."

        return dspy.Prediction(queries=queries, searched_results=searched_results, answer=info)

class MidSearch(dspy.Module):
    """
    The interface for knowledge curation stage. Given query, return collected information.
    """

    def __init__(
        self,
        retriever: Retriever,
        student_engine_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        search_top_k: int,
    ):
        """
        Store args and finish initialization.

        Args:
            retriever (Retriever): The retriever object to use for searching information.
            student_engine_lm (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model or HF model to use.
            search_top_k (int): The number of top search results to consider.
        """
        self.retriever = retriever
        self.student_engine_lm = student_engine_lm
        self.search_top_k = search_top_k
        self.retriever = retriever
        self.student = Student(
            engine=student_engine_lm,
            max_search_queries=3,
            search_top_k=search_top_k,
            retriever=retriever
        )

    def query(
        self,
        query: str,
        ground_truth_url: str,
    ) -> list[str]:
        """
        Query the Student module to get the answer.

        Args:
            query (str): The query to search for.
            ground_truth_url (str): The URL to exclude from search results.

        Returns:
            str: The generated answer.
        """
        results = self.student(query, ground_truth_url)
        return [results.answer]
