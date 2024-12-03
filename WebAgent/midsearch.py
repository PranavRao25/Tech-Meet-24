import dspy
from typing import Union
from .search import Retriever
from typing import List
from .utils import QuestionToQuery, Information, limit_word_count


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
        self.generate_queries = dspy.Predict(QuestionToQuery)
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
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            # Identify: Break down question into queries.
            queries = self.generate_queries(question=question).queries
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
