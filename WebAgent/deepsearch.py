from typing import List, Union, Tuple
from utils import AskQuestion, DialogueTurn, limit_word_count, QuestionToQuery, AnswerQuestion, Information
from search import Retriever
import dspy
import regex as re


class Student(dspy.Module):
    """
    Simulate a student asking questions based on a given query and dialogue history.
    """

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        """
        Initialize the Student module with a language model engine.

        Args:
            engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model engine to use for generating questions.
        """
        super().__init__()
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine

    def forward(
        self,
        query: str,
        dialogue_turns: List[DialogueTurn],
    ):
        """
        Generate a question based on the given query and dialogue history.

        Args:
            query (str): The main question or topic the student has.
            dialogue_turns (List[DialogueTurn]): The history of dialogue turns between the student and the teacher.

        Returns:
            dspy.Prediction: The generated question.
        """
        conv = []
        # Include only the last 4 dialogue turns in detail, and omit the answers for earlier turns to save space.
        for turn in dialogue_turns[:-4]:
            conv.append(
                f"You: {turn.user_utterance}\nTeacher: Omit the answer here due to space limit."
            )
        for turn in dialogue_turns[-4:]:
            conv.append(
                f"You: {turn.user_utterance}\nTeacher: {re.sub(r'\[\d+(?:,\s*\d+)*\]', '', turn.agent_utterance)}"
            )
        conv = "\n".join(conv)
        conv = conv.strip() or "N/A"
        conv = limit_word_count(conv, 2500)

        # Generate the question using the language model engine.
        with dspy.settings.context(lm=self.engine):
            question = self.ask_question(
                query=query, conv=conv
            ).question
        
        return dspy.Prediction(question=question)


class Teacher(dspy.Module):
    """Answer questions using search-based retrieval and answer generation. This module conducts the following steps:
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
        super().__init__()
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.retriever = retriever
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k

    def forward(self, question: str, ground_truth_url: str):
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

                try:
                    answer = self.answer_question(
                        conv=question, info=info
                    ).answer
                    
                except Exception as e:
                    print(f"Error occurs when generating answer: {e}")
                    answer = "Please ask another question."
                
            else:
                # When no information is found, the expert shouldn't hallucinate.
                answer = "No info here. Please ask another question."

        return dspy.Prediction(queries=queries, searched_results=searched_results, answer=answer)


class ConvSimulator(dspy.Module):
    """Simulate a conversation between a Teacher and a Student."""

    def __init__(
        self,
        teacher_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        student_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
    ):
        """
        Initialize the ConvSimulator with the given engines, retriever, and parameters.

        Args:
            teacher_engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model engine for the teacher.
            student_engine (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model engine for the student.
            retriever (Retriever): The retriever for searching information.
            max_search_queries_per_turn (int): The maximum number of search queries per turn.
            search_top_k (int): The number of top search results to consider.
            max_turn (int): The maximum number of dialogue turns.
        """
        super().__init__()
        self.student = Student(engine=student_engine)
        self.teacher = Teacher(
            engine=teacher_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever,
        )
        self.max_turn = max_turn

    def forward(
        self,
        query: str,
        ground_truth_url: str,
    ) -> dspy.Prediction:
        """
        Simulate the conversation between the student and the teacher.

        Args:
            query (str): The main question or topic the student has.
            ground_truth_url (str): The URL to exclude from search results.

        Returns:
            dspy.Prediction: The prediction containing the dialogue history.
        """
        dlg_history: List[DialogueTurn] = []
        for _ in range(self.max_turn):
            user_utterance = self.student(
                query=query, dialogue_turns=dlg_history
            ).question
            if user_utterance == "":
                print("Simulated Wikipedia writer utterance is empty.")
                break
            if user_utterance.startswith("bye"):
                break
            expert_output = self.teacher(
                question=user_utterance, ground_truth_url=ground_truth_url
            )
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=user_utterance,
            )
            dlg_history.append(dlg_turn)

        return dspy.Prediction(dlg_history=dlg_history)


class DeepSearch(dspy.Module):
    """
    The interface for performing deep search. Given a query, return collected information.
    """

    def __init__(
        self,
        retriever: Retriever,
        conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        student_engine_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_conv_turn: int,
    ):
        """
        Initialize the DeepSearch module with the given parameters.

        Args:
            retriever (Retriever): The retriever for searching information.
            conv_simulator_lm (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model engine for the conversation simulator.
            student_engine_lm (Union[dspy.dsp.LM, dspy.dsp.HFModel]): The language model engine for the student.
            max_search_queries_per_turn (int): The maximum number of search queries per turn.
            search_top_k (int): The number of top search results to consider.
            max_conv_turn (int): The maximum number of dialogue turns.
        """
        self.retriever = retriever
        self.conv_simulator_lm = conv_simulator_lm
        self.search_top_k = search_top_k
        self.retriever = retriever
        self.conv_simulator = ConvSimulator(
            teacher_engine=conv_simulator_lm,
            student_engine=student_engine_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn,
        )

    def query(
        self,
        query,
        ground_truth_url,
    ) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        Perform a deep search for the given query and return the collected information.

        Args:
            query (str): The main question or topic to search for.
            ground_truth_url (str): The URL to exclude from search results.

        Returns:
            List[Tuple[str, List[DialogueTurn]]]: The collected information from the conversation.
        """
        # Simulate the conversation to gather information
        conversation = self.conv_simulator(
            query=query,
            ground_truth_url=ground_truth_url,
        )
        
        # Collect the knowledge from the conversation history
        knowledge = ""
        for dialogue_turn in conversation.dlg_history:
            knowledge += dialogue_turn.log()['agent_utterance']
            
        return knowledge
