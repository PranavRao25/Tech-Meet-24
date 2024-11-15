import dspy
from typing import OrderedDict


def limit_word_count(input_string: str, max_word_count:int = 1000) -> str:
    """
    Limit the word count of an input string to a specified maximum, while preserving the integrity of complete lines.
    Args:
        input_string (str): The string to be truncated.
        max_word_count (int): The maximum number of words allowed in the truncated string.

    Returns:
        str: The truncated string with word count limited to `max_word_count`, preserving complete lines.
    """

    word_count = 0
    limited_string = ""

    # Split the input string by lines
    for word in input_string.split("\n"):
        line_words = word.split()
        # Split each line into words and count them
        for lw in line_words:
            if word_count < max_word_count:
                limited_string += lw + " "
                word_count += 1
            else:
                break
        if word_count >= max_word_count:
            break
        limited_string = limited_string.strip() + "\n"

    return limited_string.strip()


class Information:
    """Class to represent detailed information."""

    def __init__(self, url, description, snippets, title):
        """Initialize the Information object with detailed attributes.

        Args:
            url (str): The unique URL serving as the identifier for the information.
            description (str): Detailed description.
            snippets (list): List of brief excerpts or snippet.
            title (str): The title or headline of the information.
        """
        self.description = description
        self.snippets = snippets
        self.title = title
        self.url = url

    @classmethod
    def from_dict(cls, info_dict):
        """Create an Information object from a dictionary."""
        info = cls(
            url=info_dict["url"],
            description=info_dict["description"],
            snippets=info_dict["snippets"],
            title=info_dict["title"],
        )
        return info

    def to_dict(self):
        """Convert the Information object to a dictionary."""
        return {
            "url": self.url,
            "description": self.description,
            "snippets": self.snippets,
            "title": self.title,
        }


class DialogueTurn:
    def __init__(
        self,
        agent_utterance: str = None,
        user_utterance: str = None,
    ):
        """Initialize the DialogueTurn object with agent and user utterances."""
        self.agent_utterance = agent_utterance
        self.user_utterance = user_utterance

    def log(self):
        """
        Returns a json object that contains all information inside `self`
        """
        return OrderedDict(
            {
                "agent_utterance": self.agent_utterance,
                "user_utterance": self.user_utterance,
            }
        )


class AskQuestion(dspy.Signature):
    """You are a student and you have a question. Break down your main question into smaller, precise topics. Based on that topic, Ask a thoughtful, relevant question to gather useful information through google search.
    Guidelines:
        Ask a single question at a time, without conversation history or anything else.
        Never repeat a question that you previously asked.
        Keep each sub-question closely related to your sub topic.
    End the conversation by saying “bye” once you’ve gathered all the information you need.    
    """

    query = dspy.InputField(prefix="Main question that you have: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class QuestionToQuery(dspy.Signature):
    """You want to answer the question using Google search. What do you type in the search box? Give the queries without saying any other word.
    Input format: Question.
    Output format:
    - query 1
    - query 2
    ...
    - query n"""

    question = dspy.InputField(prefix="Question you want to answer: ", format=str)
    queries = dspy.OutputField(format=str)


class AnswerQuestion(dspy.Signature):
    """You are an expert who can use information effectively. You have gathered the related information and will now use the information to write a article.
    Make your response as informative as possible, ensuring that every sentence is supported by the gathered information. Do not cite. If the [gathered information] is not directly related to the [topic] or [question], provide the most relevant answer based on the available information. 
    If no appropriate answer can be formulated, respond with, “No info”.
    """

    conv = dspy.InputField(prefix="Question:\n", format=str)
    info = dspy.InputField(prefix="Gathered information:\n", format=str)
    answer = dspy.OutputField(
        prefix="Now give your response. (Try to use as many different sources as possible and add do not hallucinate.)\n",
        format=str,
    )