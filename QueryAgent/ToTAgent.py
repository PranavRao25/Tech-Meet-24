import re
from typing import List
from langchain_core.prompts import PromptTemplate


HOST = "127.0.0.1"
PORT = 8666

class ToTAgent:
    def __init__(self, vb, model_pair: tuple, reranker, best:int=3, threshold:int=7, breadth:int=5, max_depth:int=2) -> None:
        """
        Initialize the ToT Agent.
        :param vb: Instance of a vb for context retrieval
        :param threshold: Minimum relevance score to retain a context
        :param breadth: Number of subtopics to generate
        """

        self.vb = vb
        self.model = model_pair[0]
        self.rereanker = reranker
        self.threshold = threshold
        self.breadth = breadth
        self.deeped_contexts=set()
        self.max_depth=max_depth
        self.best=best
        self.relevant_contexts=set()
        
    def parse_topics(self, text: str) -> List[str]:
        """
        Parse the LLM response into clean topics.
        Removes markdown formatting and empty lines.
        """
        # Remove markdown bold syntax
        text = re.sub(r'\\(.?)\\*', r'\1', text)

        # Split into lines and clean each line
        lines = text.split('\n')
        print('topics are')
        print(lines)
        print()
        return lines

    def expand_thought(self, context: str) -> List[str]:
        """Expand a context into more specific topics using the LLM."""
        prompt_template = PromptTemplate.from_template("""
        Take the context "{context}" and list {breadth} specific subtopics or related areas of interest.
        Give a single sentence on a subtopic and seperate each subtopic with \n
        """)

        chain1 = prompt_template | self.model
        response = chain1.invoke({"context": context, "breadth": self.breadth})
        print("response")
        # print(response.content)
        # Parse the response into clean topics
        return self.parse_topics(response.content)

    def evaluate_topic(self, topic: str, query: str) -> int:
        """
        Evaluate the relevance of a topic to the main query.
        Returns a clean integer score.
        """
        prompt_template = PromptTemplate.from_template("""
        Rate the relevance of the following topic to the query '{query}':

        Topic: "{topic}"

        Respond with only a single number from 0 to 10.
        """)

        chain1 = prompt_template | self.model
        response = chain1.invoke({"query": query, "topic": topic})

        try:
            # Extract just the number from the response
            score = int(re.search(r'\d+', response.content).group())
            return min(max(score, 0), 10)  # Ensure score is between 0 and 10
        except (ValueError, AttributeError):
            print("could not evaluate topic")
            return 0  # Return 0 if we can't parse a valid score

    def recursive_retrieve(self, query: str, context: str, depth=0) -> List[dict]:
        """
        Recursively retrieve, expand, and evaluate contexts.
        Now with improved topic handling and error checking.
        """
        if depth >= self.max_depth:
            return []

        topics = self.expand_thought(context)
        relevant_contexts = set()

        for topic in topics:
            if not topic.strip():  # Skip empty topics
                continue

            score = self.evaluate_topic(topic, query)
            print(f"Evaluated topic '{topic}' with relevance score: {score}")

            if score >= self.threshold:
                try:
                    specific_contexts = self.vb.query(topic)
                    contexts = set()
                    for specific_context in specific_contexts:
                        contexts.add(specific_context['text'])
                    print("Retrieved Contexts")
                    print(contexts)
                    print()
                    relevant_contexts.update(contexts)
                    relevant_contexts.append(context)
                    for subcontext in contexts:
                        if subcontext not in self.deeped_contexts:
                            self.deeped_contexts.add(subcontext)
                            deeper_contexts = self.recursive_retrieve(
                                query, subcontext, depth + 1, self.max_depth
                            )
                            relevant_contexts.update(deeper_contexts)

                except Exception as e:
                    print(f"Error processing topic '{topic}': {str(e)}")
                    continue
        print(f"relevant_contexts at recursive{depth}")
        print(relevant_contexts)
        self.relevant_contexts.update(relevant_contexts)
        return relevant_contexts

    def query(self, query: str) -> list[str]:
        """Run the ToT retrieval and generate an answer with improved error handling."""
        try:
            initial_contexts = self.vb.query(query)
            print("initial_contexts")
            print(initial_contexts)
            if not initial_contexts:
                return "No initial contexts found for the query."
            contexts = set()
            for init_context in initial_contexts:
                contexts.add(init_context['text'])
            initial_contexts=contexts
            print("Retrieved Contexts")
            print(initial_contexts)
            print()
            relevant_contexts = set()
            for context in initial_contexts:
                if context:  # Skip empty contexts
                    if context not in self.deeped_contexts:
                        self.deeped_contexts.add(context)
                        contexts = self.recursive_retrieve(
                            query, context, depth=0
                        )
                        relevant_contexts.update(contexts)

            if not relevant_contexts:
                return "No relevant contexts found to answer the query."

            final_context = self._clean(query, list(relevant_contexts))
            print(final_context)
            return final_context

        except Exception as e:
            print("relevant_contexts:")
            print(self.relevant_contexts)
            print(f"An error occurred while processing your query: {str(e)}")      
            # return list(self.relevant_contexts)
    def _clean(self, question: str, alternate_context: list[str]) -> list[str]:
        """
        Cleans the retrieved contexts by reranking and returning only the best contexts.

        Parameters:
        question (str): The original user question for reference during reranking.
        alternate_context (list[str]): List of contexts generated from alternate queries.

        Returns:
        list[str]: A list of the top contexts after reranking.
        """
        return alternate_context
        # Rerank contexts and select the top 'best' number of contexts
        context = self._reranker.rerank(
            query=question,
            documents=alternate_context,
            return_documents=True
        )[:len(alternate_context) - self.best + 1]

        # Return the text of the top contexts
        return [c['text'] for c in context]
    
if __name__ == "__main__":
    
    import toml
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pathway.xpacks.llm.vector_store import VectorStoreClient # better to use pathway's default client as we can set a timeout limit on it

    client = VectorStoreClient(HOST, PORT, timeout=60)

    config = toml.load('../config.toml')
    os.environ["GOOGLE_API_KEY"] = config["GEMINI_API"]
    model = ChatGoogleGenerativeAI(model="gemini-pro")
    
    HOST = "127.0.0.1"
    PORT = 8666

    tb = ToTAgent(vb=client, model_pair=(model, None), reranker=None, best=3, threshold=7, breadth=5, max_depth=2)    
    tb.query("What is pathway?")