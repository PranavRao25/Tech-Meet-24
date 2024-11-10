from langchain_google_genai import ChatGoogleGenerativeAI
from abc import ABC, abstractmethod
from pathway.xpacks.llm.vector_store import VectorStoreClient

PATHWAY_PORT = 8765
client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

class ContextAgent(ABC):
    """
    Base Class for Query Context Agents
    """

    def __init__(self, vb, model_pair, reranker=None):
        self.vb = vb
        self.q_model = model_pair[0]  # llm
        self.parser = model_pair[1]  # parser
        self.cross_model = reranker

    @abstractmethod
    def query(self, question)->list[str]:
        """
            Retrieves total context for the user query
        """
        raise NotImplementedError('Implement Query function')

    @abstractmethod
    def fetch(self, question)->list[str]:
        """
            Retrieves context for a subquery by making a call to the Vector Database
        """
        raise NotImplementedError('Implement Fetch function')        
    
class ContextAgentCOT(ContextAgent):
    
    def query(self, question):
        
        messages = [
            (
                "system",
                "You are a structured assistant who decomposes complex questions into specific, three distinct sub-questions. "
                "Your task is to identify each part needed to answer the main question thoroughly. "
                "Provide each sub-question in a comma-separated list, without numbering or extra formatting, "
                "to facilitate retrieval-augmented generation (RAG) processing.\n\n"
                "Input format:\nMain question provided by the user.\n\n"
                "Output format:\nA list of three sub-questions separated by commas, in a single line.\n\n"
                "Example:\nInput: 'What is the history of artificial intelligence and its applications today?'\n"
                "Output: 'What is the origin of artificial intelligence?, How did AI develop over the years?, What are the current applications of AI?'"
            ),
            ("human", question)
        ]

        answer = self.fetch(question=question)
        subqueries = self.q_model.invoke(messages)
        for subquery in subqueries.content.split(','):
            
            subquery = str(subquery)
            answer += self.fetch(question=subquery)
            
        return answer
    
    def fetch(self, question):
        
        docs = self.vb(question)
        answer = ""
        for doc in docs:
            answer += doc['text']

        return answer
    
if __name__ == "__main__":
    
    user_question = "what is pathway?"

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )
    
    agent = ContextAgentCOT(client, (llm, None), None)
    print(agent.query(user_question))