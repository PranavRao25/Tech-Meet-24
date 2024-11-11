from ContextAgent import *


class CoTAgent(ContextAgent):

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
