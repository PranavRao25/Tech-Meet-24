from typing import List
from langchain.schema import BaseOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import logging
import os
import getpass

class ResponseSchema(BaseModel):
    """Schema for structured output"""
    answer: str = Field(description="The direct answer to the question")


class OpenAIParser(BaseOutputParser):
    """Parser to structure the LLM output"""

    def parse(self, text: str) -> ResponseSchema:
        """Parse the LLM output into structured format"""
        try:
            answer = text.replace("Answer: ", "").strip()
            return ResponseSchema(answer=answer)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}")


class LLMAgent:
    """LLM Agent class for RAG pipeline integration"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 256,
    ):

        if "OPENAI_API_KEY" not in os.environ:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

        """Initialize the LLM Agent"""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.output_parser = OpenAIParser()

    @staticmethod
    def decide_template(context: str) -> str:
        print("deciding template")
        """Decide the appropriate template based on the presence of context"""
        if not context.strip():  # No context provided
            print("no context")
            return """
            You are a helpful AI assistant. Answer the question in the following way.
            Answer: [Provide a clear, direct answer]

            The input is
            Question: {question}
            """
        else:  # Context provided
            print("context provided")
            return """
            You are a helpful AI assistant. Using the provided context, answer the question.
            Prioritize the retrieved context over external knowledge or assumptions.
            Format your response in the following way:
            Answer: [Provide a clear, direct answer]

            The inputs are
            Context: {context}

            Question: {question}
            """

    def process_query(
        self,
        question: str,
        context: List[str],
    ) -> str:
        """
        Process a query using the provided context.

        Args:
            question: User's question
            context: List of context documents

        Returns:
            Structured response containing the answer
        """
        # Format the context
        formatted_context = "\n".join(context)

        # Dynamically create the prompt using the decided template
        prompt_text = self.decide_template(formatted_context)

        # Create a dynamic prompt template and chain
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that helps everyone and behaves nicely.",
                ),
                ("human", prompt_text),
            ]
        )
        chain = prompt | self.llm

        # Get response from the LLM
        logging.info(f"Context:\n{formatted_context}")
        logging.info(f"Prompt:\n{prompt_text}")
        response = chain.invoke({
            "context" : formatted_context,
            "question" : question
        })
        logging.info(f"Response: {response}")

        # Parse and return the structured output
        return self.output_parser.parse(response.content).answer

if __name__ == '__main__':
    
    agent = LLMAgent(model_name="gpt-4o-mini", temperature=0.7, max_tokens=256) #max_tokens=1024???
    question = input("Enter a question: ")
    context = "france is a country in europe"
    print(agent.process_query(question, context))