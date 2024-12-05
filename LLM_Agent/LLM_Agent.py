from typing import List
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
import logging


class ResponseSchema(BaseModel):
    """Schema for structured output"""
    answer: str = Field(description="The direct answer to the question")


class GeminiOutputParser(BaseOutputParser):
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
        google_api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.5,
        max_tokens: int = 2000,
    ):
        """Initialize the LLM Agent"""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self.output_parser = GeminiOutputParser()

    @staticmethod
    def decide_template(context: str, question: str) -> str:
        print("deciding template")
        """Decide the appropriate template based on the presence of context"""
        if not context.strip():  # No context provided
            print("no context")
            return f"""
            You are a helpful AI assistant. Answer the question in the following way.
            Answer: [Provide a clear, direct answer]

            The input is
            Question: {question}
            """
        else:  # Context provided
            print("context provided")
            return f"""
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
        prompt_text = self.decide_template(formatted_context, question)

        # Create a dynamic prompt template and chain
        prompt_template = PromptTemplate(input_variables=[], template=prompt_text)
        chain = LLMChain(llm=self.llm, prompt=prompt_template)

        # Get response from the LLM
        logging.info(f"Context:\n{formatted_context}")
        logging.info(f"Prompt:\n{prompt_text}")
        response = chain.run({})
        logging.info(f"Response: {response}")

        # Parse and return the structured output
        return self.output_parser.parse(response).answer
