from typing import List, Dict, Any
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

class ResponseSchema(BaseModel):
    """Schema for structured output"""
    answer: str = Field(description="The direct answer to the question")
    reasoning: str = Field(description="Explanation of how the answer was derived from the context")
    sources: List[str] = Field(description="List of sources used from the context")

class GeminiOutputParser(BaseOutputParser):
    """Parser to structure the LLM output"""

    def parse(self, text: str) -> ResponseSchema:
        """Parse the LLM output into structured format"""
        try:
            # Split the response into sections
            sections = text.split("\n\n")

            answer = sections[0].replace("Answer: ", "").strip()
            reasoning = sections[1].replace("Reasoning: ", "").strip()
            sources = sections[2].replace("Sources: ", "").strip().split(", ")

            return ResponseSchema(
                answer=answer,
                reasoning=reasoning,
                sources=sources
            )
        except Exception as e:
            raise ValueError(f"Failed to parse LLM output: {e}")

class LLMAgent:
    """LLM Agent class for RAG pipeline integration"""

    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-pro",
        temperature: float = 0.5,
        max_tokens: int = 2000
    ):
        """Initialize the LLM Agent"""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        self.output_parser = GeminiOutputParser()

        # Define the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful AI assistant. Using the provided context, answer the question.
            Format your response in the following way:

            Answer: [Provide a clear, direct answer]

            Reasoning: [Explain how you arrived at the answer using the context]

            Sources: [List the relevant sources from the context]

            The inputs are
            Context: {context}

            Question: {question}
            """
        )

        # Create LangChain chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )

    def process_query(
        self,
        question: str,
        context: str
    ) -> ResponseSchema:
        """
        Process a query using the provided context

        Args:
            question: User's question
            context: context as a string

        Returns:
            Structured response containing answer, reasoning, and sources
        """
        response = self.chain.run(
            context=context,
            question=question
        )

        # Parse and return structured output
        return self.output_parser.parse(response)

    def get_sources(self, response: ResponseSchema) -> List[str]:
        """Extract sources from the response"""
        return response.sources

    def get_answer(self, response: ResponseSchema) -> str:
        """Extract the main answer from the response"""
        return response.answer

    def get_reasoning(self, response: ResponseSchema) -> str:
        """Extract the reasoning from the response"""
        return response.reasoning