from typing import List, Dict, Any
from langchain.schema import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough, RunnableMap

class ResponseSchema(BaseModel):
    """Schema for structured output"""
    answer: str = Field(description="The direct answer to the question")
    # reasoning: str = Field(description="Explanation of how the answer was derived from the context")
    # sources: List[str] = Field(description="List of sources used from the context")

class GeminiOutputParser(BaseOutputParser):
    """Parser to structure the LLM output"""

    def parse(self, text: str) -> ResponseSchema:
        """Parse the LLM output into structured format"""
        # print('RESPONSE : ', text)
        try:
            # Split the response into sections
            # sections = text.split("\n\n")
            # print('SECTIONS : ', sections)
            answer = text.replace("Answer: ", "").strip()
            # reasoning = sections[1].replace("Reasoning: ", "").strip()
            # sources = sections[2].replace("Sources: ", "").strip().split(", ")

            return ResponseSchema(
                answer=answer,
                reasoning='',
                sources=''
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
            #
            # Reasoning: [Explain how you arrived at the answer using the context]

            # Sources: [List the relevant sources from the context]
            template="""
            You are a helpful AI assistant. Using the provided context, answer the question.
            Format your response in the following way:

            Answer: [Provide a clear, direct answer]


            The inputs are
            Context: {context}

            Question: {question}
            """
        )

        # Create LangChain chain
        # self.chain = LLMChain(
        #     llm=self.llm,
        #     prompt=self.prompt_template
        # )
        
        self.chain = (
            RunnableMap({"question": RunnablePassthrough(), "context": RunnablePassthrough()})
            | self.prompt_template
            | self.llm
        )
        
    def process_query(
        self,
        question: str,
        context: List[str]
    ) -> ResponseSchema:
        """
        Process a query using the provided context
        
        Args:
            question: User's question
            context: List of context documents
            
        Returns:
            Structured response containing answer, reasoning, and sources
        """
        # Format context for the prompt
        # print(f"Context : {context}")
        formatted_context = "\n".join([
            f"Document {i+1}: {doc}"
            for i, doc in enumerate(context)
        ])
        
        # Get response from LLM
        response = self.chain.invoke(
            {
                "context": formatted_context, 
                "question": question
            }
        ).content
        
        # Parse and return structured output
        return self.output_parser.parse(response).answer

    def get_sources(self, response: ResponseSchema) -> List[str]:
        """Extract sources from the response"""
        return response.sources

    def get_answer(self, response: ResponseSchema) -> str:
        """Extract the main answer from the response"""
        return response.answer

    def get_reasoning(self, response: ResponseSchema) -> str:
        """Extract the reasoning from the response"""
        return response.reasoning
    

# llm=LLMAgent(google_api_key="api",model_name="gemini-pro")
# context_france=["\
# France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history.\
# ",
# "France is really Beautiful."]
# output=llm.process_query(question="What is the capital of France",context=context_france)
# print(output)