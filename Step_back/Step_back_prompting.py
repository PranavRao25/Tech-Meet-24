from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
import time
import getpass
import os


class QuestionAnsweringSystem:
    def __init__(self, google_api_key=None):
        # Setup environment variables for API key
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
        elif "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
        
        # Initialize the language model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
        # Initialize search API wrapper
        self.search = DuckDuckGoSearchAPIWrapper(max_results=4)
        
        # Create prompt templates and processing chain
        self.few_shot_examples = self.create_few_shot_examples()
        self.question_gen_chain = self.create_question_generation_chain()
        self.chain_with_step_back = self.create_answer_chain(with_step_back=True)
        self.chain_no_step_back = self.create_answer_chain(with_step_back=False)

    def create_few_shot_examples(self):
        # Define few-shot examples for the prompt
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "What can the members of The Police do?, What is lawful arrests?"
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "what is Jan Sindel’s personal history?, What are the common countries?"
            },
            {
                "input": "Who is taller, Yao Ming or Shaq?",
                "output": "what is the height of Yao Ming?, What is the height of Shaq?"
            },
        ]
        return examples

    def create_question_generation_chain(self):
        # Construct the few-shot prompt for generating step-back questions
        example_prompt = ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=self.few_shot_examples,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at world knowledge.
                          Your task is to step back and abstract the original question
                          to some more generic step-back questions,
                          which are easier to answer. Here are a few examples:"""),
            few_shot_prompt,
            ("user", "{question}"),
        ])
        return prompt | self.llm

    def retriever_list(self, query):
        # Retrieve answers to step-back questions by searching online
        answer = ''
        ques = ''
        query = query.content.split(",")
        for question in query:
            ques += question + '/'
            if question[-1] == '?':
                ans = self.search.run(ques)
                ques = ''
                answer += ans
                time.sleep(5)
        return answer

    def create_answer_chain(self, with_step_back):
        # Define the response prompt
        response_prompt_template = """You are an expert of world knowledge.
        I am going to ask you a question. Your response should be concise
        and refer to the following context if they are relevant.
        If they are not relevant, ignore them.
        {step_back_context}
        Original Question: {question}
        Answer:"""
        response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
        
        # Configure the chain based on whether to use step-back context or not
        if with_step_back:
            chain = {
                "step_back_context": self.question_gen_chain | self.retriever_list,
                "question": lambda x: x["question"]
            } | response_prompt | self.llm | StrOutputParser()
        else:
            chain = {
                "step_back_context": RunnableLambda(lambda x: x['question']) | self.retriever,
                "question": lambda x: x["question"]
            } | response_prompt | self.llm | StrOutputParser()
        
        return chain

    def retriever(self, query):
        # Retrieve content directly from the search wrapper
        return self.search.run(query)

    def get_answer(self, question, use_step_back=True):
        # Invoke the appropriate chain based on the user's choice for step-back
        input_data = {"question": question}
        if use_step_back:
            return self.chain_with_step_back.invoke(input_data)
        else:
            return self.chain_no_step_back.invoke(input_data)


# Usage example
if __name__ == "__main__":
    qa_system = QuestionAnsweringSystem()
    question = "If you have 3 moles of nitrogen and 4 moles of hydrogen to produce ammonia, which one will get exhausted first assuming a complete reaction?"
    
    # Get answer with step-back
    response_with_step_back = qa_system.get_answer(question, use_step_back=True)
    print("Response with Step Back:", response_with_step_back)
    
    # Get answer without step-back
    response_no_step_back = qa_system.get_answer(question, use_step_back=False)
    print("Response without Step Back:", response_no_step_back)
