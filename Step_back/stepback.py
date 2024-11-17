# from langchain.schema.output_parser import StrOutputParser
# from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnableLambda
from transformers import pipeline

class QuestionGen:
    def __init__(self, q_model):
        self.q_model = q_model
        # Create prompt templates and processing chain
        self.few_shot_examples = self.create_few_shot_examples()
        self.question_gen_chain = self.create_question_generation_chain()

    def create_few_shot_examples(self):
        # Define few-shot examples for the prompt
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "What can the members of The Police do?, What is lawful arrests?"
            },
            {
                "input": "Jan Sindel’s was born in what country?",
                "output": "What is Jan Sindel’s personal history?, What are the common countries?"
            },
            {
                "input": "Who is taller, Yao Ming or Shaq?",
                "output": "What is the height of Yao Ming?, What is the height of Shaq?"
            },
        ]
        return examples

    def create_question_generation_chain(self):
        # Simplified invocation to ensure compatibility with Hugging Face pipeline
        def generate_questions(question):
            # Prepare input for the Hugging Face pipeline
            examples_text = "\n".join(
                f"Input: {example['input']}\nOutput: {example['output']}"
                for example in self.few_shot_examples
            )
            prompt = (
                "You are an expert at world knowledge.\n"
                "Your task is to step back and abstract the original question "
                "to some more generic step-back questions, which are easier to answer.\n"
                "Here are a few examples:\n"
                f"{examples_text}\n"
                f"Input: {question}\nOutput:"
            )
            # Use the Hugging Face model to generate output
            result = self.q_model(prompt, max_length=1000, num_return_sequences=1)
            return result[0]["generated_text"].split("\nOutput:")[-1].split('\n')[0].split(',')  # Extract the generated text

        # Combine the model output with the parser
        return RunnableLambda(generate_questions) 

    def __call__(self, question):
        return self.question_gen_chain.invoke(question)
    
def query(q_model, question):
    stepback = QuestionGen(q_model)
    return stepback(question)

q_model=pipeline("text2text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
question="YOUR_QUESTION"
response=query(q_model,question)


# q_model = pipeline("text2text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct")
# question = '''What happens to the pressure, P, of an ideal gas if the temperature is increased by a factor of 2 and the volume is increased by a factor of 8 ?'''
# ['Output: What is the relationship between pressure and temperature?',
#  ' What is the relationship between pressure and volume?']

# question = '''If you have 3 moles of nitrogen and 4 moles of hydrogen to produce ammonia, which one will get exhausted first assuming a complete reaction?'''
# [' What is the chemical equation for the reaction?', ' What is the mole ratio of nitrogen to hydrogen?', ' What is the reaction between nitrogen and hydrogen?']
