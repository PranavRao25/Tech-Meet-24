from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFaceHub

class AutoWrapper:
    def __init__(self, model_name_or_path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize(self, text, **kwargs):
        return self.tokenizer(text, return_tensors="pt")

    def __call__(self, text, **kwargs):
        if not isinstance(text, str):
            text = text.to_string()
        inputs = self.tokenize(text, **kwargs)
        output_ids = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
class Thresholder:
    
    def __init__(self, model, parser):
        
        self._model = model
        self._parser = parser

        self.template = """
        
        Assess the relevance of the document to the question. \n
        Question: {question} \n
        Document: {document} \n
        Respond with 'yes' if relevant, 'no' if not and provide no premable or explanation. \n"""
        self._prompt = ChatPromptTemplate.from_template(self.template)
                
        self._docs = []
        self._chain = {
                    "question": RunnablePassthrough(),
                    "document": RunnableLambda(lambda x: self._docs) #peak 
                } | self._prompt | self._model | self._parser
        
    def grade(self, question: str, documents: list[str]) -> list[int]:
        
        grades = []
        for document in documents:
            
            self._docs = document
            answer = self._chain.invoke(question)
            print(answer)
            grade = answer.split("\n")[-1].strip().lower()
            print(grade)
            if "yes" in grade:
                grades.append(1)
            else:
                grades.append(0)
            
        return grades
    
if __name__ == "__main__":
    
    import os
    HF_TOKEN = "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"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

    model = AutoWrapper("geektech/flan-t5-large-lora-qa-gpt4")
    
    # model = HuggingFaceHub(
    #     repo_id="geektech/flan-t5-large-lora-qa-gpt4",
    #     model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
    # )
    thres = Thresholder(model, StrOutputParser())
    print(thres.grade("what is pathway?", ["pathway is shit", "pathway is company", "Nah I would win"]))