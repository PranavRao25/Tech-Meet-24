import os
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFaceHub

from transformers import AutoTokenizer, AutoModelForCausalLM

class AutoWrapper:
    def __init__(self, model_name_or_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize(self, text, **kwargs):
        return self.tokenizer(text, return_tensors="pt")

    def __call__(self, text, **kwargs):
        if not isinstance(text, str):
            text = text.to_string()
        inputs = self.tokenize(text, **kwargs)
        output_ids = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    def to(self, device:str):
        self.model = self.model.to(device)
        return self

class Thresholder:
    
    def __init__(self, model, parser=StrOutputParser()):
        
        self._model = model
        self._parser = parser
        print(model)

        self.template = """
        
        Assess the relevance of the document to the question. \n
        Question: {question} \n
        Document: {document} \n
        Respond with 'yes' if relevant, 'no' if not and provide no premable or explanation, just yes or no.\n"""
        self._prompt = ChatPromptTemplate.from_template(self.template)
                
        self._docs = []
        self._chain = {
                    "question": RunnablePassthrough(),
                    "document": RunnableLambda(lambda x: self._docs) #peak 
                } | self._prompt | self._model #| self._parser
        
    def grade(self, question: str, documents: list[str]) -> list[int]:
        
        grades = []
        for document in documents:
            
            self._docs = document
            answer = self._chain.invoke(question)
            grade = answer.split("\n")[-1].strip().lower()
            if "yes" in grade:
                grades.append(1)
            else:
                grades.append(0)
            
        return grades
    
if __name__ == "__main__":
    
    HF_TOKEN = "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"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    
    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
    )
    thres = Thresholder(model, StrOutputParser())
    print(thres.grade("what is pathway?", ["pathway is shit", "pathway is company", "Nah I would win"]))