import os
import re
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

def chunk_text(text, chunk_size=250, overlap=25):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

class Thresholder:
    
    def __init__(self, model, parser=StrOutputParser()):
        
        self._model = model
        self._parser = parser
        print(model)

        self.template = """
        You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a ternary score 'relevant' or 'ambiguous' or 'irrelevant' score to indicate the relevancy of the document to question. \n
        Provide no premable or explanation, just a single score. Don't answer with '-'.
        """
        self._prompt = ChatPromptTemplate.from_template(self.template)
                
        self._docs = []
        self._chain = {
                    "question": RunnablePassthrough(),
                    "document": RunnableLambda(lambda x: self._docs) #peak 
                } | self._prompt | self._model | self._parser
        
    def grade(self, question: str, documents: list[str]) -> list[int]:
        
        grades = []
        for document in documents:
            
            chunks = chunk_text(document)
            inter_grade = []
            for chunk in chunks:

                self._docs = chunk
                answer = self._chain.invoke(question)
                check = answer
                print(f"CHECK: {check}\n")
            
                relevant = len(re.findall(r'\brelevant\b', check, re.IGNORECASE)) - 2
                ambiguous = len(re.findall(r'\bambiguous\b', check, re.IGNORECASE)) - 1
                irrelevant = len(re.findall(r'\birrelevant\b', check, re.IGNORECASE)) - 1
                
                print(f"relevant: {relevant} ")
                print(f"ambiguous: {ambiguous}\n")
                print(f"irrelevant: {irrelevant}\n")

                if relevant == 1:
                    inter_grade.append(1)
                elif ambiguous == 1:
                    inter_grade.append(0)
                else:
                    inter_grade.append(-1)
                        
            if inter_grade.count(1) / len(inter_grade) >= 0.15:
                grades.append(1)
            elif inter_grade.count(0) / len(inter_grade) >= 0.4:
                grades.append(0)
            else:
                grades.append(-1)
            
        return grades
    
if __name__ == "__main__":
    
    HF_TOKEN = "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"
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
    
    model = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
    )
    thres = Thresholder(model, StrOutputParser())
    print(thres.grade("what is pathway?", ["""
                                           Pathway is an open framework for high-throughput and low-latency real-time data processing. It is used to create Python code which seamlessly combines batch processing, streaming, and real-time API's for LLM apps. Pathway's distributed runtime (🦀-🐍) provides fresh results of your data pipelines whenever new inputs and requests are received.
In the first place, Pathway was designed to be a life-saver (or at least a time-saver) for Python developers and ML/AI engineers faced with live data sources, where you need to react quickly to fresh data. Still, Pathway is a powerful tool that can be used for a lot of things. If you want to do streaming in Python, build an AI data pipeline, or if you are looking for your next Python data processing framework, keep reading.
Pathway provides a high-level programming interface in Python for defining data transformations, aggregations, and other operations on data streams.

                                           """, 
                                           """
                                           Indo-Aryan began with Vedic Sanskrit (1500 BCE), the language of the Rigveda and
ancient Indian spiritual texts. Its transition into Classical Sanskrit introduced formal
grammar codified by Panini in works like the Ashtadhyayi, establishing a language
system of such precision that it resembles programming syntax. This linguistic structure
facilitated the development of philosophical, scientific, and literary works, seen in epics
like the Ramayana and Mahabharata, and created vocabulary layers with words that
convey meanings difficult to capture with single English equivalents. This layered quality
lends depth to Indo-Aryan languages, making their terms often evocative of multiple
concepts
                                           """, 
                                           """

                                           """]))