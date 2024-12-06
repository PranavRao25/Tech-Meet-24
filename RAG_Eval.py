import os

from RAG import RAG
from tabulate import tabulate
import json
import os

# RAG setup
vb = None
llm = None
rag = RAG(vb, llm)

os.environ['OPENAI_API_KEY'] = ''

with open("test50.json", 'r') as f:
	data = json.load(f)

list_qa = data['data']
questions = [qa['question'] for qa in list_qa]
answers = [qa['answer'] for qa in list_qa]
contexts = [qa['context'] for qa in list_qa]

results_df = rag.ragas_evaluate(questions, answers)

print(tabulate(results_df))
