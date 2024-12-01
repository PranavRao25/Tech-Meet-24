from typing import List
import numpy as np
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder, SentenceTransformer
import json
import os
import lancedb
from QueryAgent.ToTAgent import ToTAgent
# from LanceDBSetup import TextDatabase
from UnitTests.AlternateAgentTests import MistralParser

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

class TextDatabase:
    def __init__(self, table_name, embedding_model):
        self.db = lancedb.connect('lancedb/test')
        self.table_name = table_name
        self.embedding_model = embedding_model

        self.is_created = True
        try:
            self.tbl = self.db.open_table(self.table_name)
        except:
            self.is_created = False

    def upsert(self, data):
        if isinstance(data, List) and all(isinstance(i, dict) for i in data):
            if self.is_created:
                self.tbl.add(data)
            else:
                self.db.create_table(self.table_name, data)
                self.tbl = self.db.open_table(self.table_name)
                self.tbl.create_fts_index("content")
                self.is_created = True
        else:
            raise ValueError("Incorrect Data Format: Expected List[dict]")

    def query(self, request_vector, top_k=3) -> List[dict]:
        print(self.embedding_model.encode([request_vector])[0])
        q_vector = {"vector": self.embedding_model.encode([request_vector])[0], "content": request_vector}
        # if isinstance(q_vector, np.ndarray):
        return self.tbl.search(q_vector).limit(top_k).to_list()
        # else:
        #     raise ValueError("Query must be a numpy array matching vector dimensions")

    def delete(self):
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)

    def is_empty(self) -> bool:
        return self.tbl.count_rows() == 0

# class ToTAgentTest:
#     def __init__(self, db):

#         # assume this are chunks
#         #self.db.upsert(data)
#         self.db = db

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
db = TextDatabase('tot_test', embedding_model)
with open('data.json', 'r') as f:
    data = json.load(f)
db.upsert(data)

model = HuggingFaceHub(
repo_id="mistralai/Mistral-7B-Instruct-v0.3",
model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
)

reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")
parser = RunnableLambda(MistralParser().invoke)
tot_q = ToTAgent(db, (model, parser), reranker)

#     def test(self, query):
#         return self._tot_q.query(query)
#
#
# if __name__ == '__main__':
#     # Function to split text into chunks
#     def chunk_text(text, chunk_size=200, overlap=50):
#         words = text.split()
#         chunks = []
#         for i in range(0, len(words), chunk_size - overlap):
#             chunk = " ".join(words[i:i + chunk_size])
#             chunks.append(chunk)
#         return chunks
#
#     # Step 1: Connect to LanceDB
#     db_path = "./lancedb/tests"
#     db = lancedb.connect(db_path)
#
#     # Step 2: Read text files
#     folder_path = r"./Data"
#     files = [f for f in os.listdir(folder_path)]
#     # Step 3: Prepare data with chunking
#     data = []
#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         # file_path = file
#         with open(file_path, 'r', encoding='utf-8') as f:
#             text_content = f.read()
#             chunks = chunk_text(text_content)  # Chunk the text
#             for i, chunk in enumerate(chunks):
#                 data.append({"filename": file, "chunk_id": i, "content": chunk})
#
#     # Step 4: Generate embeddings for chunks
#     model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
#     texts = [entry['content'] for entry in data]
#     embeddings = model.encode(texts)
#
#     # Add embeddings to the data
#     for i, entry in enumerate(data):
#         entry['vector'] = embeddings[i]
#
#     # Step 5: Store in LanceDB
#     table_name = "tot_test"
#     if table_name in db.table_names():
#         table = db.open_table(table_name)
#         table.add(data)
#     else:
#         table = db.create_table(table_name, data=data)
#
#     agent = Agent(table_name, model)
#
#     tot_agent = ToTAgentTest(agent)
#

with open('pathway.json', 'r') as f:
    qas = json.load(f)["qa"]

questions = [q['question'] for q in qas]
ground_truth = [q['answer'] for q in qas]
answers = [tot_q.query(q) for q in questions]

gt_a = []
for i in range(len(questions)):
    gt_a.append({
        "ground truth": ground_truth[i],
        "answers": answers[i]
    })
output = {"output":gt_a}
with open('tot_output.json', 'w') as f:
    json.dump(output, f)
