from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder, SentenceTransformer
import json
import os
import lancedb
from QueryAgent.ToTAgent import ToTAgent
from UnitTests.AlternateAgentTests import MistralParser
from UnitTests.Retriever import Agent


class ToTAgentTest:
    def __init__(self, db):
        #self.db = TextDatabase('tot_test')
        # assume this are chunks
        #self.db.upsert(data)
        self.db = db
        self._model = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens": 512}
        )
        self._reranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2", max_length=512, device="cpu")
        self._parser = RunnableLambda(MistralParser().invoke)
        self._tot_q = ToTAgent(self.db, (self._model, self._parser), self._reranker)

    def test(self, query):
        return self._tot_q.query(query)


if __name__ == '__main__':
    # Function to split text into chunks
    def chunk_text(text, chunk_size=200, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    # Step 1: Connect to LanceDB
    db_path = "./lancedb/tests"
    db = lancedb.connect(db_path)

    # Step 2: Read text files
    folder_path = r"./Data"
    files = [f for f in os.listdir(folder_path)]
    # Step 3: Prepare data with chunking
    data = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        # file_path = file
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
            chunks = chunk_text(text_content)  # Chunk the text
            for i, chunk in enumerate(chunks):
                data.append({"filename": file, "chunk_id": i, "content": chunk})

    # Step 4: Generate embeddings for chunks
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
    texts = [entry['content'] for entry in data]
    embeddings = model.encode(texts)

    # Add embeddings to the data
    for i, entry in enumerate(data):
        entry['vector'] = embeddings[i]

    # Step 5: Store in LanceDB
    table_name = "tot_test"
    if table_name in db.table_names():
        table = db.open_table(table_name)
        table.add(data)
    else:
        table = db.create_table(table_name, data=data)

    agent = Agent(table_name, model)
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

    tot_agent = ToTAgentTest(agent)

    with open('pathway.json', 'r') as f:
        qas = json.load(f)["qa"]

    questions = [q['question'] for q in qas]
    ground_truth = [q['answer'] for q in qas]
    answers = [tot_agent.test(q) for q in questions]

    gt_a = []
    for i in range(len(questions)):
        gt_a.append({
            "ground truth": ground_truth[i],
            "answers": answers[i]
        })
    output = {"output":gt_a}
    with open('tot_output.json', 'w') as f:
        json.dump(output, f)
