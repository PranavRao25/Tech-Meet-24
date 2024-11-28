import lancedb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model


class Agent:
    def __init__(self, table_name, model):
        self.model = model  # Choose an appropriate model
        db_path = "./lancedb/tests"
        self.db = lancedb.connect(db_path)
        self.table=self.db.open_table(table_name)

    def query(self, query,top_k=5):
        query_embedding = self.model.encode([query])[0]
        results = self.table.search(query_embedding).limit(top_k).to_pandas()
        return list(results["content"])
    