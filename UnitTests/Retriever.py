import lancedb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model


class Agent:
<<<<<<< HEAD
    def __init__(self, table_name, model):
        self.model = model  # Choose an appropriate model
        db_path = "./lancedb/tests"
=======
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
        db_path = "UnitTests\lancedb_folder"
>>>>>>> ee01989d6218374b37fc77d492de3f58c3cfb172
        self.db = lancedb.connect(db_path)
        self.table=self.db.open_table(table_name)

    def query(self, query,top_k=5):
        query_embedding = self.model.encode([query])[0]
        results = self.table.search(query_embedding).limit(top_k).to_pandas()
        return list(results["content"])

# if __name__ == '__main__':
#     agent = Agent()
#     query = "What is pathway?"
#     print(agent.query(query,top_k=1))