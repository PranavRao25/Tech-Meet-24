import lancedb
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model

class Agent:
    def __init__(self, query=None):
        """
        You can query by Agent().query("question")
        or
        You can query by Agent("question")
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
        db_path = "UnitTests\lancedb_folder"
        self.db = lancedb.connect(db_path)
        self.table=self.db.open_table('chunked_text_files')
        if query is not None:
            return self.query(query)
    def query(self, query,top_k=5):
        query_embedding = self.model.encode([query])[0]
        results = self.table.search(query_embedding).limit(top_k).to_pandas()
        return list(results["content"])

# if __name__ == '__main__':
#     agent = Agent()
#     query = "What is pathway?"
#     print(agent.query(query,top_k=1))