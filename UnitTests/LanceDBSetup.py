from typing import List
import lancedb
import numpy as np

class TextDatabase:
    def __init__(self, table_name):
        self.db = lancedb.connect('lancedb/test')
        self.table_name = table_name

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
        if isinstance(request_vector, np.ndarray):
            return self.tbl.search(request_vector).limit(top_k).to_list()
        else:
            raise ValueError("Query must be a numpy array matching vector dimensions")

    def delete(self):
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)

    def is_empty(self) -> bool:
        return self.tbl.count_rows() == 0


# Usage Example
db = lancedb.connect('lancedb/test')
table_name = 'demo'

# Create a TextDatabase instance
text_db = TextDatabase(table_name)

# Define the data
data = [
    {"vector": np.array(v), "metadata": f"vec{i}"}
    for i, v in enumerate([[1, 2, 3], [2, 3, 4], [4, 5, 6]])
]

# Add data to the table
text_db.upsert(data)

# Query the table
query_vector = np.array([1, 2, 3])
response = text_db.query(query_vector, top_k=2)
print("Query Response:", response)

# Check if the table is empty
print("Is Table Empty?", text_db.is_empty())

# Drop the table
text_db.delete()
