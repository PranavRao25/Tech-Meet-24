import os
import lancedb
from sentence_transformers import SentenceTransformer
print("completed imports")
# Function to split text into chunks
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Step 1: Connect to LanceDB
db_path = "UnitTests/lancedb_folder"
db = lancedb.connect(db_path)

# Step 2: Read text files
folder_path = r"DataBase/documents"
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
print("done with chunking")
# Step 4: Generate embeddings for chunks
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model
texts = [entry['content'] for entry in data]
embeddings = model.encode(texts)

# Add embeddings to the data
for i, entry in enumerate(data):
    entry['vector'] = embeddings[i]
print(data)
# Step 5: Store in LanceDB
table_name = "chunked_text_files"
if table_name in db.table_names():
    table = db.open_table(table_name)
    table.add(data)
else:
    table = db.create_table(table_name, data=data)

print(f"Chunked data has been stored in LanceDB at: {db_path}")

import lancedb
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose an appropriate model

db_path = "UnitTests/lancedb_folder"
db = lancedb.connect(db_path)

table=db.open_table('chunked_text_files')