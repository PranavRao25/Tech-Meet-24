from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# COLBERT = HuggingFaceEmbeddings(model_name="colbert-ir/colbertv2.0")
ALL_MPNET = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
COLBERT = HuggingFaceEmbeddings(model_name="colbert-ir/colbertv2.0")
BGEM3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

if __name__ == "__main__":
    print(COLBERT.embed_query("hello")[:4])
    # it throws warning like "No sentence-transformers model found with name colbert-ir/colbertv2.0. Creating a new one with mean pooling."
    # but it will still work and use colBERT only. so ignore this message if you see it
