from pathway.xpacks.llm.vector_store import VectorStoreClient
PATHWAY_PORT = 8765

client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

docs = client("it typically refers to a set of algorithms or techniques used to solve a particular problem or complete a specific task")
print(docs)