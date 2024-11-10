from pathway.xpacks.llm.vector_store import VectorStoreClient
PATHWAY_PORT = 8765

client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

docs = client("What is the history of artificial intelligence and its applications today?")