# from llama_index.retrievers.pathway import PathwayRetriever
from pathway.xpacks.llm.vector_store import VectorStoreClient # better to use pathway's default client as we can set a timeout limit on it
from time import time

HOST = "127.0.0.1"
PORT = 8666
if __name__ == "__main__":
    begin = time()
    client = VectorStoreClient(HOST, PORT, timeout=500)
    start = time()
    print("vector client initialized:",start-begin)
    query = """Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract"""
    output = client.query(query, k=20)
    print(len(output))
    print("\n====\n".join([item["text"] for item in output]))
    end = time()
    print()
    print("time_taken for querying:")
    print(end-start)