# from llama_index.retrievers.pathway import PathwayRetriever
from pathway.xpacks.llm.vector_store import VectorStoreClient # better to use pathway's default client as we can set a timeout limit on it
from time import time

HOST = "127.0.0.1"
PORT = 8666
if __name__ == "__main__":
    client = VectorStoreClient(HOST, PORT, timeout=60)
    start = time()
    output = client.query("what is Random Access Memory?")
    print(len(output))
    print([item["text"] for item in output])
    end = time()
    print()
    print("time_taken:")
    print(end-start)