from pathway.xpacks.llm.vector_store import VectorStoreClient


HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    client = VectorStoreClient(HOST, PORT, timeout=60)
    output = client.query("what is Mixture of experts?")
    print(len(output))
    print(output)