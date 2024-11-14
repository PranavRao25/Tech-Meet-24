from pathway.xpacks.llm.vector_store import VectorStoreClient


HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    client = VectorStoreClient(HOST, PORT)
    output = client.query("who is Larth?")
    print(len(output))
    print(output)