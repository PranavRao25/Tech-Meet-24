from llama_index.retrievers.pathway import PathwayRetriever
from pathway.xpacks.llm.question_answering import RAGClient
import asyncio
from time import time

HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    client = PathwayRetriever(HOST, PORT, similarity_top_k=4)
    start = time()
    output = asyncio.run(client.aretrieve("what is pathway?"))
    print(len(output))
    print([item.text for item in output])
    end = time()
    print()
    print("time_taken:")
    print(end-start)