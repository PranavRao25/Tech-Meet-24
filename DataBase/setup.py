from pathway.xpacks.llm.vector_store import VectorStoreServer
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.text_splitter import TextSplitter
from io import BytesIO
# from utils import IndexServer
import pathway as pw
import pymupdf
import torch
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.embeddings import BaseEmbedding
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device":DEVICE}) # change to dunzhang/stella_en_1.5B_v5


class LangChainToLlamaIndexEmbedder(BaseEmbedding):
    langchain_embedder:HuggingFaceEmbeddings
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        """
        return self.langchain_embedder.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text/document.
        """
        return self.langchain_embedder.embed_documents([text])[0]
    
    def _aget_query_embedding(self, text: str) -> List[float]:
        return self.langchain_embedder.aembed_query(text)

embedder = LangChainToLlamaIndexEmbedder(langchain_embedder=embedder)


# splitter = CharacterTextSplitter(separator="\n")
# class CustomSplitter(TextSplitter):
#     def split_text(self, text: str):
#         """Split text into multiple components."""
#         words = text.split()
#         chunks = []
#         for i in range(0, len(words), self._chunk_size - self._chunk_overlap):
#             chunk = " ".join(words[i:i + self._chunk_size])
#             chunks.append(chunk)
#         return chunks
# splitter = CustomSplitter(chunk_size=100, chunk_overlap=20)

class PDFParser(pw.UDF):
    def __wrapped__(self, contents: bytes) -> list[tuple[str, dict]]:
        try:
            docs: list[tuple[str, dict]] = [(" ".join(contents.decode("utf-8").split()), {})]
            return docs
        except:
            pdfile = BytesIO(contents)
            doc = pymupdf.open(stream=pdfile, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            text = " ".join(text.split())
            docs: list[tuple[str, dict]] = [(text, {}) ]
            return docs

documents = []
fs_files = pw.io.fs.read(
    "./documents/", 
    format="binary", 
    with_metadata=True
)

# g_files = pw.io.gdrive.read(
#     object_id="1MqU_1lNODSPg22zI6IzL96O15UXMjEvj",
#     service_user_credentials_file="credentials.json",
#     with_metadata=True
# )

documents.append(fs_files)
# documents.append(g_files)
# server = VectorStoreServer.from_langchain_components(
#     *documents, 
#     embedder=embedder, 
#     splitter=splitter, 
#     parser=PDFParser(),
#     )

# server = IndexServer.from_langchain_components(
#     *documents, 
#     embedder=embedder, 
#     splitter=splitter, 
#     parser=PDFParser(),
#     index="lsh"
#     )

server = VectorStoreServer.from_llamaindex_components(
    *documents,
    transformations=[TokenTextSplitter(chunk_size=100, chunk_overlap=20, separator=" "), embedder],
    parser=PDFParser()
)

HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    server.run_server(host=HOST, port=PORT)

# observation: whenever we start the server and add a new document, the first fetch will take some time. So its better to give a high
# timout limit for the retriever client. Currently giving 60s