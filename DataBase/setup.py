from pathway.xpacks.llm.vector_store import VectorStoreServer
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from io import BytesIO
from utils import IndexServer
import pathway as pw
import pymupdf

embedder = HuggingFaceEmbeddings(model_name="colbert-ir/colbertv2.0") # change to dunzhang/stella_en_1.5B_v5
splitter = CharacterTextSplitter(separator="\n")

class PDFParser(pw.UDF):
    def __wrapped__(self, contents: bytes) -> list[tuple[str, dict]]:
        try:
            docs: list[tuple[str, dict]] = [(contents.decode("utf-8"), {})]
            return docs
        except:
            pdfile = BytesIO(contents)
            doc = pymupdf.open(stream=pdfile, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
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

server = IndexServer.from_langchain_components(
    *documents, 
    embedder=embedder, 
    splitter=splitter, 
    parser=PDFParser(),
    index="brute_force"
    )

HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    server.run_server(host=HOST, port=PORT)