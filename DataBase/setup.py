from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pathway as pw
# from pathway.xpacks.llm.parsers import ParseUnstructured

splitter = CharacterTextSplitter(separator="\n")
embedder = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")

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

server = VectorStoreServer.from_langchain_components(
    fs_files, 
    embedder=embedder, 
    splitter=splitter
)

HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    server.run_server(host=HOST, port=PORT)