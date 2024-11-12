# import os
# import pathway as pw
# from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
# from pathway.xpacks.llm.splitters import TokenCountSplitter
# from pathway.xpacks.llm.vector_store import VectorStoreServer


# pw.set_license_key("demo-license-key-with-telemetry")

# data_sources = []
# data_sources.append(
#     pw.io.fs.read(
#         "./documents",
#         format="binary",
#         mode="streaming",
#         with_metadata=True,
#     )
# )

# # for gdrive later on
# # data_sources.append(
# #     pw.io.gdrive.read(
# #         object_id="1234567890",
# #         service_user_credentials_file="credentials.json"
# #     )
# # )

# PATHWAY_PORT = 8765
# text_splitter = TokenCountSplitter()
# embedder = SentenceTransformerEmbedder(model="dunzhang/stella_en_1.5B_v5")

# vector_server = VectorStoreServer(
#     *data_sources,
#     embedder=embedder,
#     splitter=text_splitter,
# )

# vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT, threaded=False, with_cache=False)

from pathway.xpacks.llm.vector_store import VectorStoreServer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from db.embedders import ALL_MPNET
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pathway as pw

splitter = CharacterTextSplitter()

embedder = HuggingFaceEmbeddings(model_name="dunzhang/stella_en_1.5B_v5")
files = pw.io.fs.read("./documents/pathway_readme.md", format="binary", with_metadata=True)
server = VectorStoreServer.from_langchain_components(files, embedder=embedder, splitter=splitter)

HOST = "127.0.0.1"
PORT = 8666

if __name__ == "__main__":
    server.run_server(host=HOST, port=PORT)