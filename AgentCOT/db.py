import os
import pathway as pw
from pathway.xpacks.llm.embedders import GeminiEmbedder
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreServer

# This is what a peak coder would write
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDQPQr_pWALivoVPqIKC6TfHi4AsUBGMm0'
pw.set_license_key("demo-license-key-with-telemetry")

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./sample_documents",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

PATHWAY_PORT = 8765
text_splitter = TokenCountSplitter()
embedder = GeminiEmbedder(api_key=os.environ["GOOGLE_API_KEY"])

vector_server = VectorStoreServer(
    *data_sources,
    embedder=embedder,
    splitter=text_splitter,
)

vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT, threaded=False, with_cache=False)