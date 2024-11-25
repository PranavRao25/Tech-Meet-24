from pathway.xpacks.llm.vector_store import VectorStoreServer
from pathway.stdlib.indexing import default_usearch_knn_document_index, default_vector_document_index, default_lsh_knn_document_index, default_brute_force_knn_document_index, default_full_text_document_index
from pathway.stdlib.indexing.data_index import _SCORE, DataIndex
from typing import Callable
import pathway as pw

INDEXES = {
    "usearch":default_usearch_knn_document_index,
    "vector_document":default_vector_document_index,
    "lsh":default_lsh_knn_document_index,
    "brute_force":default_brute_force_knn_document_index
}

def _give_build_fn(index_fn: Callable) -> Callable:
    def _build_graph(self) -> dict:
        """
        Builds the pathway computation graph for indexing documents and serving queries.
        """
        docs_s = self.docs
        if not docs_s:
            raise ValueError(
                """Please provide at least one data source, e.g. read files from disk:

    pw.io.fs.read('./sample_docs', format='binary', mode='static', with_metadata=True)
    """
            )
        if len(docs_s) == 1:
            (docs,) = docs_s
        else:
            docs: pw.Table = docs_s[0].concat_reindex(*docs_s[1:])  # type: ignore

        @pw.udf
        def parse_doc(data: bytes, metadata) -> list[pw.Json]:
            rets = self.parser(data)
            metadata = metadata.value
            return [dict(text=ret[0], metadata={**metadata, **ret[1]}) for ret in rets]  # type: ignore

        parsed_docs = docs.select(data=parse_doc(docs.data, docs._metadata)).flatten(
            pw.this.data
        )

        @pw.udf
        def post_proc_docs(data_json: pw.Json) -> pw.Json:
            data: dict = data_json.value  # type:ignore
            text = data["text"]
            metadata = data["metadata"]

            for processor in self.doc_post_processors:
                text, metadata = processor(text, metadata)

            return dict(text=text, metadata=metadata)  # type: ignore

        parsed_docs = parsed_docs.select(data=post_proc_docs(pw.this.data))

        @pw.udf
        def split_doc(data_json: pw.Json) -> list[pw.Json]:
            data: dict = data_json.value  # type:ignore
            text = data["text"]
            metadata = data["metadata"]

            rets = self.splitter(text)
            return [
                dict(text=ret[0], metadata={**metadata, **ret[1]})  # type:ignore
                for ret in rets
            ]

        chunked_docs = parsed_docs.select(data=split_doc(pw.this.data)).flatten(
            pw.this.data
        )

        chunked_docs += chunked_docs.select(text=pw.this.data["text"].as_str())

        knn_index = index_fn(
            chunked_docs.text,
            chunked_docs,
            dimensions=self.embedding_dimension,
            metadata_column=chunked_docs.data["metadata"],
            embedder=self.embedder,
        )

        parsed_docs += parsed_docs.select(
            modified=pw.this.data["metadata"]["modified_at"].as_int(),
            indexed=pw.this.data["metadata"]["seen_at"].as_int(),
            path=pw.this.data["metadata"]["path"].as_str(),
        )

        stats = parsed_docs.reduce(
            count=pw.reducers.count(),
            last_modified=pw.reducers.max(pw.this.modified),
            last_indexed=pw.reducers.max(pw.this.indexed),
            paths=pw.reducers.tuple(pw.this.path),
        )
        return locals()
    return _build_graph

class IndexServer:
    @staticmethod
    def from_langchain_components(*docs, embedder, splitter, parser, index=None) -> VectorStoreServer:
        if index == None:
            vector_store = VectorStoreServer.from_langchain_components(
                *docs,
                embedder=embedder, 
                splitter=splitter, 
                parser=parser,
                )
            return vector_store
        index_fn = INDEXES[index]
        VectorStoreServer._build_graph = _give_build_fn(index_fn)
        vector_store = VectorStoreServer.from_langchain_components(
            *docs, 
            embedder=embedder, 
            splitter=splitter, 
            parser=parser,
            )
        return vector_store