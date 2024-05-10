import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import (
    QueryBundle,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.milvus import MilvusVectorStore

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
load_dotenv()


def setup_settings(model_name: str, embedding_model: str) -> None:
    Settings.embed_model = OllamaEmbedding(model_name=embedding_model)
    Settings.llm = Ollama(model=model_name, request_timeout=360.0, temperature=0.9)
    Settings.chunk_size = 512


def print_retrieved_nodes(retrieved_nodes: list[NodeWithScore]) -> None:
    header = "=" * 50 + f"[{retrieved_nodes[0].score}]" + "=" * 50
    print(header)
    print(retrieved_nodes[0].node.get_content())
    print("=" * len(header))


def initialize_vector_store():
    connection_info = {
        "uri": os.getenv("VECTOR_STORAGE_URI"),
        "token": os.getenv("VECTOR_STORAGE_TOKEN"),
        "user": os.getenv("VECTOR_STORAGE_USER"),
        "password": os.getenv("VECTOR_STORAGE_PASSWORD"),
    }

    return MilvusVectorStore(dim=768, overwrite=True, **connection_info)


def main() -> None:
    setup_settings("llama3", "nomic-embed-text")
    documents = SimpleDirectoryReader("data").load_data(show_progress=True)
    vector_store = initialize_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5
    )
    retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

    while True:
        query = input("*** Ask me anything: ")

        if query == "exit":
            break

        retrieved_nodes = retriever.retrieve(QueryBundle(query))
        reranked_nodes = reranker.postprocess_nodes(retrieved_nodes, QueryBundle(query))
        print_retrieved_nodes(reranked_nodes)

        query_engine = index.as_query_engine(
            streaming=True, node_postprocessors=[reranker]
        )
        streaming_response = query_engine.query(query)
        print("_" * 50)
        streaming_response.print_response_stream()
        print("\n", "_" * 50)


if __name__ == "__main__":
    main()
