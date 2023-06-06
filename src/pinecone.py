import os

import pinecone
from src.encoding import encoder


class Retriever:
    """Retriever class for Pinecone index"""

    def __init__(self, index_name):
        self.index_name = pinecone.Index(index_name)
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        # check if index exists
        if self.index_name not in pinecone.list_indexes():
            raise ValueError(f"Index {index_name} does not exist")
        self.index = pinecone.Index(self.index_name)
        assert (
            self.index.describe_index_stats()["dimensions"] == 768
        ), "Index dimensions must be 768"

    def retrieve_pinecone(self, query, top_k=10) -> dict:
        query = encoder.encode(query)
        return self.index.query(query, top_k=top_k)


try:
    retriever
except Exception:
    print("loading retrieval model...")
    retriever = Retriever("test_index")
