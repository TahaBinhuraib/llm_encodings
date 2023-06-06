from fastapi import FastAPI

from src import encoding, pinecone
from src.models.requests import RetrievalRequest, SimilarityRequest

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/similarity")
def similarity(request: SimilarityRequest):
    data = request.json()
    docs = data["docs"]
    query = data["query"]
    return encoding.encoder.similarity(docs, query)


@app.post("/retrieval")
def retrieve_pinecone(request: RetrievalRequest) -> dict:
    query = encoding.encoder.encode(query)
    return pinecone.retriever.retrieve_pinecone(query)
