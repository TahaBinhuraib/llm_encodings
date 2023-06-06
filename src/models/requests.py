from typing import List, Optional

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(default="Albert Einstein", description="Query to search for")
    top_k: int = Field(default=10, description="Number of results to return")
    filters: Optional[List[str]] = None


class SimilarityRequest(BaseModel):
    docs: List[str] = Field(
        default=[
            "Around 9 Million people live in London",
            "London is known for its financial district",
            "how many people do live in london?",
        ],
        description="List of documents to search in",
    )
    query: str = Field(
        default="how many people live in london?", description="Query to search for"
    )
