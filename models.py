# models.py

from pydantic import BaseModel
from typing import List

class SearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 5  # default number of similar results to return

class SearchResult(BaseModel):
    score: float
    image_path: str
    vector: List[float]
