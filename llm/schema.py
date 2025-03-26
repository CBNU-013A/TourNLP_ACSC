from pydantic import BaseModel
from typing import Literal

class CategoryList(BaseModel):
    categories: list[str]

class ReviewLabel(BaseModel):
    category: str
    review: Literal["pos", "neg", "neu", "none"]

class ReviewSample(BaseModel):
    sentence: str
    label: list[ReviewLabel]

class SentimentList(BaseModel):
    sentiments: dict[str, Literal["pos", "neg", "neu", "none"]]