# /app/schemas/generate_schema.py

from pydantic import BaseModel
from typing import Literal, Optional, List
class CategorySetRequest(BaseModel):
    method: Literal["manual", "llm"]
    categories: Optional[List[str]] = None
    review_sample_path: Optional[str] = None    # for LLM

class CategorySetResponse(BaseModel):
    categories: List[str]