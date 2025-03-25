from pydantic import BaseModel

class CategoryList(BaseModel):
    categories: list[str]