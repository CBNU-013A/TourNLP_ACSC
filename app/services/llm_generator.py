# app/services/llm_generator.py

import json
from pathlib import Path
from typing import List

class CategoryManager:
    def __init__(self, path: str = "data/categories.json"):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._categories = self._load()

    def _load(self) -> List[str]:
        if not Path(self.path).exists():
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f).get("categories", [])
        
    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"categories": self._categories}, f, ensure_ascii=False, indent=4)

    def get_all(self) -> List[str]:
        return self._categories
    
    def set_all(self, categories: List[str]) -> None:
        self._categories = categories
        self.save()

    def add(self, category: str) -> None:
        if category not in self._categories:
            self._categories.append(category)
            self.save()
    
    def remove(self, category: str) -> None:
        if category in self._categories:
            self._categories.remove(category)
            self.save()

    def clear(self) -> None:
        self._categories = []
        self.save()

category_manager = CategoryManager()