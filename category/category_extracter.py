from llm.prompt_templete import category_extraction_prompt, category_normalization_prompt
from llm.client import call_ollama as call_llm
from llm.schema import CategoryList
from pathlib import Path
import json

class CategoryExtracter:
    def __init__(self, model="exaone3.5",
                 temperature: float = 0.9,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1,
                 num_predict: int = 256):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.num_predict = num_predict

    def _call_llm(self, messages) -> CategoryList:
        return call_llm(
            messages,
            model=self.model,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            num_predict=self.num_predict,
            format=CategoryList.model_json_schema(),
            output_format=CategoryList
        )
    def extract(self, reviews: list[str]) -> list[str]:
        messages = category_extraction_prompt(reviews)
        response = self._call_llm(messages)
        return response.categories
    
    def normalize(self, categories: list[str]) -> list[str]:
        messages = category_normalization_prompt(categories)
        response = self._call_llm(messages)
        return response.categories
    

def merge_all_categories():
    category_dir = Path("data/interim")
    all_files = category_dir.glob("*.categories.json")
    merged = set()

    for file in all_files:
        with open(file, encoding="utf-8") as f:
            merged.update(json.load(f))
    merged_list = sorted(merged)

    extractor = CategoryExtracter()
    merged_list = extractor.normalize(categories=merged_list)

    with open("data/processed/category_set.json", "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Merged {len(merged_list)} categories into data/processed/category_set.json")