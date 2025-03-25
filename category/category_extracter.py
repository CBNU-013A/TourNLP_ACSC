from llm.prompt_templete import category_extraction_prompt, category_normalization_prompt
from llm.client import call_ollama as call_llm
from llm.schema import CategoryList
from pathlib import Path
import json

class CategoryExtracter:
    def __init__(self, model="exaone3.5"):
        self.model = model

    def extract(self, reviews: list[str],
                temperature: float = 0.9,
                top_k: int = 50,
                top_p: float = 0.95,
                repetition_penalty: float = 1.1,
                num_predict: int = 256) -> list[str]:
        messages = category_extraction_prompt(reviews)
        response = call_llm(
            messages,
            model=self.model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_predict=num_predict,
            format=CategoryList.model_json_schema(),
            output_format=CategoryList
        )
        return response.categories
    
    def normalize(self, categories: list[str],
                  temperature: float = 0.9,
                  top_k: int = 50,
                  top_p: float = 0.95,
                  repetition_penalty: float = 1.1,
                  num_predict: int = 256) -> list[str]:
        messages = category_normalization_prompt(categories)
        response = call_llm(
            messages,
            model=self.model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_predict=num_predict,
            format=CategoryList.model_json_schema(),
            output_format=CategoryList
        )
        return response.categories

def merge_all_categories():
    category_dir = Path("data/interim")
    all_files = category_dir.glob("*.categories.json")
    merged = set()

    for file in all_files:
        with open(file, encoding="utf-8") as f:
            merged.update(json.load(f))
    merged_list = sorted(merged)

    with open("data/processed/category_set.json", "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)
    print(f"✅ Merged {len(merged_list)} categories into data/processed/category_set.json")

