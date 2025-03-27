import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from llm.client import call_ollama as call_llm
from llm.prompt_templete import generate_labeled_data_prompt, generate_synthetic_data_prompt
from common.schema import ReviewSample, ReviewLabel, SentimentList
from common.utils import CATEGORY_PATH, INTERIM_DATASET_DIR

class DatasetGenerator:
    def __init__(self, csv_path: str, model="exaone3.5"):
        self.model = model
        self.csv_path = csv_path
        self.reviews = self._load_csv()
        self.categories = self._load_categories()
        self.interim_path = INTERIM_DATASET_DIR / f"labeled_{Path(self.csv_path).stem}.jsonl"

    def _load_csv(self, review_column: str = "Review") -> list[str]:
        df = pd.read_csv(self.csv_path)
        return df[review_column].dropna().tolist()

    def _load_categories(self) -> list[str]:
        with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse_labels(self, response: SentimentList) -> list[ReviewLabel]:
        try:
            return [
                ReviewLabel(category=cat, review=sentiment)
                for cat, sentiment in response.sentiments.items()
            ]
        except Exception as e:
            print(f"⚠️ Failed to parse response: {response}")
            return []    
    
    def generate_synthetic_data(self):
        pass

    # --csv, Main function
    def generate_labeled_data(self, position: int = 0):
        labeled_data: list[ReviewSample] = []
        for review in tqdm(self.reviews, desc="Annotating reviews", position=position, leave=False):
            prompt = generate_labeled_data_prompt(review, self.categories)
            response = call_llm(
                messages=prompt,
                model = self.model,
                format=SentimentList.model_json_schema(),
                output_format=SentimentList
                )
            labels = self.parse_labels(response)
            sample = ReviewSample(sentence=review, label=labels)
            labeled_data.append(sample)

        self.interim_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.interim_path, "w", encoding="utf-8") as f:
            for sample in labeled_data:
                f.write(sample.model_dump_json() + "\n")
