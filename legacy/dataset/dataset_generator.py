import json
import pandas as pd
from pathlib import Path
from alive_progress import alive_bar
import re

from llm.client import call_ollama as call_llm
from llm.prompt_templete import generate_labeled_data_prompt, generate_synthetic_data_prompt
from common.schema import ReviewSample, ReviewLabel, SentimentList, SentimentList_neu, ReviewLabel_neu, ReviewSample_neu
from common.utils import CATEGORY_PATH, INTERIM_DATASET_DIR

def extract_json(text: str) -> str:
    """
    Extracts the first JSON object found in a text string.
    If extraction fails, returns an empty JSON object "{}".
    """
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return match.group(0)
    except Exception as e:
        print(f"⚠️ JSON extraction failed: {e}")
    return "{}"

class DatasetGenerator:
    def __init__(self, csv_path: str, model="exaone3.5", neutral: bool = False):
        self.model = model
        self.csv_path = csv_path
        self.reviews = self._load_csv()
        self.categories = self._load_categories()
        self.interim_path = INTERIM_DATASET_DIR / f"labeled_{Path(self.csv_path).stem}.jsonl"
        self.neutral = neutral

    def _load_csv(self, review_column: str = "Review") -> list[str]:
        df = pd.read_csv(self.csv_path)
        return df[review_column].dropna().tolist()

    def _load_categories(self) -> list[str]:
        with open(CATEGORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def parse_labels(self, response):
        try:
            if self.neutral:
                assert isinstance(response, SentimentList_neu), "Response type mismatch"
                return [
                    ReviewLabel_neu(category=cat, review=sentiment)
                    for cat, sentiment in response.sentiments.items()
                ]
            else:
                assert isinstance(response, SentimentList), "Response type mismatch"
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
        if self.neutral:
            labeled_data: list[ReviewSample_neu] = []
        else:
            labeled_data: list[ReviewSample] = []

        with alive_bar(len(self.reviews),
                        title="Annotating reviews",
                        bar="filling",
                        spinner="ball_belt",
                        ) as bar:
            for review in self.reviews:
                prompt = generate_labeled_data_prompt(review, self.categories)
                raw_response = call_llm(
                    messages=prompt,
                    model=self.model,
                    format=SentimentList.model_json_schema() if not self.neutral else SentimentList_neu.model_json_schema(),
                    output_format=str,  # <- receive raw text first
                )
                clean_response = extract_json(raw_response)
                try:
                    response_obj = (SentimentList if not self.neutral else SentimentList_neu).model_validate_json(clean_response)
                    labels = self.parse_labels(response_obj)
                except Exception as e:
                    print(f"⚠️ Failed to validate response: {e}\nRaw LLM response:\n{raw_response}")
                    labels = []
                sample = ReviewSample(sentence=review, label=labels) if not self.neutral else ReviewSample_neu(sentence=review, label=labels)
                labeled_data.append(sample)
                bar()

        self.interim_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.interim_path, "w", encoding="utf-8") as f:
            for sample in labeled_data:
                f.write(sample.model_dump_json() + "\n")
