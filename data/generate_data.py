import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from llm.client import call_ollama as call_llm
from llm.prompt_templete import generate_labeled_data_prompt, generate_synthetic_data_prompt
from llm.schema import ReviewSample, ReviewLabel, SentimentList

CATEGORY_PATH = Path("data/processed/category_set.json")
OUTPUT_PATH = Path("data/processed/labeled_reviews.jsonl")
RAW_DATA_PATH = Path("data/raw/")

class DatasetGenerator:
    def __init__(self, csv_path: str, model="exaone3.5"):
        self.model = model
        self.csv_path = csv_path
        self.reviews = self.load_csv()
        self.categories = self.load_categories()
        self.interim_path = Path(f"data/interim/dataset/labeled_{Path(self.csv_path).stem}.jsonl")

    def load_csv(self, review_column: str = "Review") -> list[str]:
        df = pd.read_csv(self.csv_path)
        return df[review_column].dropna().tolist()

    def load_categories(self) -> list[str]:
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
    
    @staticmethod
    def merge_interim_files():
        interim_dir = Path("data/interim/dataset")
        total = 0
        with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
            for file in sorted(interim_dir.glob("labeled_*.jsonl")):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        out_file.write(line)
                        total += 1
        print(f"📦 Merged all interim files into {OUTPUT_PATH}, Total data num: {total}")

    @staticmethod
    def process_csv_all_files(csv_files: list[Path]):
        for csv_file in tqdm(csv_files, desc="Processing CSV files"):
            tqdm.write(f"📄 Processing {csv_file}")
            generator = DatasetGenerator(str(csv_file))
            generator.generate_labeled_data(position=1)

    @classmethod
    def from_args(cls, args):
        if args.csv == "all":
            csv_files = list(RAW_DATA_PATH.glob("*.csv"))
            cls.process_csv_all_files(csv_files)
            cls.merge_interim_files()
        elif args.csv == "merge":
            cls.merge_interim_files()
        elif args.csv:
            generator = cls(args.csv)
            generator.generate_labeled_data()
        else:
            print("⚠️ Please provide --csv <path> or --csv all")
