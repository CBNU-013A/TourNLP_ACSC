from pathlib import Path
from tqdm import tqdm
from dataset.dataset_generator import DatasetGenerator
from common.utils import RAW_DATA_DIR, INTERIM_DATASET_DIR, DATASET_PATH
from sklearn.model_selection import train_test_split
import json

class GeneratorRunner:
    def __init__(self, model="exaone3.5"):
        self.model = model
    
    def run_on_csv(self, csv_path: Path, position: int = 0):
        generator = DatasetGenerator(str(csv_path), model=self.model)
        generator.generate_labeled_data(position=position)

    def run_all(self):
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        for csv_path in tqdm(csv_files, desc="Processing CSV files"):
            tqdm.write(f"📄 Processing {csv_path.name}")
            self.run_on_csv(csv_path, position=1)
        self.merge_interim_files()


    @staticmethod
    def final_split(train_ratio=0.8, dev_ratio=0.1, seed=42):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        
        train_data, temp_data = train_test_split(
            data, test_size=(1 - train_ratio), random_state=seed
        )

        dev_size = dev_ratio / (1 - train_ratio)  # temp 중 dev가 차지하는 비율
        dev_data, test_data = train_test_split(
            temp_data, test_size=(1 - dev_size), random_state=seed
        )

        for name, split in zip(["train", "dev", "test"], [train_data, dev_data, test_data]):
            with open(f"data/processed/{name}.jsonl", "w", encoding="utf-8") as f:
                for item in split:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")

    @classmethod
    def from_args(cls, args):
        runner = cls()
        if args.csv == "all":
            runner.run_all()
        elif args.csv:
            runner.run_on_csv(Path(args.csv))
        else:
            print("Error: CSV path required unless --all is specified.")
        
        if args.merge:
            cls.merge_interim_files()
        if args.split:
            cls.final_split()
        

    @staticmethod
    def merge_interim_files():
        total = 0
        with open(DATASET_PATH, "w", encoding="utf-8") as out_file:
            for file in sorted(INTERIM_DATASET_DIR.glob("labeled_*.jsonl")):
                with open(file, "r", encoding="utf-8") as f:
                    for line in f:
                        out_file.write(line)
                        total += 1
        print(f"📦 Merged all interim files into {DATASET_PATH}, Total data num: {total}")
