from pathlib import Path
from tqdm import tqdm
from dataset.dataset_generator import DatasetGenerator
from common.utils import RAW_DATA_DIR, INTERIM_DATASET_DIR, DATASET_PATH

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

    @classmethod
    def from_args(cls, args):
        runner = cls()
        if args.csv == "all":
            runner.run_all()
        elif args.csv == "merge":
            cls.merge_interim_files()
        elif args.csv:
            runner.run_on_csv(Path(args.csv))
        else:
            print("Error: CSV path required unless --all is specified.")

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