# scripts/generate_dataset.py
# 학습 데이터 생성

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from dataset.generator_runner import GeneratorRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled ACSC data from reviews")
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to input CSV file (raw review data) or 'all' to use all CSV files in data/raw"
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all interim files into a single dataset file"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Split the dataset into train, dev, and test"
    )
    args = parser.parse_args()
    GeneratorRunner.from_args(args)