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
        "--csv", type=str,
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
    parser.add_argument("--no-dev", action="store_true", help="Omit dev split when splitting the dataset")
    parser.add_argument("--neutral", action="store_true", help="Include neutral label in the dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of data used for training")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Proportion of data used for dev (if included)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()
    GeneratorRunner.from_args(args)
    if args.split:
        GeneratorRunner.final_split(
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            include_dev=not args.no_dev,
            seed=args.seed
        )