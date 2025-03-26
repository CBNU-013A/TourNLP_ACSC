# scripts/generate_dataset.py
# 학습 데이터 생성

# scripts/generate_dataset.py
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.generate_data import DatasetGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate labeled ACSC data from reviews")
    parser.add_argument(
        "--csv", type=str, required=True,
        help="Path to input CSV file (raw review data) or 'all' to use all CSV files in data/raw"
    )
    args = parser.parse_args()
    DatasetGenerator.from_args(args)