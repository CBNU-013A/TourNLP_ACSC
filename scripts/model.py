# scripts/model.py
# 모델 학습 / 평가

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.finetune_runner import ModelRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on labeled ACSC data")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., 'bert-base-uncased')"
    )
    parser.add_argument(
        "--train", type=str, required=True,
        help="Train Mode"
    )
    parser.add_argument(
        "--test", type=str, required=True,
        help="Test Mode"
    )

    args = parser.parse_args()
    ModelRunner.from_args(args)