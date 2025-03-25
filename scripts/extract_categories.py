import sys
from pathlib import Path
import pandas as pd
import json
import argparse

sys.path.append(str(Path(__file__).resolve().parent.parent))

from category.category_extracter import CategoryExtracter

def main():
    parser = argparse.ArgumentParser(description="Find Topics from Reviews CSV")
    parser.add_argument("csv_path", type=str, help="Full path to the CSV file")
    args = parser.parse_args()

    # 1. 데이터 로드
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)

    # 2. 리뷰 샘플링
    reviews = df["Review"].dropna().sample(30, random_state=42).tolist()

    # 3. 추출
    extractor = CategoryExtracter()
    categories = extractor.extract(reviews)

    # 4. 결과 저장
    output_path = Path("data/interim/extracted_categories.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"Saved to {output_path}, {len(categories)} categories found")

if __name__ == "__main__":
    main()