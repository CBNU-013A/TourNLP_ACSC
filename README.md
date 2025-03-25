# TourNLP_ACSC(개발중)
Aspect-based Sentiment Classification for Tourist Reviews

TourNLP_ACSC는 관광지 리뷰 데이터를 기반으로 관광객의 관심사(카테고리)와 그 감성을 추출하는 NLP 파이프라인입니다.
ollama를 통해, exaone3.5모델을 활용하여 학습 데이터를 생성, 학습하고, KoELECTRA 기반 모델로 ACSC태스크를 수행합니다.

## 주요 기능
- LLM 기반 Aspect Category 추출(개발중)
- 감성 정보를 포함한 학습데이터 자동 생성(LLM few-shot)(개발중)
- 기존 데이터에 새 카테고리 반영(개발중)
- KoELECTRA 기반 한국어 ACSC모델 학습(개발중)

## Requirement
- ollama
- 

## 실행 방법

### 1. 리뷰에서 카테고리 추출(개발중)
```bash
python scripts/extract_categories.py [csv 데이터 경로]
```
output은 data/interim/extracted_categories.json에 생성

### 2. 학습 데이터 생성(개발중)
- 새 카테고리를 반영한 새로운 데이터 생성
```bash
python scripts/generate_data.py
```
- 기존 학습 데이터에서 새 카테고리 반영
```bash
python scripts/update_categories.py
```

### 3. 모델 학습(개발중) 
```bash
python model/train.py
```