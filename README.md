# TourNLP_ACSC(개발중)
Aspect-based Sentiment Classification for Tourist Reviews

TourNLP_ACSC는 관광지 리뷰 데이터를 기반으로 관광객의 관심사(카테고리)와 그 감성을 추출하는 NLP 파이프라인입니다.
ollama를 통해, exaone3.5모델을 활용하여 학습 데이터를 생성, 학습하고, KoELECTRA 기반 모델로 ACSC태스크를 수행합니다.

## 주요 기능
- LLM 기반 Aspect Category 추출
    - 프롬포트 지속적으로 개선 예정
    - 추출한 카테고리들을 LLM을 사용하여 정규화
- 감성 정보를 포함한 학습데이터 자동 생성(LLM few-shot)(개발중)
- 기존 데이터에 새 카테고리 반영(개발중)
- KoELECTRA 기반 한국어 ACSC모델 학습(개발중)

## Requirement
- ollama
    - model: exaone3.5
- python >= 3.9

## 실행 방법

### 1. 리뷰에서 카테고리 추출
- 데이터에서 추출
```bash
python extract_categories.py --csv [Raw 데이터 경로]
```
output은 data/interim/[데이터명].extracted_categories.json에 생성
- 카테고리 목록 병합
```bash
python extract_categories.py --merge
```
ouput은 data/processed/category_set.json

### 2. 학습 데이터 생성
- 새 카테고리를 반영한 새로운 데이터 생성
```bash
python generate_dataset.py --csv [option] (--merge) (--split)
```
- option
    - `--csv`:
        - `[경로]`: 단일 파일 처리
        - `all`: 디렉토리 내의 모든 파일 처리
    - `--merge`: 처리한 모든 파일 병합
    - `--split`: train / dev / test 로 분할
- 기존 학습 데이터에서 새 카테고리 반영 *(개발중)*

### 3. 모델 학습 
```bash
python finetune.py --config_file [설정 파일 경로] --mode [option]
```
- mode
    - train: 학습
    - eval: 평가
    - test: 테스트