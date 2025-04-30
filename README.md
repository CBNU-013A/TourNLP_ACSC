# TourNLP_ACSC
> Aspect-based Sentiment Classification for Tourist Reviews

**목표**: 관광지 리뷰 데이터를 기반으로 관광객의 관심사(카테고리)와 그 감성을 분석


- ollama(LG의 exaone3.5모델)을 활용하여 학습 데이터를 생성, 학습 
- BERT 기반 모델로 ACSC Task 수행

## 주요 기능
- LLM 기반 Aspect Category 추출
- 감성 정보를 포함한 학습데이터 자동 생성(LLM few-shot)
- ~~기존 데이터에 새 카테고리 반영~~(개발예정)
- Huggingface를 통해 BERT 기반 한국어 ACSC Task Finetuning

## Requirement
- ollama model: exaone3.5
- requirements.txt 참고

## 실행 방법

### 1. 리뷰에서 카테고리 추출
- 데이터에서 추출
``` bash
python ./extract_categories.py --csv <Raw 데이터 경로>
```

- 디렉토리(data/raw)의 모든 데이터 추출
``` bash
python ./extract_categories.py -all
```
output은 data/interim/<데이터명>.extracted_categories.json에 생성
- 카테고리 목록 병합
```bash
python extract_categories.py --merge
```
ouput은 data/processed/category_set.json

### 2. 학습 데이터 생성
- 새 카테고리를 반영한 새로운 데이터 생성
```bash
python generate_dataset.py --csv [option]
```
- option
    - `--csv`:
        - `[경로]`: 단일 파일 처리
        - `all`: 디렉토리 내의 모든 파일 처리
        - `--neutral`: 중립 label 포함하여 처리
    - `--merge` : 모든 데이터 병합
    - `--split` : Train / Dev / Test로 분할
        - `--no-dev` : Train / Test로 분할
        - `--train_ratio`: Train / Test 비율 조정(default=0.8)
        - `--dev_ratio`: Train / dev 비율 조정(default=0.1)
        - `--seed`: 분할random seed(default=42)
### 3. 모델 학습 
```bash
python finetune.py --config_file [설정 파일 경로] --mode [option]
```
- mode
    - train: 학습
    - eval: 평가
    - test: 테스트

## 학습 결과
|model|F1:pos|F1:neg|F1:none|accuracy|
|--|--|--|--|--|
|[KoBERT](https://github.com/SKTBrain/KoBERT)|0.83|0.53|0.94|0.90|
|[KoELECTRA-base-v3](https://github.com/monologg/KoELECTRA)|0.85|0.59|0.94|0.90|
|[KoELECTRA-small-v3](https://github.com/monologg/KoELECTRA)|0.81|0.53|0.92|0.88|
|[KcELECTRA-base-v2022](https://github.com/Beomi/KcELECTRA)|0.85|0.59|0.94|0.91|

> [!NOTE]
> 학습데이터가 적어서 클래스 불균형 존재
> 모든 결과는 동일 config에서 진행