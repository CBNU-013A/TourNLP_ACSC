# prompt_templete.py
# LLM에 입력될 프롬프트 작성

import json

# LLM을 통해 카테고리 분석
def category_extraction_prompt(reviews: list[str]) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광객이 관광지에 남긴 리뷰에서 관광지에서 나타나는 관광지의 특성(카테고리)을 추출하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "1. 카테고리는 명사형으로 최대한 간결하게 가급적 한 단어로 작성합니다. (예: 음식, 경관, 접근성 등)\n"
            "2. 의미가 겹치는 항목은 통합합니다.\n"
            "3. 설명 없이 JSON 형식으로만 출력하세요.\n\n"
            "출력 형식 예시:\n"
            "{\n"
            "  \"categories\": [\"경관\", \"음식\", \"접근성\"]\n"
            "}"
        )
    }
    joined = "\n".join(f"- {r}" for r in reviews)
    user_msg = {
        "role": "user",
        "content": f"다음은 사용자 리뷰입니다:\n{joined}\n\n카테고리:"
    }
    return [system_msg, user_msg]

def category_normalization_prompt(categories: list[str]) -> list[dict]:
    system_msg = {
        "role": "system",
        "content":(
            "다음은 리뷰에서 추출한 관광지의 특성을 나타내는 카테고리 후보입니다.\n"
            "다음 규칙을 따르세요:\n"
            "1. 의미가 유사하거나 중복되는 항목은 통합해서, 명확한 대표 카테고리만 남깁니다.\n"
            "2. 카테고리는 관광지에 적용할만한 범용적인 카테고리여야 합니다.\n"
            "3. 카테고리는 명사형으로 간결하게 가급적 한 단어로 작성합니다. (예: 음식, 경관, 접근성 등)\n"
            "4. 설명 없이 JSON 형식으로만 출력하세요.\n\n"
            "출력 형식 예시:\n"
            "{\n"
            "  \"categories\": [\"카테고리1\", \"카테고리2\", \"카테고리3\"]\n"
            "}"
        )
    }
    
    user_msg = {
        "role": "user",
        "content": json.dumps({"categories": categories}, ensure_ascii=False)
    }
    return [system_msg, user_msg]

# LLM을 통해 학습 데이터 라벨링

def generate_labeled_data_prompt(review: str, category_set: list[str]) -> list[dict]:
    system_msg = {
        "role": "system",
        "content": (
            "당신은 관광지 리뷰에 대해 주어진 카테고리들 중 해당하는 항목에 대해 감성을 분석하는 어시스턴트입니다.\n"
            "다음 규칙을 따르세요:\n"
            "1. 감성은 'pos', 'neg', 'neu', 'none' 중 하나로 작성합니다.\n"
            "2. 감성이 없는 카테고리는 'none'으로 작성합니다.\n"
            "3. 출력은 JSON 형식으로 작성하세요.\n"
            "4. 설명 없이 출력만 하세요.\n\n"
            "출력 형식 예시:\n"
            "{\n"
            "  \"labels\": {\n"
            "    \"경관\": \"pos\",\n"
            "    \"음식\": \"neg\",\n"
            "    \"가격\": \"none\"\n"
            "  }\n"
            "}"
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"리뷰: {review}\n"
            f"카테고리 목록: {', '.join(category_set)}"
        )
    }

    return [system_msg, user_msg]

def generate_synthetic_data_prompt(category_sentiment_pairs: list[tuple[str, str]]) -> list[dict]:
    '''
    카테고리-감정 쌍으로 가짜 리뷰 생성
    '''
    system_msg={}
    user_msg={}
    return [system_msg, user_msg]