# prompt_templete.py
# LLM에 입력될 프롬프트 작성

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