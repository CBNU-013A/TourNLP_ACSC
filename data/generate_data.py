from ollama import chat
from pydantic import BaseModel

# ----- 데이터 구조 정의 -----
class AspectSentiment(BaseModel):
    category: str
    sentimnet: str

class Sentence(BaseModel):
    text: str
    category: list[AspectSentiment]

class SentenceResponse(BaseModel):
    data: list[Sentence]

# ----- 프롬프트 생성기 -----
class PromptGenerator:
    def generate(self, input_text: str) -> str:
        return input_text.strip()  # 향후 few-shot 포함 가능

# ----- LLM 클라이언트 -----
class LLMClient:
    def __init__(self, model: str = "exaone3.5", max_tokens: int = 4096):
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, prompt: str) -> SentenceResponse:
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a language model that **must respond only in Korean**. You must strictly follow the rules below:\n"
                    "1. Respond naturally and fluently in Korean based on the user's input.\n"
                    "2. Do not include any English words, romanizations, or translations in the response.\n"
                    "3. Do not say things like 'I will follow the rules' — just start answering directly.\n"
                    "4. If you break these rules, the session may be terminated immediately.\n"
                )},
                {"role": "user", "content": prompt}
            ],
            options={
                "num_predict": self.max_tokens,
                "temperature": 0.7
            },
            format=SentenceResponse.model_json_schema()
        )

        return SentenceResponse.model_validate_json(response["message"]["content"])

# ----- 데이터 생성기 -----
class DataGenerator:
    def __init__(self, llm_client: LLMClient, prompt_generator: PromptGenerator):
        self.llm_client = llm_client
        self.prompt_generator = prompt_generator

    def generate_dataset(self, raw_texts: list[str]) -> list[Sentence]:
        dataset = []
        for text in raw_texts:
            prompt = self.prompt_generator.generate(text)
            response = self.llm_client.chat(prompt)
            dataset.extend(response.data)
        return dataset

# ----- 실행 예시 -----
if __name__ == "__main__":
    raw_reviews = [
        "제주도에 갔는데 바다 풍경이 정말 아름다웠어요. 해산물 음식도 신선하고 맛있었습니다. 다만 숙소 위치가 좀 외진 곳이라 이동이 불편했어요."
    ]

    prompt_generator = PromptGenerator()
    llm_client = LLMClient()
    data_generator = DataGenerator(llm_client, prompt_generator)

    dataset = data_generator.generate_dataset(raw_reviews)

    for item in dataset:
        print(item)