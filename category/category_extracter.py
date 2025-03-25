from llm.prompt_templete import category_extraction_prompt
from llm.client import call_llm
from llm.schema import CategoryList

class CategoryExtracter:
    def __init__(self, model="exaone3.5"):
        self.model = model

    def extract(self, reviews: list[str],
                temperature: float = 0.9,
                top_k: int = 50,
                top_p: float = 0.95,
                repetition_penalty: float = 1.1,
                num_predict: int = 256) -> list[str]:
        messages = category_extraction_prompt(reviews)
        response = call_llm(
            messages,
            model=self.model,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_predict=num_predict,
            format=CategoryList.model_json_schema(),
            output_format=CategoryList
        )
        return response.categories
