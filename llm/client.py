from ollama import chat
from typing import Any

def call_llm(
        messages: list[dict], 
        model: str = "exaone3.5",
        temperature = 0.9,
        top_k = 50,
        top_p = 0.95,
        repetition_penalty = 1.1,
        num_predict = 256,
        format = None,
        output_format = None
        ) -> Any:
    responce = chat(
        model = model, 
        messages = messages,
        format = format,
        options={
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "num_predict": num_predict,
        }
    )
    content = responce.message.content
    if output_format:
        return output_format.model_validate_json(content)
    return content