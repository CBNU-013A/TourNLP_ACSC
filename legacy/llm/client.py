# llm/client.py:
# llm 설정(ollama, openAI)

from ollama import chat
from typing import Any
import openai

def call_ollama(
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
        if output_format == str:
            return content
        return output_format.model_validate_json(content)
    return content

def call_openai(
        messages: list[dict],
        model: str = "gpt-3.5-turbo",
        temperature = 0.9,
        top_p = 0.95,
        max_tokens = 256,
        output_format = None
    ) -> Any:
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    content = response["choices"][0]["message"]["content"]
    if output_format:
        if output_format == str:
            return content
        return output_format.model_validate_json(content)
    return content