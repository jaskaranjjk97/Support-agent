from langchain_openai import ChatOpenAI

from app import config


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.OPENAI_MODEL,
        temperature=temperature,
        api_key=config.OPENAI_API_KEY,
    )
