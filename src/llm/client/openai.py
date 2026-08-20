import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from src.llm.client.base import BaseClient
from src.llm.types import NotGiven, NOT_GIVEN
from src.utils.log import logger
import openai


# OpenAI 各模型的上下文窗口大小（tokens）
# 参考：https://platform.openai.com/docs/models
OPENAI_CONTEXT_WINDOWS = {
    "gpt-4o":          128_000,
    "gpt-4o-mini":     128_000,
    "gpt-4-turbo":     128_000,
    "gpt-4":             8_192,
    "gpt-3.5-turbo":  16_385,
}
OPENAI_DEFAULT_CONTEXT = 128_000


class OpenAIClient(BaseClient):
    """OpenAI client for chat models."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set it in the environment variables.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_model = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")

    def get_max_context_tokens(self) -> int:
        """返回当前模型的上下文窗口大小"""
        return OPENAI_CONTEXT_WINDOWS.get(self.default_model, OPENAI_DEFAULT_CONTEXT)

    def completions(self,
                    messages: List[Dict[str, str]],
                    model: Union[Optional[str], NotGiven] = NOT_GIVEN,
                    ) -> str:
        model = model or self.default_model
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content
