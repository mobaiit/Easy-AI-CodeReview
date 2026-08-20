import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from src.llm.client.base import BaseClient
from src.llm.types import NotGiven, NOT_GIVEN
from src.utils.log import logger
import requests


# DeepSeek 各模型的上下文窗口大小（tokens）
# 参考：https://api-docs.deepseek.com/quick_start/pricing
DEEPSEEK_CONTEXT_WINDOWS = {
    "deepseek-chat": 64_000,    # DeepSeek-V3
    "deepseek-reasoner": 64_000, # DeepSeek-R1
}
DEEPSEEK_DEFAULT_CONTEXT = 64_000


class DeepSeekClient(BaseClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set it in the environment variables.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) # DeepSeek supports OpenAI API SDK
        self.default_model = os.getenv("DEEPSEEK_API_MODEL", "deepseek-chat")

    def get_max_context_tokens(self) -> int:
        """返回当前模型的上下文窗口大小"""
        return DEEPSEEK_CONTEXT_WINDOWS.get(self.default_model, DEEPSEEK_DEFAULT_CONTEXT)

    def completions(self,
                    messages: List[Dict[str, str]],
                    model: Union[Optional[str], NotGiven] = NOT_GIVEN,
                    ) -> str:
        try:
            model = model or self.default_model
            logger.debug(f"Sending request to DeepSeek API. Model: {model}, Messages: {messages}")
            
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            if not completion or not completion.choices:
                logger.error("Empty response from DeepSeek API")
                return "AI服务返回为空，请稍后重试"
                
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            # 检查是否是认证错误
            if "401" in str(e):
                return "DeepSeek API认证失败，请检查API密钥是否正确"
            elif "404" in str(e):
                return "DeepSeek API接口未找到，请检查API地址是否正确"
            else:
                return f"调用DeepSeek API时出错: {str(e)}"
