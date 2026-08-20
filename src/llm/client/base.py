from abc import abstractmethod
from typing import List, Dict, Optional, Union

from src.llm.types import NotGiven, NOT_GIVEN
from src.utils.log import logger


class BaseClient:
    """ Base class for chat models client. """

    def get_max_context_tokens(self) -> int:
        """
        返回当前模型的上下文窗口大小（tokens）。
        子类应根据实际模型覆盖此方法。
        此处返回保守的兜底值，适用于未知的自定义模型/代理。
        """
        return 32_000

    def ping(self) -> bool:
        """Ping the model to check connectivity."""
        try:
            result = self.completions(messages=[{"role": "user", "content": '请仅返回 "ok"。'}])
            return result and result == 'ok'
        except Exception:
            logger.error("尝试连接LLM失败， {e}")
            return False

    @abstractmethod
    def completions(self,
                    messages: List[Dict[str, str]],
                    model: Union[Optional[str], NotGiven] = NOT_GIVEN,
                    ) -> str:
        """Chat with the model.
        """
