"""Langfuse 客户端工厂

提供 Langfuse 可观测性平台的客户端和 LangChain CallbackHandler。
未配置密钥时自动降级，不影响主流程。
"""

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from loguru import logger

from app.config import config


class LangfuseClient:
    """Langfuse 客户端管理器"""

    def __init__(self):
        self._client = None
        self._handler = None
        self._enabled = bool(config.langfuse_public_key and config.langfuse_secret_key)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_client(self) -> Langfuse | None:
        """获取 Langfuse 原生客户端（用于手动创建 trace/span）"""
        if not self._enabled:
            return None
        if self._client is None:
            try:
                self._client = Langfuse(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    host=config.langfuse_host,
                )
                logger.info("Langfuse 客户端初始化成功")
            except Exception as e:
                logger.warning(f"Langfuse 客户端初始化失败: {e}")
                self._enabled = False
        return self._client

    def get_callback_handler(self) -> CallbackHandler | None:
        """获取 LangChain CallbackHandler（用于自动追踪 LLM/Tool/Chain）"""
        if not self._enabled:
            return None
        if self._handler is None:
            try:
                self._handler = CallbackHandler(
                    public_key=config.langfuse_public_key,
                    secret_key=config.langfuse_secret_key,
                    host=config.langfuse_host,
                )
                logger.info("Langfuse CallbackHandler 初始化成功")
            except Exception as e:
                logger.warning(f"Langfuse CallbackHandler 初始化失败: {e}")
                self._enabled = False
        return self._handler


# 全局单例
langfuse_client = LangfuseClient()
