"""Elasticsearch 客户端工厂模块"""

from elasticsearch import Elasticsearch
from loguru import logger

from app.config import config


class ElasticsearchClientManager:
    """Elasticsearch 客户端管理器"""

    INDEX_NAME: str = "parent_docs"
    INDEX_MAPPING: dict = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "ik_max_word"},
                "content": {"type": "text", "analyzer": "ik_max_word"},
                "file_path": {"type": "keyword"},
                "level": {"type": "keyword"},
                "alarm_type": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "created_at": {"type": "date"},
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "ik_max_word": {
                        "type": "ik_max_word"
                    }
                }
            }
        }
    }

    def __init__(self) -> None:
        """初始化 Elasticsearch 客户端管理器"""
        self._client: Elasticsearch | None = None

    def connect(self) -> Elasticsearch:
        """
        连接到 Elasticsearch 服务器并初始化 index

        Returns:
            Elasticsearch: ES 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        if self._client is not None:
            logger.debug("Elasticsearch 已连接，跳过重复 connect")
            return self._client

        try:
            logger.info(f"正在连接到 Elasticsearch: {config.elasticsearch_host}:{config.elasticsearch_port}")

            # 创建客户端（无认证模式）
            self._client = Elasticsearch(
                hosts=[f"http://{config.elasticsearch_host}:{config.elasticsearch_port}"],
                request_timeout=30,
            )

            # 检查连接
            if not self._client.ping():
                raise RuntimeError("Elasticsearch 连接失败")

            logger.info("成功连接到 Elasticsearch")

            # 检查并创建 index
            if not self._index_exists():
                logger.info(f"index '{self.INDEX_NAME}' 不存在，正在创建...")
                self._create_index()
                logger.info(f"成功创建 index '{self.INDEX_NAME}'")
            else:
                logger.info(f"index '{self.INDEX_NAME}' 已存在")

            return self._client

        except Exception as e:
            logger.error(f"连接 Elasticsearch 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Elasticsearch 失败: {e}") from e

    def _index_exists(self) -> bool:
        """检查 index 是否存在"""
        return self._client.indices.exists(index=self.INDEX_NAME)

    def _create_index(self) -> None:
        """创建 parent_docs index"""
        # 先尝试创建 IK 分词器（可能不支持，跳过即可）
        try:
            self._client.indices.create(index=self.INDEX_NAME, body=self.INDEX_MAPPING)
        except Exception as e:
            # 如果 IK 分词器不存在，使用默认 mapping
            logger.warning(f"创建 index 使用 IK 分词器失败: {e}，使用默认配置")
            default_mapping = {
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "title": {"type": "text"},
                        "content": {"type": "text"},
                        "file_path": {"type": "keyword"},
                        "level": {"type": "keyword"},
                        "alarm_type": {"type": "keyword"},
                        "metadata": {"type": "object", "enabled": True},
                        "created_at": {"type": "date"},
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                }
            }
            self._client.indices.create(index=self.INDEX_NAME, body=default_mapping)

    def get_client(self) -> Elasticsearch:
        """
        获取客户端实例

        Returns:
            Elasticsearch: ES 客户端实例

        Raises:
            RuntimeError: 客户端未初始化时抛出
        """
        if self._client is None:
            raise RuntimeError("Elasticsearch 客户端未初始化，请先调用 connect()")
        return self._client

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False
            return self._client.ping()
        except Exception as e:
            logger.error(f"Elasticsearch 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("已关闭 Elasticsearch 连接")


# 全局单例
es_manager = ElasticsearchClientManager()