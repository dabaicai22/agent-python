"""Elasticsearch 服务模块 - 父文档操作"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.elasticsearch_client import es_manager


class ElasticsearchService:
    """Elasticsearch 服务 - 负责父文档的存储和查询"""

    def __init__(self):
        """初始化 ES 服务"""
        self._client = None

    def _ensure_client(self):
        """确保客户端已连接"""
        if self._client is None:
            self._client = es_manager.connect()

    def index_parent_doc(self, doc_id: str, title: str, content: str, file_path: str,
                         level: str = None, alarm_type: str = None, metadata: Dict = None) -> bool:
        """
        索引父文档到 Elasticsearch

        Args:
            doc_id: 文档唯一ID
            title: 文档标题
            content: 文档完整内容
            file_path: 源文件路径
            level: 告警级别（可选）
            alarm_type: 告警类型（可选）
            metadata: 其他元数据（可选）

        Returns:
            bool: 是否成功
        """
        try:
            self._ensure_client()

            doc_body = {
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "file_path": file_path,
                "level": level,
                "alarm_type": alarm_type,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }

            self._client.index(
                index=es_manager.INDEX_NAME,
                id=doc_id,
                body=doc_body,
                refresh=True,
            )

            logger.info(f"父文档索引成功: doc_id={doc_id}")
            return True

        except Exception as e:
            logger.error(f"索引父文档失败: doc_id={doc_id}, error={e}")
            return False

    def get_parent_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 doc_id 获取父文档

        Args:
            doc_id: 文档唯一ID

        Returns:
            Optional[Dict]: 文档内容，不存在则返回 None
        """
        try:
            self._ensure_client()

            result = self._client.get(index=es_manager.INDEX_NAME, id=doc_id)
            return result["_source"] if result else None

        except Exception as e:
            logger.error(f"获取父文档失败: doc_id={doc_id}, error={e}")
            return None

    def get_parent_docs_by_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """
        批量获取父文档

        Args:
            doc_ids: doc_id 列表

        Returns:
            List[Dict]: 文档列表
        """
        if not doc_ids:
            return []

        try:
            self._ensure_client()

            result = self._client.mget(
                index=es_manager.INDEX_NAME,
                body={"ids": doc_ids},
            )

            docs = []
            for doc in result.get("docs", []):
                if doc.get("found"):
                    docs.append(doc["_source"])

            return docs

        except Exception as e:
            logger.error(f"批量获取父文档失败: doc_ids={doc_ids}, error={e}")
            return []

    def delete_parent_doc(self, doc_id: str) -> bool:
        """
        删除父文档

        Args:
            doc_id: 文档唯一ID

        Returns:
            bool: 是否成功
        """
        try:
            self._ensure_client()

            self._client.delete(
                index=es_manager.INDEX_NAME,
                id=doc_id,
                refresh=True,
            )

            logger.info(f"父文档删除成功: doc_id={doc_id}")
            return True

        except Exception as e:
            logger.error(f"删除父文档失败: doc_id={doc_id}, error={e}")
            return False

    def delete_by_file_path(self, file_path: str) -> int:
        """
        删除指定文件路径的所有父文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        try:
            self._ensure_client()

            result = self._client.delete_by_query(
                index=es_manager.INDEX_NAME,
                body={
                    "query": {
                        "term": {"file_path": file_path}
                    }
                },
                refresh=True,
            )

            deleted_count = result.get("deleted", 0)
            logger.info(f"删除文件相关父文档: file_path={file_path}, count={deleted_count}")
            return deleted_count

        except Exception as e:
            logger.error(f"删除文件相关父文档失败: file_path={file_path}, error={e}")
            return 0

    def search_parent_docs(self, query: str, size: int = 10) -> List[Dict[str, Any]]:
        """
        搜索父文档

        Args:
            query: 搜索关键词
            size: 返回数量

        Returns:
            List[Dict]: 匹配的文档列表
        """
        try:
            self._ensure_client()

            result = self._client.search(
                index=es_manager.INDEX_NAME,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["title", "content"],
                        }
                    },
                    "size": size,
                },
            )

            hits = result.get("hits", {}).get("hits", [])
            return [hit["_source"] for hit in hits]

        except Exception as e:
            logger.error(f"搜索父文档失败: query={query}, error={e}")
            return []


# 全局单例
elasticsearch_service = ElasticsearchService()