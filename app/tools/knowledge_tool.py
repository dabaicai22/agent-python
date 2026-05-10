"""知识检索工具 - 从向量数据库中检索相关信息"""

import json
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.core.langfuse_client import langfuse_client
from app.services.vector_store_manager import vector_store_manager


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题
    
    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。
    
    Args:
        query: 用户的问题或查询
        
    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        
        # 从向量存储中检索相关文档
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": config.rag_top_k}
        )
        
        docs = retriever.invoke(query)
        
        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []
        
        # 格式化文档为上下文
        context = format_docs(docs)

        logger.info(f"检索到 {len(docs)} 个相关文档")

        # 记录检索详情到 Langfuse（用于评估召回效果）
        _log_retrieval_to_langfuse(query, docs)

        return context, docs
        
    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)


def _log_retrieval_to_langfuse(query: str, docs: List[Document]) -> None:
    """将检索结果记录到 Langfuse，用于评估召回效果"""
    client = langfuse_client.get_client()
    if not client:
        return

    try:
        trace = client.trace(
            name="retrieval",
            metadata={"query": query, "top_k": config.rag_top_k},
        )

        for i, doc in enumerate(docs):
            source = doc.metadata.get("_source", "unknown")
            file_name = doc.metadata.get("_file_name", "unknown")
            h1 = doc.metadata.get("h1", "")
            h2 = doc.metadata.get("h2", "")

            trace.span(
                name=f"chunk-{i}",
                input=query,
                output=doc.page_content,
                metadata={
                    "source": source,
                    "file_name": file_name,
                    "h1": h1,
                    "h2": h2,
                    "chunk_index": i,
                    "chunk_length": len(doc.page_content),
                },
            )

        trace.update()
    except Exception as e:
        logger.debug(f"Langfuse 检索记录失败（不影响主流程）: {e}")
