"""文档分割服务模块 - 基于 LangChain 的智能文档分割"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, TextSplitter
from loguru import logger

from app.config import config


class ParentChildSplitResult:
    """父子文档切分结果"""

    def __init__(
        self,
        doc_id: str,
        title: str,
        parent_content: str,
        file_path: str,
        level: str = None,
        alarm_type: str = None,
        child_chunks: Optional[List[Document]] = None,
    ):
        self.doc_id = doc_id
        self.title = title
        self.parent_content = parent_content
        self.file_path = file_path
        self.level = level
        self.alarm_type = alarm_type
        self.child_chunks = child_chunks or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "parent_content": self.parent_content,
            "file_path": self.file_path,
            "level": self.level,
            "alarm_type": self.alarm_type,
            "child_chunks_count": len(self.child_chunks),
        }


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化文档分割服务"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown 标题分割器 (只按一级和二级标题分割，减少分片数)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不再按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        # 递归字符分割器 (用于二次分割，使用更大的chunk_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # 加倍chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段: 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段: 按大小进一步分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段: 合并太小的分片 (< 300字符)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        else:
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                # 当前文档太小且合并后不会太大，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs

    def split_document_with_parent(self, content: str, file_path: str = "") -> Optional[ParentChildSplitResult]:
        """
        父子文档切分：按 ## 标题切分章节，保持语义独立

        切分规则：
        - 按 Markdown 一级/二级标题切割
        - 单个 Chunk 控制在 300~800 字，过长再二次切，过短合并
        - 每个子 Chunk 携带同一个 doc_id 与父文档绑定
        - 不跨章节，保持语义独立

        Args:
            content: Markdown 内容
            file_path: 文件路径

        Returns:
            Optional[ParentChildSplitResult]: 父子文档切分结果，包含父文档信息 + 子 chunks 列表
        """
        if not content or not content.strip():
            logger.warning(f"文档内容为空: {file_path}")
            return None

        try:
            # 生成 doc_id: 取文件名（不含扩展名）+ 序号
            file_name = Path(file_path).stem
            doc_id = f"{file_name}_001"

            # 按 ## 标题分割章节
            sections = self._split_by_headers(content)

            if not sections:
                logger.warning(f"无法分割文档章节: {file_path}")
                return None

            # 提取标题（第一个 # 标题作为父文档标题）
            title = self._extract_title(content)

            # 解析告警级别和类型（从标题中提取，如 "【P2】CPU高负载告警"）
            level, alarm_type = self._extract_alarm_meta(title)

            # 子 chunk 列表
            child_chunks: List[Document] = []
            parent_chunks: List[str] = []

            for i, section in enumerate(sections):
                section_content = section.strip()
                section_len = len(section_content)

                # 处理子 chunk
                if section_len < 300:
                    # 太短，合并到上一个 chunk
                    if child_chunks:
                        child_chunks[-1].page_content += "\n\n" + section_content
                elif section_len > 800:
                    # 过长，二次切分
                    sub_chunks = self._split_chunk(section_content, min_size=300, max_size=800)
                    for sub_chunk in sub_chunks:
                        chunk_doc = Document(
                            page_content=sub_chunk,
                            metadata={
                                "doc_id": doc_id,
                                "_source": file_path,
                                "_file_name": Path(file_path).name,
                                "chunk_index": len(child_chunks),
                            }
                        )
                        child_chunks.append(chunk_doc)
                else:
                    # 正常大小
                    chunk_doc = Document(
                        page_content=section_content,
                        metadata={
                            "doc_id": doc_id,
                            "_source": file_path,
                            "_file_name": Path(file_path).name,
                            "chunk_index": len(child_chunks),
                        }
                    )
                    child_chunks.append(chunk_doc)

                # 收集父文档内容（保留原始章节结构）
                parent_chunks.append(section_content)

            # 组装父文档内容
            parent_content = "\n\n".join(parent_chunks)

            result = ParentChildSplitResult(
                doc_id=doc_id,
                title=title,
                parent_content=parent_content,
                file_path=file_path,
                level=level,
                alarm_type=alarm_type,
                child_chunks=child_chunks,
            )

            logger.info(
                f"父子文档切分完成: {file_path} -> "
                f"doc_id={doc_id}, 子chunk数={len(child_chunks)}"
            )
            return result

        except Exception as e:
            logger.error(f"父子文档切分失败: {file_path}, 错误: {e}")
            return None

    def _split_by_headers(self, content: str) -> List[str]:
        """
        按 ## 标题分割章节，保留标题和内容

        Args:
            content: Markdown 内容

        Returns:
            List[str]: 章节列表，每个元素包含标题 + 内容
        """
        import re

        # 匹配 ## 开头的行作为分隔点
        header_pattern = r"(?=^## )"
        parts = re.split(header_pattern, content, flags=re.MULTILINE)

        # 过滤空章节，保留有内容的部分
        sections = []
        for part in parts:
            stripped = part.strip()
            if stripped and len(stripped) > 0:
                sections.append(stripped)

        return sections

    def _extract_title(self, content: str) -> str:
        """
        提取文档标题（第一个 # 标题）

        Args:
            content: Markdown 内容

        Returns:
            str: 标题文本
        """
        import re

        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # 如果没有 # 标题，取第一行或文件名
        first_line = content.split("\n")[0].strip()
        return first_line[:100] if first_line else "未命名文档"

    def _extract_alarm_meta(self, title: str) -> tuple:
        """
        从标题中提取告警级别和类型

        Args:
            title: 标题文本

        Returns:
            tuple: (level, alarm_type)
        """
        import re

        # 匹配 【P0】【P1】【P2】【P3】等告警级别
        level_match = re.search(r"【([P1234])】", title)
        level = level_match.group(1) if level_match else None

        # 匹配常见告警类型关键词
        alarm_keywords = ["CPU", "内存", "磁盘", "网络", "数据库", "服务", "应用", "告警"]
        alarm_type = None
        for keyword in alarm_keywords:
            if keyword in title:
                alarm_type = keyword
                break

        return level, alarm_type

    def _split_chunk(self, text: str, min_size: int = 300, max_size: int = 800) -> List[str]:
        """
        将文本切分为指定大小的 chunk（不跨行切分）

        Args:
            text: 文本内容
            min_size: 最小 chunk 大小
            max_size: 最大 chunk 大小

        Returns:
            List[str]: chunk 列表
        """
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for line in lines:
            line_len = len(line)
            if current_len + line_len > max_size and current_chunk:
                # 当前 chunk 达到上限，保存
                chunk_text = "\n".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_chunk = [line]
                current_len = line_len
            else:
                current_chunk.append(line)
                current_len += line_len

        # 处理最后一个 chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk).strip()
            if chunk_text:
                if chunks and len(chunks[-1]) + current_len < max_size:
                    # 合并到上一个 chunk
                    chunks[-1] += "\n" + chunk_text
                else:
                    chunks.append(chunk_text)

        return chunks


# 全局单例
document_splitter_service = DocumentSplitterService()
