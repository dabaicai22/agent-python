"""RAG 检索召回评估脚本

基于黄金测试集评估文档切分和检索召回效果，
结果写入 Langfuse 作为 Dataset/Evaluation。

用法:
    .venv/Scripts/python.exe eval_rag.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.core.langfuse_client import langfuse_client
from loguru import logger


# 黄金测试集：query + 期望命中的文档文件名
# 文件名对应 aiops-docs/ 下的文档
GOLDEN_DATASET = [
    {
        "query": "CPU使用率过高怎么处理",
        "expected_files": ["cpu_high_usage.md"],
        "tags": ["cpu", "告警"],
    },
    {
        "query": "服务器CPU超过80%告警排查步骤",
        "expected_files": ["cpu_high_usage.md"],
        "tags": ["cpu", "排查"],
    },
    {
        "query": "内存使用率持续超过75%会有什么影响",
        "expected_files": ["memory_high_usage.md"],
        "tags": ["内存", "告警"],
    },
    {
        "query": "内存泄漏如何排查和处理",
        "expected_files": ["memory_high_usage.md"],
        "tags": ["内存", "排查"],
    },
    {
        "query": "磁盘空间不足告警怎么处理",
        "expected_files": ["disk_high_usage.md"],
        "tags": ["磁盘", "告警"],
    },
    {
        "query": "磁盘使用率超过90%的应急方案",
        "expected_files": ["disk_high_usage.md"],
        "tags": ["磁盘", "应急"],
    },
    {
        "query": "服务不可用503错误怎么排查",
        "expected_files": ["service_unavailable.md"],
        "tags": ["服务", "503"],
    },
    {
        "query": "服务健康检查失败如何恢复",
        "expected_files": ["service_unavailable.md"],
        "tags": ["服务", "健康检查"],
    },
    {
        "query": "接口响应慢怎么优化",
        "expected_files": ["slow_response.md"],
        "tags": ["响应", "优化"],
    },
    {
        "query": "API延迟超过5秒的排查方法",
        "expected_files": ["slow_response.md"],
        "tags": ["响应", "延迟"],
    },
]


def search_and_evaluate(query: str, expected_files: list[str], top_k: int = 5) -> dict:
    """执行检索并评估召回效果"""
    vector_store = vector_store_manager.get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    # 提取召回文档的来源文件名
    retrieved_files = []
    for doc in docs:
        source = doc.metadata.get("_source", "")
        file_name = doc.metadata.get("_file_name", "")
        if file_name:
            retrieved_files.append(file_name)

    # 计算 Recall@K
    hit_count = 0
    for expected in expected_files:
        if expected in retrieved_files:
            hit_count += 1
    recall_at_k = hit_count / len(expected_files) if expected_files else 0

    # 计算 MRR（第一个正确结果的倒数排名）
    mrr = 0.0
    for rank, file_name in enumerate(retrieved_files, 1):
        if file_name in expected_files:
            mrr = 1.0 / rank
            break

    return {
        "query": query,
        "expected_files": expected_files,
        "retrieved_files": retrieved_files,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "hit": recall_at_k > 0,
        "top_k": top_k,
        "num_retrieved": len(docs),
    }


def upload_dataset_to_langfuse():
    """将黄金测试集上传到 Langfuse Dataset"""
    client = langfuse_client.get_client()
    if not client:
        logger.warning("Langfuse 未配置，跳过数据集上传")
        return None

    dataset_name = "rag-retrieval-eval"

    # 创建或获取 Dataset
    try:
        dataset = client.get_dataset(dataset_name)
        logger.info(f"Langfuse Dataset 已存在: {dataset_name}")
    except Exception:
        dataset = client.create_dataset(name=dataset_name)
        logger.info(f"Langfuse Dataset 已创建: {dataset_name}")

    # 上传测试项
    for item in GOLDEN_DATASET:
        client.create_dataset_item(
            dataset_name=dataset_name,
            input={"query": item["query"]},
            expected_output={"expected_files": item["expected_files"]},
            metadata={"tags": item.get("tags", [])},
        )

    logger.info(f"已上传 {len(GOLDEN_DATASET)} 条测试项到 Langfuse Dataset")
    return dataset_name


def run_evaluation():
    """运行评估并输出结果"""
    logger.info("=" * 60)
    logger.info("RAG 检索召回评估开始")
    logger.info(f"测试集大小: {len(GOLDEN_DATASET)}")
    logger.info(f"Top-K: {config.rag_top_k}")
    logger.info("=" * 60)

    results = []
    for i, item in enumerate(GOLDEN_DATASET, 1):
        logger.info(f"[{i}/{len(GOLDEN_DATASET)}] 查询: {item['query']}")
        result = search_and_evaluate(item["query"], item["expected_files"], top_k=config.rag_top_k)
        results.append(result)

        hit_mark = "HIT" if result["hit"] else "MISS"
        logger.info(f"  {hit_mark} 召回: {result['retrieved_files']}, Recall@K={result['recall_at_k']:.2f}, MRR={result['mrr']:.2f}")

    # 汇总统计
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    avg_recall = sum(r["recall_at_k"] for r in results) / total
    avg_mrr = sum(r["mrr"] for r in results) / total
    hit_rate = hits / total

    logger.info("")
    logger.info("=" * 60)
    logger.info("评估结果汇总")
    logger.info("=" * 60)
    logger.info(f"总查询数:     {total}")
    logger.info(f"命中数:       {hits}")
    logger.info(f"命中率:       {hit_rate:.1%}")
    logger.info(f"平均 Recall@K: {avg_recall:.2f}")
    logger.info(f"平均 MRR:      {avg_mrr:.2f}")
    logger.info("")

    # 未命中的 bad case
    bad_cases = [r for r in results if not r["hit"]]
    if bad_cases:
        logger.warning(f"未命中查询 ({len(bad_cases)} 条):")
        for r in bad_cases:
            logger.warning(f"  - Query: {r['query']}")
            logger.warning(f"    期望: {r['expected_files']}, 实际: {r['retrieved_files']}")

    # 将评估结果写入 Langfuse
    _log_evaluation_to_langfuse(results, avg_recall, avg_mrr, hit_rate)

    return results


def _log_evaluation_to_langfuse(results: list, avg_recall: float, avg_mrr: float, hit_rate: float):
    """将评估结果写入 Langfuse"""
    client = langfuse_client.get_client()
    if not client:
        logger.warning("Langfuse 未配置，跳过评估结果上传")
        return

    try:
        for r in results:
            trace = client.trace(
                name="rag-eval",
                metadata={
                    "type": "retrieval_evaluation",
                    "query": r["query"],
                    "expected_files": r["expected_files"],
                    "retrieved_files": r["retrieved_files"],
                    "recall_at_k": r["recall_at_k"],
                    "mrr": r["mrr"],
                    "hit": r["hit"],
                },
            )

            # 为每个召回的 chunk 记录 span
            for i, file_name in enumerate(r["retrieved_files"]):
                is_expected = file_name in r["expected_files"]
                trace.span(
                    name=f"retrieved-{i}",
                    input=r["query"],
                    output=file_name,
                    metadata={"rank": i + 1, "is_expected": is_expected},
                )

            # 记录评估分数
            trace.score(name="recall_at_k", value=r["recall_at_k"])
            trace.score(name="mrr", value=r["mrr"])

        # 记录整体评估摘要
        summary_trace = client.trace(
            name="rag-eval-summary",
            metadata={
                "type": "evaluation_summary",
                "total_queries": len(results),
                "avg_recall": avg_recall,
                "avg_mrr": avg_mrr,
                "hit_rate": hit_rate,
                "top_k": config.rag_top_k,
                "chunk_max_size": config.chunk_max_size,
                "chunk_overlap": config.chunk_overlap,
            },
        )
        summary_trace.score(name="avg_recall", value=avg_recall)
        summary_trace.score(name="avg_mrr", value=avg_mrr)
        summary_trace.score(name="hit_rate", value=hit_rate)

        # 确保数据发送
        client.flush()

        logger.info("评估结果已写入 Langfuse")

    except Exception as e:
        logger.warning(f"Langfuse 评估结果写入失败: {e}")


if __name__ == "__main__":
    # 先上传数据集
    upload_dataset_to_langfuse()
    # 运行评估
    run_evaluation()
