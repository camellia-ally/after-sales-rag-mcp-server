"""
收集 Rerank 后的分数分布，确定拒答阈值

目标：
1. 收集可回答问题的 Rerank 分数
2. 收集不可回答问题的 Rerank 分数
3. 分析分数分布，推荐合适的阈值
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Dict
import chromadb
from chromadb.config import Settings as ChromaSettings

# 可回答问题（客服相关）
ANSWERABLE_CASES = [
    {"id": 1, "query": "7天无理由退货需要什么条件？", "type": "answerable"},
    {"id": 2, "query": "我的订单什么时候能发货？", "type": "answerable"},
    {"id": 3, "query": "支付失败显示502错误怎么办？", "type": "answerable"},
    {"id": 4, "query": "如何开具增值税专用发票？", "type": "answerable"},
    {"id": 5, "query": "退款什么时候能到账？", "type": "answerable"},
    {"id": 6, "query": "登录时显示401错误是什么意思？", "type": "answerable"},
    {"id": 7, "query": "微信支付有什么限额？", "type": "answerable"},
    {"id": 8, "query": "偏远地区的快递费是多少？", "type": "answerable"},
    {"id": 9, "query": "质量问题退货需要支付运费吗？", "type": "answerable"},
    {"id": 10, "query": "APP闪退怎么解决？", "type": "answerable"},
]

# 不可回答问题（无关问题）
UNANSWERABLE_CASES = [
    {"id": 101, "query": "今天天气怎么样？", "type": "unanswerable"},
    {"id": 102, "query": "帮我写一首诗", "type": "unanswerable"},
    {"id": 103, "query": "1+1等于几？", "type": "unanswerable"},
    {"id": 104, "query": "推荐一部电影", "type": "unanswerable"},
    {"id": 105, "query": "北京到上海多少公里？", "type": "unanswerable"},
    {"id": 106, "query": "如何做红烧肉？", "type": "unanswerable"},
    {"id": 107, "query": "什么是量子力学？", "type": "unanswerable"},
    {"id": 108, "query": "给我讲个笑话", "type": "unanswerable"},
]


def collect_scores():
    """收集 Rerank 后的分数"""
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.core.settings import load_settings
    from src.libs.reranker.reranker_factory import RerankerFactory

    settings = load_settings()
    embedding_client = EmbeddingFactory.create(settings)
    reranker = RerankerFactory.create(settings)

    # 加载向量数据库
    chroma_client = chromadb.PersistentClient(
        path="./data/db/chroma",
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection("knowledge_hub")

    results = {
        "answerable": [],
        "unanswerable": []
    }

    print("\n" + "="*60)
    print("收集可回答问题的 Rerank 分数")
    print("="*60)

    for case in ANSWERABLE_CASES:
        query = case["query"]

        # 向量检索
        query_embedding = embedding_client.embed_query(query)
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

        # 准备 rerank 输入
        if not search_results or not search_results.get("documents"):
            print(f"[{case['id']}] {query} - 无检索结果")
            results["answerable"].append({
                "id": case["id"],
                "query": query,
                "max_score": 0.0,
                "top3_scores": []
            })
            continue

        docs = [{"text": doc} for doc in search_results["documents"][0]]

        # Rerank
        reranked = reranker.rerank(query, docs, top_k=5)

        if reranked:
            max_score = reranked[0].get("score", 0.0)
            top3_scores = [r.get("score", 0.0) for r in reranked[:3]]
        else:
            max_score = 0.0
            top3_scores = []

        results["answerable"].append({
            "id": case["id"],
            "query": query,
            "max_score": max_score,
            "top3_scores": top3_scores
        })

        print(f"[{case['id']}] {query}")
        print(f"    Top-1 分数: {max_score:.4f}")
        print(f"    Top-3 分数: {top3_scores}")

    print("\n" + "="*60)
    print("收集不可回答问题的 Rerank 分数")
    print("="*60)

    for case in UNANSWERABLE_CASES:
        query = case["query"]

        # 向量检索
        query_embedding = embedding_client.embed_query(query)
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10
        )

        # 准备 rerank 输入
        if not search_results or not search_results.get("documents"):
            print(f"[{case['id']}] {query} - 无检索结果")
            results["unanswerable"].append({
                "id": case["id"],
                "query": query,
                "max_score": 0.0,
                "top3_scores": []
            })
            continue

        docs = [{"text": doc} for doc in search_results["documents"][0]]

        # Rerank
        reranked = reranker.rerank(query, docs, top_k=5)

        if reranked:
            max_score = reranked[0].get("score", 0.0)
            top3_scores = [r.get("score", 0.0) for r in reranked[:3]]
        else:
            max_score = 0.0
            top3_scores = []

        results["unanswerable"].append({
            "id": case["id"],
            "query": query,
            "max_score": max_score,
            "top3_scores": top3_scores
        })

        print(f"[{case['id']}] {query}")
        print(f"    Top-1 分数: {max_score:.4f}")
        print(f"    Top-3 分数: {top3_scores}")

    return results


def analyze_scores(results: Dict):
    """分析分数分布，推荐阈值"""
    answerable_scores = [r["max_score"] for r in results["answerable"]]
    unanswerable_scores = [r["max_score"] for r in results["unanswerable"]]

    print("\n" + "="*60)
    print("分数分布分析")
    print("="*60)

    print(f"\n可回答问题（{len(answerable_scores)} 个）:")
    print(f"  最高分: {max(answerable_scores):.4f}")
    print(f"  最低分: {min(answerable_scores):.4f}")
    print(f"  平均分: {sum(answerable_scores)/len(answerable_scores):.4f}")
    print(f"  中位数: {sorted(answerable_scores)[len(answerable_scores)//2]:.4f}")

    print(f"\n不可回答问题（{len(unanswerable_scores)} 个）:")
    print(f"  最高分: {max(unanswerable_scores):.4f}")
    print(f"  最低分: {min(unanswerable_scores):.4f}")
    print(f"  平均分: {sum(unanswerable_scores)/len(unanswerable_scores):.4f}")
    print(f"  中位数: {sorted(unanswerable_scores)[len(unanswerable_scores)//2]:.4f}")

    # 推荐阈值
    min_answerable = min(answerable_scores)
    max_unanswerable = max(unanswerable_scores)

    print("\n" + "="*60)
    print("阈值推荐")
    print("="*60)

    if min_answerable > max_unanswerable:
        # 有明显分界
        gap = min_answerable - max_unanswerable
        recommended_threshold = max_unanswerable + gap * 0.5
        print(f"\n✓ 两类问题分数有明显分界")
        print(f"  可回答最低分: {min_answerable:.4f}")
        print(f"  不可回答最高分: {max_unanswerable:.4f}")
        print(f"  分数差距: {gap:.4f}")
        print(f"\n推荐阈值: {recommended_threshold:.4f}")
        print(f"  (设置在两类分数中间位置)")
    else:
        # 有重叠
        overlap = max_unanswerable - min_answerable
        print(f"\n✗ 两类问题分数有重叠")
        print(f"  可回答最低分: {min_answerable:.4f}")
        print(f"  不可回答最高分: {max_unanswerable:.4f}")
        print(f"  重叠范围: {overlap:.4f}")

        # 尝试不同阈值的效果
        print(f"\n尝试不同阈值的效果:")
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            correct_reject = sum(1 for s in unanswerable_scores if s < threshold)
            wrong_reject = sum(1 for s in answerable_scores if s < threshold)
            reject_accuracy = correct_reject / len(unanswerable_scores) * 100
            false_reject_rate = wrong_reject / len(answerable_scores) * 100

            print(f"  阈值 {threshold:.1f}: 拒答准确率 {reject_accuracy:.1f}%, 误拒率 {false_reject_rate:.1f}%")

        # 推荐平衡点
        avg_answerable = sum(answerable_scores) / len(answerable_scores)
        avg_unanswerable = sum(unanswerable_scores) / len(unanswerable_scores)
        recommended_threshold = (avg_answerable + avg_unanswerable) / 2
        print(f"\n推荐阈值（折中方案）: {recommended_threshold:.4f}")

    # 保存结果
    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "threshold_analysis.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "answerable": results["answerable"],
            "unanswerable": results["unanswerable"],
            "statistics": {
                "answerable_min": min(answerable_scores),
                "answerable_max": max(answerable_scores),
                "answerable_avg": sum(answerable_scores)/len(answerable_scores),
                "unanswerable_min": min(unanswerable_scores),
                "unanswerable_max": max(unanswerable_scores),
                "unanswerable_avg": sum(unanswerable_scores)/len(unanswerable_scores),
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n详细数据已保存至: {output_file}")


if __name__ == "__main__":
    try:
        results = collect_scores()
        analyze_scores(results)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
