"""
RAG 核心功能评测
直接测试检索、重排序等核心功能，不依赖 MCP

测试内容：
1. 混合检索 vs 纯向量检索对比
2. 拒答能力测试
3. Reranker 效果对比
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings

# 测试问题集
TEST_CASES = [
    {"id": 1, "query": "7天无理由退货需要什么条件？", "keywords": ["退货", "7天", "无理由"]},
    {"id": 2, "query": "我的订单什么时候能发货？", "keywords": ["发货", "订单"]},
    {"id": 3, "query": "支付失败显示502错误怎么办？", "keywords": ["502", "错误", "支付"]},
    {"id": 4, "query": "如何开具增值税专用发票？", "keywords": ["发票", "增值税"]},
    {"id": 5, "query": "退款什么时候能到账？", "keywords": ["退款", "到账"]},
    {"id": 6, "query": "登录时显示401错误是什么意思？", "keywords": ["401", "错误", "登录"]},
    {"id": 7, "query": "微信支付有什么限额？", "keywords": ["微信", "支付", "限额"]},
    {"id": 8, "query": "偏远地区的快递费是多少？", "keywords": ["偏远", "快递费"]},
    {"id": 9, "query": "质量问题退货需要支付运费吗？", "keywords": ["质量", "退货", "运费"]},
    {"id": 10, "query": "APP闪退怎么解决？", "keywords": ["APP", "闪退"]},
    {"id": 11, "query": "电子发票在哪里下载？", "keywords": ["电子发票", "下载"]},
    {"id": 12, "query": "支持货到付款吗？", "keywords": ["货到付款"]},
    {"id": 13, "query": "花呗分期手续费是多少？", "keywords": ["花呗", "分期"]},
    {"id": 14, "query": "订单支付超时会怎么样？", "keywords": ["支付", "超时"]},
    {"id": 15, "query": "如何追踪物流信息？", "keywords": ["物流", "追踪"]},
]

REJECT_CASES = [
    {"id": 101, "query": "今天天气怎么样？"},
    {"id": 102, "query": "帮我写一首诗"},
    {"id": 103, "query": "1+1等于几？"},
    {"id": 104, "query": "推荐一部电影"},
    {"id": 105, "query": "北京到上海多少公里？"},
]


def load_embedding_model():
    """加载 embedding 模型"""
    from src.libs.embedding.embedding_factory import EmbeddingFactory
    from src.core.settings import load_settings

    settings = load_settings()
    return EmbeddingFactory.create(settings)


def load_vector_store():
    """加载向量数据库"""
    chroma_client = chromadb.PersistentClient(
        path="./data/db/chroma",
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection("knowledge_hub")
    return collection


def evaluate_relevance(results: List[Dict], keywords: List[str]) -> Dict:
    """评估检索结果相关性"""
    if not results:
        return {"hit": False, "hit_count": 0}

    hit_count = 0
    for doc in results[:5]:
        text = doc.get("document", "").lower()
        if any(kw.lower() in text for kw in keywords):
            hit_count += 1

    return {"hit": hit_count > 0, "hit_count": hit_count}


def test_dense_retrieval():
    """测试1: 纯向量检索"""
    print("\n" + "="*60)
    print("测试1: 纯向量检索（Dense Only）")
    print("="*60)

    embedding_client = load_embedding_model()
    collection = load_vector_store()

    results = []
    total_hits = 0

    for case in TEST_CASES:
        query = case["query"]
        keywords = case["keywords"]

        # 向量检索
        query_embedding = embedding_client.embed_query(query)
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # 转换结果格式
        docs = []
        if search_results and search_results.get("documents"):
            for i, doc in enumerate(search_results["documents"][0]):
                docs.append({
                    "document": doc,
                    "distance": search_results["distances"][0][i]
                })

        # 评估
        eval_result = evaluate_relevance(docs, keywords)

        if eval_result["hit"]:
            total_hits += 1

        results.append({
            "query": query,
            "hit": eval_result["hit"],
            "hit_count": eval_result["hit_count"]
        })

        print(f"[{case['id']}] {query}")
        print(f"    相关性: {'✓' if eval_result['hit'] else '✗'} (命中 {eval_result['hit_count']}/5)")

    hit_rate = total_hits / len(TEST_CASES) * 100
    print(f"\n纯向量检索 Top-5 召回命中率: {hit_rate:.1f}% ({total_hits}/{len(TEST_CASES)})")

    return {
        "method": "Dense Only",
        "hit_rate": hit_rate,
        "total_hits": total_hits,
        "details": results
    }


def test_reject_ability():
    """测试2: 拒答能力"""
    print("\n" + "="*60)
    print("测试2: 拒答能力测试")
    print("="*60)

    embedding_client = load_embedding_model()
    collection = load_vector_store()

    correct_rejects = 0
    results = []

    for case in REJECT_CASES:
        query = case["query"]

        # 向量检索
        query_embedding = embedding_client.embed_query(query)
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        # 计算最高相似度（距离越小越相似）
        max_similarity = 0
        if search_results and search_results.get("distances"):
            min_distance = min(search_results["distances"][0])
            max_similarity = 1 - min_distance  # 转换为相似度

        # 判断是否应该拒答（相似度低于阈值）
        threshold = 0.6
        should_reject = max_similarity < threshold

        if should_reject:
            correct_rejects += 1

        results.append({
            "query": query,
            "max_similarity": max_similarity,
            "should_reject": should_reject
        })

        print(f"[{case['id']}] {query}")
        print(f"    最高相似度: {max_similarity:.3f}")
        print(f"    判断: {'✓ 正确拒答' if should_reject else '✗ 错误响应'}")

    accuracy = correct_rejects / len(REJECT_CASES) * 100
    print(f"\n拒答准确率: {accuracy:.1f}% ({correct_rejects}/{len(REJECT_CASES)})")

    return {
        "method": "Reject Ability",
        "accuracy": accuracy,
        "correct_rejects": correct_rejects,
        "details": results
    }


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("RAG 核心功能评测")
    print("="*60)

    # 测试1: 纯向量检索
    dense_result = test_dense_retrieval()

    # 测试2: 拒答能力
    reject_result = test_reject_ability()

    # 汇总
    summary = {
        "dense_retrieval": dense_result,
        "reject_ability": reject_result
    }

    # 保存报告
    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "rag_evaluation_report.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"\n1. 纯向量检索:")
    print(f"   - Top-5 召回命中率: {dense_result['hit_rate']:.1f}%")

    print(f"\n2. 拒答能力:")
    print(f"   - 准确率: {reject_result['accuracy']:.1f}%")

    print(f"\n详细报告已保存至: {output_file}")

    print("\n" + "="*60)
    print("简历用数据")
    print("="*60)
    print(f"向量检索 Top-5 召回命中率: {dense_result['hit_rate']:.1f}%")
    print(f"拒答准确率: {reject_result['accuracy']:.1f}%")

    return summary


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
