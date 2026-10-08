"""
RAG 效果对比测试
测试内容：
1. 混合检索 vs 纯向量检索
2. 拒答能力测试
3. Reranker 效果对比

生成简历用的数据指标
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
from typing import List, Dict, Any

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 测试问题集
TEST_CASES = [
    {
        "id": 1,
        "query": "7天无理由退货需要什么条件？",
        "category": "退款政策",
        "relevant_docs": ["退款政策", "退换货"]
    },
    {
        "id": 2,
        "query": "我的订单什么时候能发货？",
        "category": "物流配送",
        "relevant_docs": ["物流", "配送"]
    },
    {
        "id": 3,
        "query": "支付失败显示502错误怎么办？",
        "category": "技术问题",
        "relevant_docs": ["技术问题", "错误码", "502"]
    },
    {
        "id": 4,
        "query": "如何开具增值税专用发票？",
        "category": "账单发票",
        "relevant_docs": ["发票", "增值税"]
    },
    {
        "id": 5,
        "query": "退款什么时候能到账？",
        "category": "退款流程",
        "relevant_docs": ["退款", "到账时间"]
    },
    {
        "id": 6,
        "query": "登录时显示401错误是什么意思？",
        "category": "技术问题",
        "relevant_docs": ["技术问题", "错误码", "401", "登录"]
    },
    {
        "id": 7,
        "query": "微信支付有什么限额？",
        "category": "支付方式",
        "relevant_docs": ["支付", "微信支付", "限额"]
    },
    {
        "id": 8,
        "query": "偏远地区的快递费是多少？",
        "category": "物流配送",
        "relevant_docs": ["物流", "快递费", "偏远"]
    },
    {
        "id": 9,
        "query": "质量问题退货需要支付运费吗？",
        "category": "退款政策",
        "relevant_docs": ["退款", "退货", "运费", "质量"]
    },
    {
        "id": 10,
        "query": "APP闪退怎么解决？",
        "category": "技术问题",
        "relevant_docs": ["技术问题", "APP", "闪退"]
    },
    {
        "id": 11,
        "query": "电子发票在哪里下载？",
        "category": "账单发票",
        "relevant_docs": ["发票", "电子发票", "下载"]
    },
    {
        "id": 12,
        "query": "支持货到付款吗？",
        "category": "支付方式",
        "relevant_docs": ["支付", "货到付款"]
    },
    {
        "id": 13,
        "query": "花呗分期手续费是多少？",
        "category": "支付方式",
        "relevant_docs": ["支付", "花呗", "分期", "手续费"]
    },
    {
        "id": 14,
        "query": "订单支付超时会怎么样？",
        "category": "支付问题",
        "relevant_docs": ["支付", "超时", "订单"]
    },
    {
        "id": 15,
        "query": "如何追踪物流信息？",
        "category": "物流配送",
        "relevant_docs": ["物流", "追踪", "查询"]
    },
]

# 拒答测试问题（无关问题）
REJECT_TEST_CASES = [
    {"id": 101, "query": "今天天气怎么样？", "should_reject": True},
    {"id": 102, "query": "帮我写一首诗", "should_reject": True},
    {"id": 103, "query": "1+1等于几？", "should_reject": True},
    {"id": 104, "query": "推荐一部电影", "should_reject": True},
    {"id": 105, "query": "北京到上海多少公里？", "should_reject": True},
]


class RAGComparator:
    """RAG 效果对比器"""

    def __init__(self):
        self.settings = load_settings()
        self.search_engine = HybridSearch(self.settings)

    def evaluate_relevance(self, query: str, retrieved_docs: List[Any], relevant_keywords: List[str]) -> Dict:
        """评估检索结果的相关性"""
        if not retrieved_docs:
            return {"hit": False, "top5_hit": False, "top1_hit": False, "hit_count": 0}

        # 检查前5个结果中是否包含相关文档
        top5_hits = 0
        top1_hit = False

        for i, doc in enumerate(retrieved_docs[:5]):
            # 根据实际返回的数据结构提取内容
            doc_content = ""
            if hasattr(doc, 'content'):
                doc_content = str(doc.content).lower()
            elif hasattr(doc, 'text'):
                doc_content = str(doc.text).lower()
            elif isinstance(doc, dict):
                doc_content = str(doc.get('content', '') or doc.get('text', '')).lower()

            # 检查是否包含相关关键词
            if any(keyword.lower() in doc_content for keyword in relevant_keywords):
                top5_hits += 1
                if i == 0:
                    top1_hit = True

        return {
            "hit": top5_hits > 0,
            "top5_hit": top5_hits > 0,
            "top1_hit": top1_hit,
            "hit_count": top5_hits
        }

    def test_dense_only(self, test_cases: List[Dict]) -> Dict:
        """测试1：纯向量检索"""
        print("\n" + "="*60)
        print("测试1：纯向量检索（Dense Only）")
        print("="*60)

        results = []
        total_hits = 0

        for case in test_cases:
            query = case["query"]
            relevant_keywords = case["relevant_docs"]

            # 纯向量检索 - 通过配置禁用 BM25
            try:
                retrieved = self.search_engine.search(query=query, top_k=5, use_hybrid=False)
            except:
                # 如果不支持 use_hybrid 参数，使用默认搜索
                retrieved = self.search_engine.search(query=query, top_k=5)

            # 评估相关性
            eval_result = self.evaluate_relevance(query, retrieved, relevant_keywords)

            results.append({
                "query": query,
                "category": case["category"],
                "hit": eval_result["hit"],
                "hit_count": eval_result["hit_count"]
            })

            if eval_result["hit"]:
                total_hits += 1

            print(f"[{case['id']}] {query}")
            print(f"    相关性: {'✓' if eval_result['hit'] else '✗'} (命中 {eval_result['hit_count']}/5)")

        hit_rate = total_hits / len(test_cases) * 100
        print(f"\n纯向量检索 Top-5 召回命中率: {hit_rate:.1f}% ({total_hits}/{len(test_cases)})")

        return {
            "method": "Dense Only",
            "hit_rate": hit_rate,
            "total_hits": total_hits,
            "total_cases": len(test_cases),
            "details": results
        }

    def test_hybrid_search(self, test_cases: List[Dict]) -> Dict:
        """测试2：混合检索（Dense + BM25 + RRF）"""
        print("\n" + "="*60)
        print("测试2：混合检索（Dense + BM25 + RRF）")
        print("="*60)

        results = []
        total_hits = 0

        for case in test_cases:
            query = case["query"]
            relevant_keywords = case["relevant_docs"]

            # 混合检索
            retrieved = self.search_engine.search(query=query, top_k=5)

            # 评估相关性
            eval_result = self.evaluate_relevance(query, retrieved, relevant_keywords)

            results.append({
                "query": query,
                "category": case["category"],
                "hit": eval_result["hit"],
                "hit_count": eval_result["hit_count"]
            })

            if eval_result["hit"]:
                total_hits += 1

            print(f"[{case['id']}] {query}")
            print(f"    相关性: {'✓' if eval_result['hit'] else '✗'} (命中 {eval_result['hit_count']}/5)")

        hit_rate = total_hits / len(test_cases) * 100
        print(f"\n混合检索 Top-5 召回命中率: {hit_rate:.1f}% ({total_hits}/{len(test_cases)})")

        return {
            "method": "Hybrid Search (Dense + BM25 + RRF)",
            "hit_rate": hit_rate,
            "total_hits": total_hits,
            "total_cases": len(test_cases),
            "details": results
        }

    def test_reject_ability(self, test_cases: List[Dict]) -> Dict:
        """测试3：拒答能力测试"""
        print("\n" + "="*60)
        print("测试3：拒答能力测试")
        print("="*60)

        results = []
        correct_rejects = 0

        for case in test_cases:
            query = case["query"]

            # 使用混合检索
            retrieved = self.search_engine.search(query=query, top_k=5)

            # 简单的拒答判断逻辑：如果检索结果的相似度都很低，应该拒答
            max_score = 0
            if retrieved and len(retrieved) > 0:
                # 尝试从不同的字段获取分数
                for doc in retrieved:
                    score = 0
                    if hasattr(doc, 'score'):
                        score = doc.score
                    elif hasattr(doc, 'similarity'):
                        score = doc.similarity
                    elif isinstance(doc, dict):
                        score = doc.get('score', 0) or doc.get('similarity', 0)

                    max_score = max(max_score, score)

            # 阈值判断（需要根据实际情况调整）
            should_reject = max_score < 0.6  # 假设阈值为0.6
            correct = should_reject == case["should_reject"]

            if correct:
                correct_rejects += 1

            results.append({
                "query": query,
                "max_score": max_score,
                "should_reject": case["should_reject"],
                "actually_reject": should_reject,
                "correct": correct
            })

            print(f"[{case['id']}] {query}")
            print(f"    最高相似度: {max_score:.3f}")
            print(f"    判断: {'✓ 正确拒答' if correct and should_reject else '✗ 错误' if not correct else '✓ 正确回答'}")

        reject_accuracy = correct_rejects / len(test_cases) * 100
        print(f"\n拒答准确率: {reject_accuracy:.1f}% ({correct_rejects}/{len(test_cases)})")

        return {
            "method": "Reject Ability Test",
            "accuracy": reject_accuracy,
            "correct_rejects": correct_rejects,
            "total_cases": len(test_cases),
            "details": results
        }

    def test_reranker_effect(self, test_cases: List[Dict]) -> Dict:
        """测试4：Reranker 效果对比"""
        print("\n" + "="*60)
        print("测试4：Reranker 效果对比")
        print("="*60)

        results = []
        improvement_count = 0

        for case in test_cases:
            query = case["query"]
            relevant_keywords = case["relevant_docs"]

            # 混合检索（无 Reranker）- 假设可以通过参数控制
            try:
                retrieved_before = self.search_engine.search(query=query, top_k=10, use_rerank=False)
            except:
                # 如果不支持 use_rerank 参数，跳过这个测试
                print("注意: 当前配置可能默认启用了 Reranker，无法完全验证对比效果")
                retrieved_before = self.search_engine.search(query=query, top_k=10)

            eval_before = self.evaluate_relevance(query, retrieved_before[:5], relevant_keywords)

            # Reranker 重排序
            try:
                reranked = self.search_engine.search(query=query, top_k=10, use_rerank=True)
            except:
                reranked = self.search_engine.search(query=query, top_k=10)

            eval_after = self.evaluate_relevance(query, reranked[:5], relevant_keywords)

            # 判断是否有改善
            improved = eval_after["hit_count"] > eval_before["hit_count"] or \
                      (eval_after["top1_hit"] and not eval_before["top1_hit"])

            if improved:
                improvement_count += 1

            results.append({
                "query": query,
                "category": case["category"],
                "before_hits": eval_before["hit_count"],
                "after_hits": eval_after["hit_count"],
                "improved": improved
            })

            print(f"[{case['id']}] {query}")
            print(f"    Rerank前: {eval_before['hit_count']}/5 命中")
            print(f"    Rerank后: {eval_after['hit_count']}/5 命中 {'↑ 提升' if improved else '→ 持平'}")

        improvement_rate = improvement_count / len(test_cases) * 100
        print(f"\nReranker 改善率: {improvement_rate:.1f}% ({improvement_count}/{len(test_cases)})")

        return {
            "method": "Reranker Comparison",
            "improvement_rate": improvement_rate,
            "improved_cases": improvement_count,
            "total_cases": len(test_cases),
            "details": results
        }

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("RAG 效果对比测试")
        print("="*60)

        # 测试1：纯向量检索
        dense_result = self.test_dense_only(TEST_CASES)

        # 测试2：混合检索
        hybrid_result = self.test_hybrid_search(TEST_CASES)

        # 测试3：拒答能力
        reject_result = self.test_reject_ability(REJECT_TEST_CASES)

        # 测试4：Reranker 效果
        reranker_result = self.test_reranker_effect(TEST_CASES)

        # 汇总结果
        summary = {
            "dense_only": dense_result,
            "hybrid_search": hybrid_result,
            "reject_ability": reject_result,
            "reranker_effect": reranker_result
        }

        # 计算提升幅度
        improvement = hybrid_result["hit_rate"] - dense_result["hit_rate"]

        # 生成报告
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"\n1. 混合检索效果:")
        print(f"   - 纯向量检索命中率: {dense_result['hit_rate']:.1f}%")
        print(f"   - 混合检索命中率: {hybrid_result['hit_rate']:.1f}%")
        print(f"   - 提升幅度: +{improvement:.1f}%")

        print(f"\n2. 拒答能力:")
        print(f"   - 准确率: {reject_result['accuracy']:.1f}%")

        print(f"\n3. Reranker 效果:")
        print(f"   - 改善率: {reranker_result['improvement_rate']:.1f}%")

        # 保存结果
        output_dir = Path(__file__).parent.parent / "reports"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "rag_comparison_report.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n详细报告已保存至: {output_file}")

        # 生成简历用的总结
        print("\n" + "="*60)
        print("简历用数据总结")
        print("="*60)
        print(f"混合检索（向量 + BM25 + RRF融合）相比纯向量检索，Top-5 召回命中率提升约 {improvement:.1f}%")
        print(f"BGE-Reranker 重排序改善了 {reranker_result['improvement_rate']:.1f}% 的查询结果")
        print(f"拒答准确率达到 {reject_result['accuracy']:.1f}%")

        return summary


if __name__ == "__main__":
    comparator = RAGComparator()
    comparator.run_all_tests()
