"""
RAG 效果对比测试 - 简化版
直接通过配置文件测试，模拟不同的检索策略

测试内容：
1. 混合检索效果验证
2. 拒答能力测试
3. 检索结果质量评估

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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# 测试问题集 - 客服领域相关问题
TEST_CASES = [
    {
        "id": 1,
        "query": "7天无理由退货需要什么条件？",
        "category": "退款政策",
        "relevant_keywords": ["退货", "7天", "无理由", "条件"]
    },
    {
        "id": 2,
        "query": "我的订单什么时候能发货？",
        "category": "物流配送",
        "relevant_keywords": ["发货", "时间", "订单", "物流"]
    },
    {
        "id": 3,
        "query": "支付失败显示502错误怎么办？",
        "category": "技术问题",
        "relevant_keywords": ["502", "错误", "支付", "失败"]
    },
    {
        "id": 4,
        "query": "如何开具增值税专用发票？",
        "category": "账单发票",
        "relevant_keywords": ["发票", "增值税", "专用发票", "开具"]
    },
    {
        "id": 5,
        "query": "退款什么时候能到账？",
        "category": "退款流程",
        "relevant_keywords": ["退款", "到账", "时间"]
    },
    {
        "id": 6,
        "query": "登录时显示401错误是什么意思？",
        "category": "技术问题",
        "relevant_keywords": ["401", "错误", "登录", "认证"]
    },
    {
        "id": 7,
        "query": "微信支付有什么限额？",
        "category": "支付方式",
        "relevant_keywords": ["微信", "支付", "限额"]
    },
    {
        "id": 8,
        "query": "偏远地区的快递费是多少？",
        "category": "物流配送",
        "relevant_keywords": ["偏远", "快递费", "运费"]
    },
    {
        "id": 9,
        "query": "质量问题退货需要支付运费吗？",
        "category": "退款政策",
        "relevant_keywords": ["质量", "退货", "运费"]
    },
    {
        "id": 10,
        "query": "APP闪退怎么解决？",
        "category": "技术问题",
        "relevant_keywords": ["APP", "闪退", "崩溃"]
    },
    {
        "id": 11,
        "query": "电子发票在哪里下载？",
        "category": "账单发票",
        "relevant_keywords": ["电子发票", "下载"]
    },
    {
        "id": 12,
        "query": "支持货到付款吗？",
        "category": "支付方式",
        "relevant_keywords": ["货到付款"]
    },
    {
        "id": 13,
        "query": "花呗分期手续费是多少？",
        "category": "支付方式",
        "relevant_keywords": ["花呗", "分期", "手续费"]
    },
    {
        "id": 14,
        "query": "订单支付超时会怎么样？",
        "category": "支付问题",
        "relevant_keywords": ["支付", "超时", "订单"]
    },
    {
        "id": 15,
        "query": "如何追踪物流信息？",
        "category": "物流配送",
        "relevant_keywords": ["物流", "追踪", "查询"]
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


def evaluate_relevance(retrieved_chunks: List[Dict], relevant_keywords: List[str]) -> Dict:
    """评估检索结果的相关性"""
    if not retrieved_chunks:
        return {"hit": False, "hit_count": 0, "top1_hit": False}

    hit_count = 0
    top1_hit = False

    for i, chunk in enumerate(retrieved_chunks[:5]):
        text = chunk.get("text", "").lower()

        # 检查是否包含相关关键词
        if any(keyword.lower() in text for keyword in relevant_keywords):
            hit_count += 1
            if i == 0:
                top1_hit = True

    return {
        "hit": hit_count > 0,
        "hit_count": hit_count,
        "top1_hit": top1_hit
    }


def test_search_quality():
    """测试搜索质量

    由于需要实际的向量数据库连接，这个测试需要：
    1. 确保数据已经导入到 ChromaDB
    2. 确保配置文件正确

    这里提供测试框架，实际执行需要手动运行或通过 MCP 接口测试
    """

    print("\n" + "="*60)
    print("RAG 效果测试准备")
    print("="*60)
    print("\n测试说明:")
    print("1. 本测试需要先导入客服文档到向量数据库")
    print("2. 确保 config/settings.yaml 配置正确")
    print("3. 运行数据导入脚本: python -m src.ingestion.ingest_documents")
    print("\n测试用例准备:")
    print(f"- 客服相关问题: {len(TEST_CASES)} 个")
    print(f"- 拒答测试问题: {len(REJECT_TEST_CASES)} 个")

    print("\n问题示例:")
    for case in TEST_CASES[:5]:
        print(f"  [{case['id']}] {case['query']}")

    print("\n" + "="*60)
    print("手动测试步骤")
    print("="*60)
    print("\n方法1: 通过 MCP 接口测试")
    print("  1. 启动 MCP Server: python start_mcp_server.py")
    print("  2. 使用 MCP 客户端调用 'search_knowledge' 工具")
    print("  3. 记录每个测试问题的检索结果")

    print("\n方法2: 直接使用 Python API 测试")
    print("  运行: python scripts/run_search_test.py")

    print("\n" + "="*60)
    print("评估指标")
    print("="*60)
    print("1. Top-5 召回命中率: 前5个结果中是否包含相关文档")
    print("2. 拒答准确率: 对无关问题能否正确拒答")
    print("3. Reranker 改善率: 重排序后相关文档排名提升比例")

    # 保存测试用例到文件
    output_dir = PROJECT_ROOT / "tests" / "fixtures"
    output_dir.mkdir(exist_ok=True, parents=True)

    test_set_file = output_dir / "rag_test_set.json"
    with open(test_set_file, 'w', encoding='utf-8') as f:
        json.dump({
            "relevant_queries": TEST_CASES,
            "reject_queries": REJECT_TEST_CASES
        }, f, ensure_ascii=False, indent=2)

    print(f"\n测试用例已保存至: {test_set_file}")

    return {
        "test_cases": TEST_CASES,
        "reject_cases": REJECT_TEST_CASES,
        "test_set_file": str(test_set_file)
    }


if __name__ == "__main__":
    result = test_search_quality()

    print("\n" + "="*60)
    print("后续步骤")
    print("="*60)
    print("\n1. 确保已导入客服文档数据")
    print("2. 运行实际的搜索测试脚本")
    print("3. 收集测试结果并生成报告")
    print("\n建议: 先用少量问题手动测试，确认系统正常后再批量测试")
