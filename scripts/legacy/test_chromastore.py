"""
直接测试ChromaStore是否能读取portfolio集合
"""
import sys
from pathlib import Path

# 添加MCP服务器路径
sys.path.insert(0, "D:/PyCharm/pytest/实习/MODULAR-RAG-MCP-SERVER-main")

from src.core.settings import load_settings
from src.libs.vector_store.chroma_store import ChromaStore

print("="*60)
print("测试ChromaStore读取portfolio集合")
print("="*60)

# 加载配置
settings = load_settings()
print(f"\n配置加载成功")
print(f"Embedding模型: {settings.embedding.model}")
print(f"Embedding维度: {settings.embedding.dimensions}")

# 创建ChromaStore实例
try:
    print("\n初始化ChromaStore...")
    store = ChromaStore(settings, collection_name="portfolio")

    print(f"✓ ChromaStore初始化成功")
    print(f"  集合名称: {store.collection_name}")
    print(f"  数据条数: {store.collection.count()}")

    # 测试查询
    print("\n测试查询...")
    # 这里需要先生成embedding向量
    # 暂时只测试初始化是否成功

except Exception as e:
    print(f"\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
