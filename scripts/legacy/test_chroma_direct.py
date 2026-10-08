"""
在MCP服务器环境中，使用相同的配置测试ChromaDB
运行: cd D:/PyCharm/pytest/实习/MODULAR-RAG-MCP-SERVER-main && .venv/Scripts/python.exe test_chroma_direct.py
"""
import os
os.environ["CHROMA_PERSIST_DIR"] = "./data/db/chroma"

import chromadb
from pathlib import Path

persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/db/chroma")
print(f"数据库路径: {persist_dir}")
print(f"绝对路径: {Path(persist_dir).resolve()}")

# 使用与MCP服务器相同的方式连接
client = chromadb.PersistentClient(path=persist_dir)

# 列出集合
collections = client.list_collections()
print(f"\n集合列表: {[c.name for c in collections]}")

for col in collections:
    count = col.count()
    print(f"  {col.name}: {count} 条数据")

    if col.name == "portfolio":
        # 查看数据
        print(f"\nportfolio集合详情:")
        data = col.peek(limit=3)
        print(f"  IDs: {data['ids'][:3]}")
        print(f"  Documents:")
        for i, doc in enumerate(data['documents'][:3], 1):
            print(f"    {i}. {doc[:60]}...")

        # 测试查询
        print(f"\n测试查询 'RAG是什么':")
        results = col.query(
            query_texts=["RAG是什么"],
            n_results=3
        )
        print(f"  查询结果: {len(results['documents'][0])} 条")
        for i, doc in enumerate(results['documents'][0], 1):
            print(f"    {i}. {doc[:60]}...")
