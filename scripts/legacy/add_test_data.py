"""
使用MCP服务器添加测试数据
运行命令: cd D:/PyCharm/pytest/实习/MODULAR-RAG-MCP-SERVER-main && .venv/Scripts/python.exe add_test_data.py
"""

import os
import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions

# 连接到MCP服务器的数据库
db_path = Path("./data/db/chroma")
print(f"数据库路径: {db_path}")

if not os.environ.get("DASHSCOPE_API_KEY"):
    raise RuntimeError("DASHSCOPE_API_KEY is required; never hardcode API credentials")

# 使用与MCP服务器相同的embedding配置
embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("DASHSCOPE_API_KEY"),
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen3.7-text-embedding-flash"
)

print("连接数据库...")
client = chromadb.PersistentClient(
    path=str(db_path),
    settings=chromadb.Settings(anonymized_telemetry=False)
)

# 删除旧的portfolio集合（如果存在）
try:
    client.delete_collection("portfolio")
    print("删除旧的portfolio集合")
except:
    print("portfolio集合不存在，跳过删除")

# 创建portfolio集合（使用与ChromaStore一致的metadata）
print("\n创建portfolio集合...")
collection = client.create_collection(
    name="portfolio",
    metadata={"hnsw:space": "cosine"},  # 与ChromaStore一致
    embedding_function=embedding_function
)

# 添加测试文档
documents = [
    "我叫张三，是一名AI工程师，有3年的Python开发经验。",
    "我熟悉的技术栈包括：Python、FastAPI、Django、RAG系统、向量数据库ChromaDB。",
    "我的项目经验包括：开发过RAG增强的对话系统，集成了MCP协议实现模块化检索。",
    "Python是一种高级编程语言，广泛应用于Web开发、数据科学、机器学习等领域。",
    "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的AI技术，可以让LLM基于知识库回答问题。",
    "我参与过EchoMind项目的开发，这是一个基于记忆增强的AI助手系统。",
    "我擅长使用FastAPI构建高性能的Web API，并集成LLM进行智能对话。",
    "我有向量数据库的使用经验，包括ChromaDB、Milvus等，用于语义检索。"
]

metadatas = [
    {"source": "resume", "type": "personal_info"},
    {"source": "resume", "type": "skills"},
    {"source": "resume", "type": "projects"},
    {"source": "knowledge", "type": "tech"},
    {"source": "knowledge", "type": "tech"},
    {"source": "resume", "type": "projects"},
    {"source": "resume", "type": "skills"},
    {"source": "resume", "type": "experience"}
]

ids = [f"doc_{i}" for i in range(len(documents))]

print(f"\n添加 {len(documents)} 条文档...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

# 验证
count = collection.count()
print(f"OK 添加完成，集合中共有 {count} 条文档")

# 测试查询
print("\n测试查询...")
results = collection.query(
    query_texts=["介绍一下你自己"],
    n_results=3
)

print(f"查询到 {len(results['documents'][0])} 条结果:")
for i, doc in enumerate(results['documents'][0], 1):
    print(f"  [{i}] {doc[:60]}...")

print("\n✓ 数据添加成功！")
print("\n现在可以运行 EchoMind 的 Agent 测试了。")
