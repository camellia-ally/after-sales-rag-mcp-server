"""
修复MCP服务器ChromaDB版本兼容问题
步骤：
1. 卸载ChromaDB 1.5.9
2. 安装ChromaDB 0.4.24
3. 删除旧数据库
4. 重新创建数据库并添加数据

运行: cd D:/PyCharm/pytest/实习/MODULAR-RAG-MCP-SERVER-main && .venv/Scripts/python.exe fix_chromadb.py
"""

import subprocess
import sys
import shutil
from pathlib import Path

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n[{description}]")
    print(f"命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result.returncode == 0:
        print(f"OK 成功")
        if result.stdout:
            print(result.stdout[:200])
    else:
        print(f"X 失败")
        print(result.stderr[:500])
        return False
    return True

def main():
    print("="*60)
    print("修复MCP服务器ChromaDB版本兼容性")
    print("="*60)

    python_exe = sys.executable

    # 步骤1：卸载旧版本
    print("\n步骤1: 卸载ChromaDB 1.5.9...")
    if not run_command(
        [python_exe, "-m", "pip", "uninstall", "-y", "chromadb"],
        "卸载ChromaDB"
    ):
        print("卸载失败，继续...")

    # 步骤2：安装0.4.24
    print("\n步骤2: 安装ChromaDB 0.4.24...")
    if not run_command(
        [python_exe, "-m", "pip", "install", "chromadb==0.4.24"],
        "安装ChromaDB 0.4.24"
    ):
        print("安装失败！")
        return

    # 步骤3：删除旧数据库
    db_path = Path("./data/db/chroma")
    if db_path.exists():
        print(f"\n步骤3: 删除旧数据库 {db_path}...")
        try:
            shutil.rmtree(db_path)
            print("✓ 已删除")
        except Exception as e:
            print(f"✗ 删除失败: {e}")
            return
    else:
        print(f"\n步骤3: 数据库目录不存在，跳过")

    # 步骤4：创建新数据库
    print("\n步骤4: 创建新数据库并添加测试数据...")

    import chromadb

    print(f"  ChromaDB版本: {chromadb.__version__}")

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=chromadb.Settings(anonymized_telemetry=False)
    )

    # 创建portfolio集合
    collection = client.create_collection(
        name="portfolio",
        metadata={"description": "个人简历和项目信息"}
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

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    count = collection.count()
    print(f"  ✓ 添加完成，集合中共有 {count} 条文档")

    # 步骤5：验证
    print("\n步骤5: 验证查询功能...")
    results = collection.query(
        query_texts=["RAG是什么"],
        n_results=3
    )

    print(f"  查询到 {len(results['documents'][0])} 条结果")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"    [{i}] {doc[:50]}...")

    print("\n" + "="*60)
    print("✓ 修复完成！")
    print("="*60)
    print("\n现在可以重新测试EchoMind的Agent系统了。")

if __name__ == "__main__":
    main()
