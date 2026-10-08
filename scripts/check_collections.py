"""检查 ChromaDB 中的 collection"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings as ChromaSettings

chroma_client = chromadb.PersistentClient(
    path="./data/db/chroma",
    settings=ChromaSettings(anonymized_telemetry=False)
)

collections = chroma_client.list_collections()
print(f"找到 {len(collections)} 个 collection:")
for col in collections:
    print(f"  - {col.name} (count: {col.count()})")
