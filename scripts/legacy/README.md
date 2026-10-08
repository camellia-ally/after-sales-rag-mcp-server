# 历史一次性脚本

本目录保留早期调试和数据准备脚本，便于回溯；它们不是当前项目的标准运行入口，也没有纳入本次回归测试。

| 文件 | 原用途 | 注意 |
| --- | --- | --- |
| `add_test_data.py` | 写入临时简历数据 | 会重建 `portfolio` 集合；已移除硬编码凭证，仍不应在现有数据库上随意运行 |
| `fix_chromadb.py` | 旧版本兼容性应急修复 | 会改动依赖并删除数据库，不适用于当前环境 |
| `convert_to_docs.py` | 临时 Markdown 转 PDF/Word | 依赖额外包和本机字体 |
| `test_chroma_direct.py`、`test_chromastore.py` | 手工数据库排错 | 使用固定本地路径或旧配置，不能代替自动测试 |
| `analyze_threshold.py` | 早期分数实验 | 固定题目和旧集合；当前评测入口为 `scripts/evaluate_jd_after_sales.py --mode scores` |
| `test_rag_comparison*.py`、`test_rag_evaluation.py` | 早期简历指标实验 | 固定示例题、调用路径与当前评测集不一致；已移出 `tests/`，避免被日常 pytest 收集 |

正式入口见仓库根目录的 [README](../../README.md)。
