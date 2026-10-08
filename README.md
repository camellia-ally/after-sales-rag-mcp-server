# 售后客服 RAG 问答系统

这是在 [Modular RAG MCP Server](https://github.com/jerry-ai-dev/MODULAR-RAG-MCP-SERVER) 基础上扩展的学习项目。当前实例使用公开售后帮助文档，支持文档摄取、Dense + BM25 混合检索、RRF 融合、重排、带引用的答案生成、拒答评测与 MCP 工具调用。原项目的说明保存在 [上游 README](docs/archive/README_upstream.md)；本页描述此工作区的实际入口和结果。

## 当前能力与边界

| 链路 | 入口 | 当前行为 |
| --- | --- | --- |
| 文档摄取 | `scripts/ingest.py` | 支持 PDF、TXT、Markdown；建立向量与 BM25 索引 |
| 检索 | `scripts/query.py` | Dense、BM25、RRF，可选重排 |
| 答案 | `scripts/answer.py` | 根据检索证据生成答案、校验引用编号、资料不足时拒答；可选重排相关度门槛 |
| MCP | `python -m src.mcp_server.server`、`mcp-server` | stdio 工具服务，提供知识查询、文档摄取与集合相关工具；MCP 查询工具与答案 CLI 是不同入口 |
| 可视化 | `scripts/start_dashboard.py` | Streamlit 数据、追踪与评测页面 |
| 售后评测 | `scripts/evaluate_jd_full.py` | 检索、重排和拒答对照；调用真实 API |

## 环境与运行

需要 Python 3.10 及以上。在 PowerShell 中：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

使用 [配置模板](.github/skills/setup/references/settings_template.yaml) 或 Setup Skill 创建私有的 `config/settings.yaml`，填写自己的 Provider、模型、存储路径和凭证。该配置及 `.env` 被 Git 忽略，不要把密钥写入脚本或报告。现有用户可沿用已配置的本地环境。

```powershell
# 先列出准备摄取的文件；正式摄取会调用 Embedding API
python scripts/ingest.py --path data/documents/jd_after_sales --collection jd_after_sales --dry-run

# 检索、答案和服务入口
python scripts/query.py --query "质量问题退货，运费谁承担？" --collection jd_after_sales
python scripts/answer.py --query "质量问题退货，运费谁承担？" --collection jd_after_sales
python -m src.mcp_server.server
```

MCP 使用 stdio，服务运行时不要向 stdout 打印诊断信息。安装项目后也可使用 `mcp-server`。Dashboard 通过 `python scripts/start_dashboard.py` 启动。

## 售后场景评测

评测集包含 **50 道可回答检索题**（关键词 17、语义 17、跨文档 16），另有 **10 道不可回答题和 6 道可回答对照题**。一键脚本和指标定义见 [评测说明](JD_AFTER_SALES_EVAL.md)：

```powershell
python scripts/evaluate_jd_full.py --dry-run
python scripts/evaluate_jd_full.py
```

当前[完整报告](reports/jd_after_sales_full_pilot.json)记录：纯向量与 RRF 的证据 Chunk Hit@5 均为 46/50；RRF + Qwen 重排为 49/50、Top 1 为 47/50，跨文档全部证据找齐 15/16。拒答基线为 10/10，可回答对照题误拒答 0/6。不能据此声称 RRF 的总体 Hit@5 高于纯向量。标签状态为 `agent_draft`；[复核清单](reports/jd_after_sales_label_review_v3.md)保留了来源证据。

可选相关度门槛作用于**重排分数**，不是 RRF 分数，也不代替模型判断资料是否足够。阈值 `0.60` 的[拒答报告](reports/jd_refusal_threshold_060.json)显示：10 道不可回答题全部拒答，其中 4 道在生成前被门槛拦截；6 道可回答题无误拒答。此阈值仅在当前售后题集上验证，使用时显式传入：

```powershell
python scripts/answer.py --query "今天天气怎么样？" --collection jd_after_sales --min-rerank-score 0.6
```

## 文档与维护

- [DEV_SPEC.md](DEV_SPEC.md)：上游的大型设计规格与原始排期，属于设计资料；当前实现以代码和本页为准。
- [LEARNING_PLAN.md](LEARNING_PLAN.md)、[RAG_PROJECT_PLAN.md](RAG_PROJECT_PLAN.md)：学习过程与阶段计划，部分阶段状态早于本次售后扩展。
- [RAG_ANSWER_SPEC.md](RAG_ANSWER_SPEC.md)、[RAG_ANSWER_DESIGN.md](RAG_ANSWER_DESIGN.md)：答案行为与架构设计。
- [JD_AFTER_SALES_EVAL.md](JD_AFTER_SALES_EVAL.md)：评测命令、题集和指标解释。
- `scripts/legacy/`：保留的一次性实验与修复脚本，不作为日常入口。

继续开发时按“需求与验收标准 → 当前行为与数据 → 最小改动 → 针对性测试 → 真实场景验证 → 文档与指标更新”的顺序进行。单元测试可用 `python -m pytest tests/unit -q`；外部 API 与 MCP 端到端测试需单独运行。

当前已知维护事项：`SparseEncoder` 使用 jieba 切分英文连字符与下划线，两个旧单元测试要求保留复合词，现有行为与测试预期不一致。修改分词会改变 BM25 索引，需先确定规则、重建索引并重跑检索评测；本次文档与入口维护没有改动该算法。仓库整体仍有较多历史 Ruff 格式问题，本次没有批量改写无关模块。`.gitignore` 忽略 `data/` 和 `logs/`，因此本地文档和索引不会随源码提交。此工作区的 Git 仓库尚无提交，正式对外整理前应先检查待提交文件，确保不包含私有配置或凭证。
