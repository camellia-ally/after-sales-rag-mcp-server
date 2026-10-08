# RAG 内部答案生成层技术设计

## 1. 设计目标

在不重写现有摄取和检索链路的前提下，为项目补充内部答案生成能力：

```text
用户问题
→ HybridSearch 检索
→ 将 Top-K 片段组织成上下文
→ 现有 BaseLLM.chat()
→ 返回答案和真实来源
```

第一步先完成最小可运行闭环。引用校验、证据充分性判断和生成评估在闭环跑通后逐项增强，不在第一版同时拆成多个独立组件。

## 2. 现有能力与缺口

### 可以直接复用

| 现有能力 | 作用 |
|---|---|
| `HybridSearch.search()` | 根据问题返回排序后的 `RetrievalResult` |
| `RetrievalResult` | 提供 Chunk 正文、ID、分数和元数据 |
| `LLMFactory.create()` | 根据 `settings.yaml` 创建当前配置的 LLM |
| `BaseLLM.chat()` | 使用统一接口调用不同模型供应商 |
| `CitationGenerator` | 从真实检索结果生成结构化来源 |
| `TraceContext` | 记录检索和生成阶段 |

### 当前缺口

`scripts/query.py` 和 MCP 的 `query_knowledge_hub` 当前只返回检索片段。`ResponseBuilder` 负责把片段格式化为 MCP 响应，不负责让 LLM 综合生成最终答案。

## 3. 最小模块边界

第一版只新增一个核心服务，概念接口如下：

```python
class RAGAnswerService:
    def __init__(self, hybrid_search, llm, citation_generator): ...

    def answer(
        self,
        query: str,
        top_k: int = 5,
        trace: TraceContext | None = None,
    ) -> RAGAnswerResponse: ...
```

职责：

1. 调用 `HybridSearch.search()`；
2. 为候选片段分配 `[1]`、`[2]` 等编号；
3. 把问题和编号后的片段组成 LLM 消息；
4. 调用 `BaseLLM.chat()`；
5. 将答案、引用和运行元数据组装为统一结果；
6. 记录检索、上下文构建和 LLM 生成阶段。

暂时不负责：

- 创建向量库、Embedding 或 BM25 组件；
- 解析 PDF 或执行摄取；
- 实现多轮对话；
- 实现 Agent 循环；
- 提供 Web 页面；
- 直接输出 MCP 协议对象。

## 4. 为什么放在 Core 层

CLI、MCP 和未来的 HTTP API 都只是入口：

```text
CLI ─┐
MCP ─┼→ RAGAnswerService → HybridSearch + LLM
API ─┘
```

如果把生成逻辑直接写进 `scripts/query.py`，MCP 无法复用；写进 MCP Tool，CLI 无法复用；写进 `ResponseBuilder`，又会把业务生成和 MCP 格式化混在一起。

因此答案服务应位于 Core 层，入口只负责接收参数和展示结果。

## 5. 第一版数据流

```text
1. answer(query, top_k)
2. HybridSearch.search(query, top_k)
3. 没有结果：返回固定的资料不足提示
4. 有结果：构造编号上下文
5. llm.chat(system_message + user_message)
6. 从真实 RetrievalResult 生成来源
7. 返回 RAGAnswerResponse
```

上下文概念格式：

```text
[1] chunk_id=...
来源=...
正文=...

[2] chunk_id=...
来源=...
正文=...
```

LLM 只看到程序分配的引用编号。文件名、Chunk ID、页码等字段由程序从 `RetrievalResult` 读取，不能让模型自行编造。

## 6. 第一版输出契约

```python
@dataclass
class RAGAnswerResponse:
    status: str              # answered | refused | error
    answer: str
    citations: list[Citation]
    metadata: dict
```

第一版 `metadata` 至少包含：

- `query`；
- `retrieved_count`；
- `model`；
- `usage`（供应商返回时记录）；
- `stop_reason`。

不能把检索异常伪装为拒答：

- 没有相关资料 → `refused`；
- Embedding、向量库或 LLM 调用失败 → `error`。

## 7. 递增实现顺序

### V1：跑通生成闭环

检索 Top-K，组织上下文，调用现有 LLM，返回答案和来源。

### V2：引用约束

要求模型在结论后使用 `[n]`；程序只展示答案实际使用且编号有效的引用。无效编号不得直接返回。

### V3：资料不足拒答

MVP采用一次调用的固定拒答标记：Prompt要求资料不能直接回答核心问题时只输出`INSUFFICIENT_EVIDENCE`，程序将其转换为统一拒答。该方案简单且不增加第二次模型调用，但仍依赖模型判断，作为已知限制记录；只有评估证明不可靠时再比较独立Judge或相关性阈值。

### V4：入口复用

新增答案 CLI，并让 MCP 根据产品决策调用同一答案服务；保留原始检索工具用于调试或 Agent 自主读取证据。

### V5：质量评估

评估 Faithfulness、Answer Relevancy、Citation Validity/Coverage、拒答准确率、延迟、Token 和成本。

## 8. 第一版实现位置（设计建议）

以下路径在编码任务开始前最终确认：

```text
src/core/answering/
├── __init__.py
├── types.py              # RAGAnswerResponse
└── rag_answer_service.py # 最小编排服务

tests/unit/core/answering/
└── test_rag_answer_service.py

scripts/
└── answer.py             # 后续CLI入口
```

第一版不创建 `EvidenceJudge`、`AnswerGenerator`、`CitationValidator` 三个独立文件。只有当职责变复杂、需要独立替换或独立测试时才提取。

## 9. 测试边界

先使用 Mock LLM 验证确定性逻辑：

1. 有检索结果时，LLM 收到问题和编号上下文；
2. 无检索结果时，不调用 LLM，返回 `refused`；
3. 检索失败时返回/抛出明确错误，不能显示为“知识库无答案”；
4. LLM 失败时保留错误阶段；
5. 引用来源只来自本次真实检索结果；
6. Trace 中能看到 retrieval、context_building 和 answer_generation。

Mock 测试通过后，再用真实 Qwen 运行少量端到端样例。

## 10. 当前待验证问题

进入编码前还需确认：

1. `RAGAnswerResponse` 是新建通用类型，还是扩展当前 MCP 专用响应；
2. V1 是否立即要求答案包含 `[n]` 标记；
3. 当前 Qwen 对编号引用和 JSON 输出的稳定性；
4. 答案 Prompt 是先作为代码常量完成探针，还是直接配置化；
5. MCP 是新增 `answer_knowledge_hub`，还是改变现有 `query_knowledge_hub` 的语义。

建议：保留现有检索 Tool 的语义，后续新增答案 Tool，避免破坏已经通过的 MCP E2E 契约。

## 11. 设计验收

进入代码实现前，学习者应能说明：

1. `HybridSearch` 为什么不直接调用 LLM；
2. `RAGAnswerService` 为什么不写在 CLI 或 MCP Tool 中；
3. `LLMFactory` 在新流程中负责什么、不负责什么；
4. 为什么引用元数据由程序生成，而不是交给 LLM；
5. 为什么先做最小闭环，再根据测试结果提取更多组件。
