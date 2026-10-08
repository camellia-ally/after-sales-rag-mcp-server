# Modular RAG MCP Server 学习与求职计划

> 状态说明：这是按阶段积累的学习记录，部分“当前任务”描述早于售后场景扩展。当前可运行入口和评测结果见 [README.md](README.md)；本次维护不改动历史验收状态。

## 1. 最终目标

目标岗位：大模型应用开发 / RAG / Agent 开发实习。

本计划不是为了把仓库命令跑一遍，而是让学习者最终能够：

1. 讲清原项目做了什么、为什么这样设计；
2. 独立定位 RAG、LLM、MCP 链路中的问题；
3. 按职业开发流程完成需求、设计、实现、测试和评估；
4. 在原项目上做出有指标、有测试、有个人决策的扩展；
5. 把项目整理成可演示、可写入简历、经得住追问的作品。

## 当前交付范围冻结（2026-09-19）

当前版本只完成以下目标，不再主动增加范围外能力：

### 第二阶段：从演示升级到生产质量

1. BM25 + Dense Embedding 混合检索；
2. RRF 融合排序；
3. Cross-Encoder 重排序，并用当前中文资料验证效果；
4. 内部 LLM 答案生成；
5. 资料不足时基础拒答；
6. 回答引用真实 Chunk，并校验引用编号；
7. Prompt 保存为带版本的配置文件。

### 第三阶段：让系统具备交付能力

1. 从候选草稿中人工筛选 39 题高置信度 Golden Test Set，确认参考答案或证据；首版覆盖可回答、部分可回答和不可回答三类场景；
2. 离线评估检索指标与 Faithfulness；
3. 将稳定、无需真实付费 API 的评估接入 CI；
4. 设置有基线依据的质量门槛，低于门槛时 CI 失败；
5. 整理 README、运行方式、评估结果、已知限制和面试表达。

### 明确不在当前交付范围

- LangGraph 或自研 Agent 循环；
- 独立 Evidence Judge、NLI 模型或复杂多阶段拒答；
- 查询改写、二次检索和多 Agent；
- 新增 Web 聊天页面；
- 扩展新的 MCP Tool；
- 与上述两阶段无直接关系的架构重构。

只有截图目标全部完成并验收后，才重新讨论这些扩展；Agent 不得自行加入。

### 当前完成情况

| 目标 | 状态 | 证据或下一步 |
|---|---|---|
| BM25 + Dense + RRF | 已完成 | 真实中文 PDF 查询、Trace、Hit@5/MRR 基线 |
| 内部 LLM 答案生成 | 已完成 | `scripts.answer` 真实调用 `qwen-flash`成功 |
| 引用编号校验 | 已完成 | 回归测试通过；真实回答只展示实际使用的引用；非法编号被拦截 |
| 基础拒答 | 已完成 | 回归测试和真实无答案问题验收通过，固定返回 `refused` |
| Reranker | 已完成初步验收 | `qwen3.7-text-rerank` 接入成功；5题A/B中 Hit@5 保持1.00，MRR从0.90提升到1.00，平均增加371ms |
| Prompt 版本化配置 | 已完成 | `config/prompts/rag_answer_v1.txt`；响应与Trace记录版本 |
| 39题 Golden Set | 已完成 | `tests/fixtures/golden_test_set_portfolio.json`；29道可回答、5道部分可回答、5道不可回答，覆盖中英文资料 |
| 重复 Chunk 去重 | 暂缓 | 已确认存在重复Chunk，但当前日常使用场景未证明其影响足以进入本次课程交付；保留为后续优化候选 |
| Faithfulness 离线评估 | 已完成基线 | 34个已回答问题平均Faithfulness为0.9485；状态与引用规则指标均为1.0，已完成Bad Case分析 |
| CI质量门禁 | 课程交付后续 | 当前先保留正式评测命令和报告；若作为长期求职作品，再增加无付费API的检索回归门禁 |
| README与求职材料 | 当前收口 | 课程交付先整理README、运行命令、指标和已知限制；简历与模拟面试随后单独完成 |

## 2. 教学与开发规则

每个任务开始前，Agent 必须先说明：

- **做什么**：本次任务的明确边界；
- **为什么**：它解决什么问题，处于开发流程的哪一步；
- **你要理解什么**：学习者需要掌握的职业能力；
- **产出**：完成后留下什么文件、代码、测试或指标；
- **验收**：用什么证据判断完成，而不是只看代码是否存在。

其他规则：

- 以仓库真实代码为准，区分“已有能力”“设计目标”“计划新增”。
- 不逐行讲代码，重点讲模块职责、数据流、接口、取舍和调试方法。
- 能由学习者运行的命令，优先让学习者亲自运行并观察输出。
- 修改代码前先完成必要的规格和技术设计，避免边写边猜。
- 不重复创建职责相同的文档；已有内容优先引用或更新。
- 每次实验只改变一个主要变量，并保留基线和对比结果。
- 每阶段结束由 Agent 提出 3～5 道面试题；回答合格后才更新为完成。
- 不读取、输出、截图或提交 API Key。
- 不再把“补充单元测试”作为独立学习任务，也不为数量堆测试；现有单元测试保留。当前改进优先使用 Golden Set、A/B 指标和 CI 回归证明价值，只有关键接口或失败保护确有需要时才补最小测试。

### 改进任务的求职验收模板

任何算法、架构或可靠性改进开始前，必须先讲清并记录：

1. **问题证据**：真实输出、Trace、指标或测试说明当前哪里不好；
2. **改进目标**：希望改善哪个行为或指标，不能只说“优化一下”；
3. **方案合理性**：为什么该方案能解决问题，依赖什么假设；
4. **备选方案取舍**：至少比较一个更简单或更常见的方案，并说明没有选择它的原因；
5. **核心测试**：只保留能证明主要收益、失败保护和关键回归的测试，不为数量堆测试；
6. **前后对比**：在同一问题集和配置下比较修改前后行为、质量、延迟和成本；
7. **面试表达**：能够用“问题—分析—方案—取舍—结果”的方式独立讲述。

代码运行成功只说明实现可执行，不等于改进有效。没有前后证据和核心测试，不得把改进标记为完成或写入简历。

## 3. 文档与代码的职责链

```text
需求与行为规则
RAG_ANSWER_SPEC.md
        ↓
技术架构、模块边界、接口和异常策略
RAG_ANSWER_DESIGN.md（阶段四当前要产出的文档）
        ↓
LLM执行指令
config/prompts/*（只有设计确认后才创建）
        ↓
业务实现
src/*
        ↓
规格验证与质量证据
tests/* + Golden Set + 评估报告
```

文件职责：

| 内容 | 职责 | 不负责什么 |
|---|---|---|
| `RAG_ANSWER_SPEC.md` | 定义回答、部分回答、拒答、引用和异常时应有什么行为 | 不决定类名、文件路径和具体代码 |
| `RAG_ANSWER_DESIGN.md` | 定义模块、接口、数据契约、调用时序、配置和测试边界 | 不重复描述产品背景 |
| `config/prompts/*` | 给 LLM 的版本化执行指令 | 不代替 Spec 或技术设计 |
| `src/*` | 实现技术设计 | 不自行发明未确认的行为 |
| `tests/*` 与评估集 | 验证代码是否满足 Spec，并提供指标证据 | 不通过降低标准掩盖实现问题 |

> `answer_generation.txt` 当前不创建。若技术设计最终决定 Prompt 配置化，再创建对应 Prompt 文件；它是实现资源，不是第二份答案规格。

## 4. 六阶段路线

| 阶段 | 主题 | 求职能力 | 状态 |
|---|---|---|---|
| 一 | 真正读懂原项目 | 能讲清目录、摄取链路、查询链路和设计取舍 | 已完成 |
| 二 | 掌握 RAG 核心技术 | 能运行、调试、评估 Hybrid Search，并分析 Bad Case | 已通过验收 |
| 三 | 掌握 MCP 与 Agent 基础 | 能解释 Tool、MCP、Skill、Agent 的边界并验证工具调用 | MCP 基础已完成；Agent 实现留到阶段五 |
| 四 | 把 RAG 做成完整应用 | 新增可靠答案生成、引用、拒答和生成评估 | **当前阶段** |
| 五 | Agent 集成与 Agent 协作开发 | 在可靠 RAG 上增加决策循环，并掌握 Spec 驱动的 AI 协作开发 | 未开始 |
| 六 | 求职交付 | 形成演示、指标、README、简历和面试材料 | 未开始 |

---

## 5. 已完成阶段

### 阶段一：真正读懂原项目

已掌握：

- `config/`、`src/`、`scripts/`、`tests/`、`data/` 的职责；
- Document → Chunk → Embedding → ChromaDB/BM25 的摄取链路；
- QueryProcessor → Dense/Sparse → RRF → 可选 Reranker 的查询链路；
- 文件哈希、稳定 ID、增量摄取、分块和文本清理；
- collection、Trace、Dashboard、Factory 和配置驱动设计。

完成证据：PDF Loader、分块实验、真实 Embedding 调用和摄取链路均已运行。

### 阶段二：掌握 RAG 核心技术

已完成：

- 真实 PDF 摄取和 Hybrid Search 查询；
- Dense、BM25、RRF 输出与 Trace 分析；
- 5 题 Golden Test Set；
- 自动基线：`Hit@5 = 1.00`，`MRR = 0.90`；
- 修复 `CustomEvaluator` 与 `RetrievalResult.chunk_id` 的接口问题；
- 分析负 IDF、重复 Chunk 和 Embedding 批处理错误传播问题。

阶段面试验收：2026-09-19 通过。

### 阶段三：掌握 MCP 与 Agent 基础

已完成：

- 理解 MCP 是 Agent 调用外部能力的协议，Skill 是 Agent 的工作说明；
- MCP `initialize + tools/list` E2E 通过；
- `tools/call query_knowledge_hub` E2E 通过；
- Trace 确认调用真实进入 Query Processing、Dense、Sparse 和 Fusion；
- 将 MCP 依赖限制为 `mcp>=1.0.0,<2.0.0`，保证当前实现可复现。

尚未做：自研业务 Agent。原因是 Agent 应建立在可靠的完整 RAG 服务上，而当前项目还缺答案生成、引用和拒答层。

---

## 6. 当前阶段：把检索系统做成完整 RAG 应用

### 阶段目标

当前项目已经能“找到片段”，本阶段要让它能“基于片段可靠回答”：

```text
用户问题
→ HybridSearch
→ Evidence Judge
├─ 证据不足：安全拒答
└─ 证据充分/部分充分
   → Answer Generator
   → Citation Validator
   → 答案 + 真实引用
```

本阶段不是简单调用一次 LLM。核心能力是控制证据、约束生成、验证引用、处理失败，并用测试证明系统不会基于弱相关内容乱答。

### A0：答案功能规格

- **做什么**：定义充分回答、部分回答、拒答、引用、异常和验收场景。
- **为什么**：先确定系统“应该做什么”，避免实现人员或 Agent 自行猜需求。
- **你要理解什么**：Spec 描述外部行为，不等于 Prompt，也不等于代码设计。
- **产出**：`RAG_ANSWER_SPEC.md`。
- **验收**：规格覆盖有答案、无答案、部分答案、引用无效和检索异常。
- **状态**：已完成。

### A1：现有接口盘点与技术设计

- **做什么**：检查 `HybridSearch.search`、`RetrievalResult`、`LLMFactory/BaseLLM.chat`、`CitationGenerator`、`ResponseBuilder`、`TraceContext` 的真实接口；确定新模块的边界、数据流和调用者。
- **为什么**：新增答案层必须复用现有检索、LLM、引用和 Trace 能力，不能把生成逻辑塞进检索器，也不能另造一套重复基础设施。
- **你要理解什么**：
  - 为什么检索与生成要分层；
  - 为什么 CLI、MCP 和未来 API 应调用同一个答案服务；
  - 为什么 LLM 只返回受约束的判断和引用编号；
  - 为什么文件名、Chunk ID 等引用元数据必须由程序生成。
- **产出**：`RAG_ANSWER_DESIGN.md`，包含模块图、接口、数据契约、时序、异常策略和测试边界。
- **验收**：文档与真实接口一致；上述四个“为什么”能够独立解释；尚未确定的问题明确标注为技术探针，而不是假装已定案。
- **状态**：已完成（`RAG_ANSWER_DESIGN.md` 已确认采用渐进实现）。

### A2：V1 最小答案服务与 Mock 测试

- **做什么**：先写行为测试，再实现一个 `RAGAnswerService`，串联检索、编号上下文、`BaseLLM.chat()` 和统一响应。
- **为什么**：先证明最小生成闭环成立，避免在还没有真实需求证据时拆出大量组件。
- **你要理解什么**：依赖注入、Mock LLM、上下文构建以及“无结果”和“系统失败”的区别。
- **产出**：答案响应类型、最小答案服务和单元测试。
- **验收**：有结果时 LLM 收到真实编号上下文；无结果时不调用 LLM；检索或 LLM 异常不会被伪装成拒答；测试通过。
- **状态**：已完成。证据：`tests/unit/test_rag_answer_service.py` 三个场景全部通过（真实检索上下文传递、空结果不调用LLM、检索异常不伪装为拒答）。

### A3：V1 真实 LLM 与答案 CLI

- **做什么**：通过现有 `LLMFactory` 创建 Qwen，新增答案 CLI，运行一次真实“检索 → 生成”。
- **为什么**：Mock 只能证明程序逻辑，真实模型验证配置、消息格式、Token 和响应契约。
- **你要理解什么**：配置选择模型、Factory 创建实例、答案服务调用模型是三层不同职责。
- **产出**：`scripts/answer.py`、真实回答、Trace 和调用数据。
- **验收**：中文问题得到基于知识库的回答；能在 Trace 中区分检索和生成耗时；原有查询脚本不受影响。
- **状态**：已完成。证据：真实 `default` Collection 成功完成 Hybrid Search → Qwen `qwen-flash` → 带引用回答；原 `scripts.query` 回归正常；Dashboard Trace 可区分检索、上下文构建与答案生成阶段。

### A4：引用约束、MVP拒答与 MCP 复用

- **做什么**：校验引用编号；MVP使用单次LLM调用和固定`INSUFFICIENT_EVIDENCE`标记实现基础拒答；让 MCP 复用同一答案服务。
- **为什么**：先完成可展示的可靠性基线，避免在小型项目中提前引入独立Judge、NLI和未经标定的分数阈值。
- **你要理解什么**：检索 Top-K 不等于存在答案；固定拒答标记能降低风险但不能完全消除幻觉；更复杂方案必须由评估数据驱动。
- **产出**：引用校验、拒答逻辑、对应测试和答案 MCP Tool。
- **验收**：有答案问题引用真实 Chunk；无答案问题可靠拒答；非法引用不会直接返回；CLI 与 MCP 核心行为一致。
- **状态**：核心能力已完成（CLI答案生成、引用校验和基础拒答已验收）。现有MCP继续提供知识检索结果；新增答案型MCP Tool不在当前三项优先任务内。

### A5：正式 Golden Set、完整质量评估、去重实验与质量门禁

- **做什么**：从候选草稿中筛选39题高置信度 Golden Set，覆盖 answerable、partial 和 unanswerable，并在同一数据集上评估检索质量（Hit@K、MRR）、生成忠实度（Faithfulness）、引用和拒答；随后对重复正文 Chunk 做单变量去重实验，最后把稳定、无需付费API的质量检查接入 CI。
- **为什么**：5题只能证明链路和初步趋势，不能支撑最终简历指标；检索命中不等于生成答案忠实；重复Chunk会挤占Top-K、Rerank输入和LLM上下文；CI必须建立在稳定数据与可信基线上。
- **你要理解什么**：Golden Set是评估输入和人工真值，不是指标结果本身；`expected_chunk_ids`用于Hit/MRR，标准证据与系统答案用于Faithfulness，可回答性标签用于拒答评估；为什么一次实验只改变一个变量。
- **产出**：39题人工标注集、检索与生成质量基线报告、去重前后A/B报告、CI配置和质量门槛说明。
- **验收**：题型和资料来源有明确分布；报告包含Hit@K、MRR、Faithfulness、引用有效性和拒答准确率；去重后正确证据召回不下降且重复率/成本有改善；CI能在稳定指标低于基线阈值时失败，且默认不调用真实付费API。
- **状态**：课程核心目标已完成，进入交付收口。重复Chunk去重暂缓，不作为本次交付阻塞项。

执行顺序：

```text
任务1：完成39题人工真值标注并生成正式Golden Set
→ 任务2：运行检索、Faithfulness、引用和拒答基线评估
→ 任务3：整理README、运行方式、评估结果和已知限制
→ 任务4：课程交付后再决定是否做去重A/B与CI门禁
```

### A5 实施记录与求职证据

#### 1. 正式检索基线

- **评测输入**：39题Golden Set中的29道`answerable`与5道`partial`，共34题；5道`unanswerable`没有正确Chunk，不参与Hit@5/MRR计算。
- **生产链路**：Dense Embedding + BM25 + RRF + `qwen3.7-text-rerank`，最终Top 5。
- **结果**：`Hit@5=0.9412`，`MRR=0.9118`；30题正确证据排第1，2题排第2，2题未命中；总耗时90.865秒，单题平均约2.673秒。
- **分类型结果**：answerable共29题，`Hit@5=0.9655`、`MRR=0.9310`；partial共5题，`Hit@5=0.8000`、`MRR=0.8000`。
- **Bad Case**：`portfolio_q29`与`portfolio_q54`均为中文问题检索英文论文；q54还是同时询问评测维度与特定数据集结果的复合问题。正确英文Chunk未进入最终Top 5，说明当前主要短板是跨语言与复杂查询召回，不是指标程序故障。
- **正式证据**：`reports/portfolio_retrieval_baseline.json`。

#### 2. 评测职责拆分

```text
人工Golden Set
├─ expected_chunk_ids → Hit@5、MRR（确定性计算）
├─ answerability → 回答/部分回答/拒答状态准确率
├─ reference_answer → Bad Case人工核对与后续正确性分析
└─ expected_sources → 证据来源核验

真实系统输出
├─ Ragas Faithfulness → 回答陈述是否被本次检索上下文支撑
├─ 程序规则 → 引用编号与引用Chunk是否合法
└─ 程序规则 → answerable/partial/unanswerable状态是否符合预期
```

结论：Ragas不能替代人工Golden Set。Faithfulness高只说明回答忠于提供的上下文；如果检索上下文本身错误，回答仍可能“忠实但不正确”，因此必须同时保留检索真值、状态标签和引用校验。

#### 3. Ragas接入问题与取舍

- **问题一：Python生态兼容**。项目声明`requires-python >=3.10`，但分类器只列出3.10～3.12；当前虚拟环境实际为Python 3.14。Ragas 0.4.3依赖的`scikit-network`没有对应Windows/Python 3.14 Wheel，安装时尝试源码编译并要求Microsoft Visual C++ 14+。
- **取舍**：不为单个评测依赖安装完整C++工具链，也不在当前阶段重建全部项目环境；固定`ragas>=0.2.15,<0.3.0`。该版本支持所需Faithfulness且不依赖`scikit-network`。长期标准环境仍建议Python 3.12。
- **问题二：LangChain API代际差异**。Ragas 0.2.15在导入时无条件引用已被`langchain-community 0.4.x`删除的旧VertexAI类型，即使当前只使用OpenAI-compatible接口也会失败。
- **取舍**：增加仅补齐未使用类型名的最小兼容垫片，不降级整套LangChain；真实Judge仍通过当前百炼OpenAI-compatible `base_url`调用Qwen。
- **验证**：Ragas、答案服务和评测运行器相关测试共46项通过；Ruff通过；`pip check`无依赖冲突。当前安装版本为Ragas 0.2.15、datasets 5.0.1。
- **已知限制**：这是Python 3.14下的兼容方案；Ragas与新版LangChain仍产生弃用警告，但不影响当前Faithfulness路径。README最终必须记录推荐Python版本和已验证依赖版本。

#### 5. 首次生成评测运行问题与修复

- **首次运行结果**：39题答案链路完成；34题回答、5题拒答，`status_accuracy=1.0`、`citation_coverage=1.0`、`citation_validity=1.0`。但34题Ragas均报`unknown async library, or not in async context`，因此`faithfulness_count=0`，不能把该报告当作完整生成基线。
- **根因**：Ragas 0.2.15的同步`single_turn_score()`内部使用`get_event_loop().run_until_complete()`；在Python 3.14与当前LangChain/OpenAI异步调用链组合下，`sniffio/anyio`无法识别合法异步上下文。
- **改进**：改用Ragas公开的异步`single_turn_ascore()`，由项目显式使用`asyncio.run()`创建异步上下文；增加`--resume-judges`模式，只重试已有报告中缺失的Judge分数，不重复检索、Rerank或答案生成。
- **二次问题**：首次异步修复后仍出现`Timeout should be used inside a task`。进一步定位发现Ragas导入`executor.py`时会执行`nest_asyncio.apply()`，全局替换Python 3.14的`asyncio.run`、Task和事件循环实现，导致`asyncio.wait_for()`无法识别当前Task。
- **最终修复**：在导入Ragas期间临时禁用`nest_asyncio.apply()`；CLI本身不需要嵌套事件循环，因此保留Python 3.14原生`asyncio.run()`更加合理。导入后恢复函数引用，不修改第三方包文件。
- **修复验收**：独立无API探针确认`asyncio.run`未被替换、`nest_patched=False`且`asyncio.wait_for`正常；相关47项无API测试通过，Ruff通过。
- **下一步**：执行恢复命令，确认34题Faithfulness成功后，才生成正式生成基线结论。

#### 6. 正式生成质量基线

- **评测输入**：全部39题；真实执行`RAGAnswerService + qwen3.7-text-rerank + qwen-flash`。34道已回答问题使用Ragas 0.2.15计算Faithfulness，5道不可回答题评估固定拒答行为。
- **汇总结果**：34题回答、5题拒答；系统错误0、Judge错误0；状态级准确率1.0；引用覆盖率1.0；引用结构有效率1.0；34题平均Faithfulness为`0.9485`。
- **分题型Faithfulness**：29道answerable平均`0.9637`，最低`0.6667`；5道partial平均`0.8605`，最低`0.6667`。5道unanswerable全部拒答，不计算Faithfulness。
- **分数分布**：25题为1.0；2题在[0.9,1.0)；5题在[0.8,0.9)；2题低于0.8。
- **正式证据**：`reports/portfolio_generation_baseline.json`。

指标解释边界：

1. `status_accuracy=1.0`目前是状态级检查：answerable/partial应为`answered`，unanswerable应为`refused`。它尚不能自动证明partial回答确实只回答了有证据部分，因此不能表述为“部分回答语义准确率100%”。
2. `citation_validity=1.0`表示引用编号和Chunk都来自本次检索结果，是结构有效性；它不等于每条引用在语义上充分支持对应结论。
3. Faithfulness只判断回答是否被当前Top 5上下文支撑，不判断Top 5是否为人工真值，因此不能替代Hit@5/MRR或人工参考答案。

关键Bad Case：

- `portfolio_q22`（answerable，Faithfulness 0.6667）：系统答案与人工参考答案基本一致。低分说明LLM Judge的声明拆分/蕴含判断会产生噪声，单题低分需要人工复核，不能直接认定为幻觉。
- `portfolio_q51`（partial，Faithfulness 0.6667）：答案正确给出建成年份和建筑风格，并说明资料未记录开放时间。Ragas难以从有限Top 5严格推出“资料未记录”，因此否定性/缺失性声明可能被判为不受支持，partial题均分也低于answerable。
- `portfolio_q29`（检索Hit@5=0，Faithfulness 1.0）：未命中人工标注的英文方法论Chunk，但从其他检索资料生成了受到当前上下文支持的替代答案。证明Faithfulness高不代表命中预期证据或完全符合参考答案。
- `portfolio_q54`（检索Hit@5=0，Faithfulness 0.8235）：正确英文Ragas维度Chunk未进入Top 5，模型根据另一份中文资料回答了不同的通用评测维度。回答对当前上下文较忠实，但偏离Golden Set参考答案，说明检索正确性是生成正确性的前置条件。

阶段结论：

```text
完整质量判断
= 检索真值（Hit@5/MRR）
+ 生成忠实度（Ragas Faithfulness）
+ 状态与引用规则检查
+ 对低分、检索失败和partial问题进行人工复核
```

当前基线可以证明系统整体具有较高的上下文忠实度和稳定拒答/引用行为，但跨语言与复合查询召回仍是主要质量短板；Ragas Judge对否定性陈述和个别声明拆分存在测量噪声。

#### 4. 生产链路一致性修复

- **发现**：`scripts.answer`虽然创建了Reranker，却以`_reranker`变量丢弃，真实答案链路实际使用RRF结果直接生成；而正式检索基线使用了RRF + Rerank。若不修复，检索与生成评测测的不是同一套系统，指标无法解释。
- **改进**：`RAGAnswerService`可接收现有Reranker；启用时先取`top_k × 2`粗排候选，再精排到最终Top K后交给LLM；Reranker fallback被视为系统错误，不伪装成知识不足或正常回答。
- **核心测试**：验证LLM收到精排后的Chunk顺序；验证fallback返回`error_stage=rerank`且不调用LLM；答案服务与评测相关回归共46项通过。
- **结论**：评测前必须确认“被评估代码路径”和“线上/演示代码路径”一致，否则即使指标计算正确，也不能代表真实系统质量。

### 阶段四面试式验收

完成 A0～A5 后，至少回答：

1. 为什么完整 RAG 不能只把 Top-K 片段拼进 Prompt？
2. Evidence Judge 与 Answer Generator 为什么分开？
3. 为什么引用元数据不能让 LLM 自己生成？
4. 如何区分“知识库无答案”和“系统调用失败”？
5. 如何证明一次 Prompt 或 Reranker 修改真的提升了系统？

---

## 7. 阶段五：Agent 集成与 Agent 协作开发

### B1：设计业务 Agent

- **做什么**：在完整 RAG 服务之上定义 Agent 的目标、工具、状态、停止条件和失败处理。
- **为什么**：Agent 的价值不是再调用一次 LLM，而是根据任务状态决定是否检索、是否改写、是否二次检索以及何时停止。
- **你要理解什么**：普通 RAG 流水线、Tool Calling 和 Agent 决策循环的区别。
- **产出**：Agent Spec 与技术设计。
- **验收**：至少存在一个“固定 RAG 流程无法完成，但 Agent 决策能改善”的真实场景。

### B2：实现最小 Agent 循环

- **做什么**：让模型通过现有 MCP Tool 或内部服务调用知识库，观察结果后决定回答、拒答或再次检索。
- **为什么**：验证 Agent 的核心闭环，而不是堆叠框架名词。
- **你要理解什么**：模型负责决策，Tool 负责执行，程序负责状态、权限、校验和停止条件。
- **产出**：单轮 Agent、状态 Trace 和 E2E 测试。
- **验收**：正常、无需检索、需要二次检索、工具异常和达到最大步数场景可复现。

### B3：用开发 Agent 完成一次 Spec 驱动任务

- **做什么**：从核心待办中选择一个任务，依次完成 Spec、Agent 计划、代码实现、人工 Review、QA 和复盘。
- **为什么**：职业开发不是把需求一句话交给 AI；人负责目标、约束、设计判断和验收，Agent 提高实现效率。
- **你要理解什么**：如何审查 Agent 的假设、Diff、测试覆盖和架构一致性。
- **产出**：任务 Spec、实现 Diff、测试记录、Review 结论和复盘。
- **验收**：能够清楚区分“Agent 做了什么”和“自己做了哪些关键决策”。

### B4：封装项目 Skill

- **做什么**：把已重复且稳定的开发/评估流程写成项目 Skill。
- **为什么**：Skill 适合固化流程，不适合替代尚未想清楚的需求和设计。
- **你要理解什么**：`AGENTS.md` 是仓库通用协作规则；Skill 是按需调用的专项工作流。
- **产出**：一个有触发条件、输入、步骤、检查项和停止条件的 Skill。
- **验收**：用两个同类任务验证 Skill 可复用，且不会越权或跳过验收。

### 阶段五面试式验收

围绕 Agent 循环、MCP/Skill 边界、状态管理、失败恢复、人机职责和安全停止条件进行 3～5 题问答。

---

## 8. 阶段六：求职交付

### C1：工程质量收口

- **做什么**：整理稳定测试、CI 质量门禁、依赖边界、错误可见性和已知限制。
- **为什么**：求职项目不仅要能演示，还要证明可维护、可复现。
- **产出**：CI、测试报告、环境说明和限制清单。
- **验收**：新环境可按 README 复现；稳定测试自动执行；真实付费评测与普通 CI 分离。

### C2：项目表达材料

- **做什么**：编写最终 README、架构图、3～5 分钟演示脚本和实验对比。
- **为什么**：面试官需要快速看懂场景、架构、难点、个人贡献和指标。
- **产出**：README、架构图、演示流程和实验表。
- **验收**：每个能力声明都能链接到代码、测试、Trace 或指标证据。

### C3：简历与模拟面试

- **做什么**：把项目整理为简历 bullet，并完成项目介绍和多轮追问。
- **为什么**：会做项目和能清楚证明自己会做，是两个不同能力。
- **产出**：简历描述、问题库、回答提纲和模拟面试记录。
- **验收**：能够脱离 Agent 完整讲述项目；能解释一个成功案例、一个 Bad Case、一个取舍和一个量化改进。

---

## 9. 核心改进待办（只保留可形成亮点的事项）

| ID | 核心问题 | 证据 | 验收方向 | 状态 |
|---|---|---|---|---|
| RAG-OPT-01 | BM25 负 IDF 可能降低融合可靠性 | G3 中正确 Chunk Dense 第1、Sparse 第14、Fusion 第2；多个高频查询词 IDF 为负 | 单变量比较当前 IDF 与非负 IDF，检查整体 Hit/MRR 和 G3 排名，不允许只优化单题 | 待实验 |
| RAG-OPT-02 | 重复正文占据 Top-K | 不同 `chunk_index` 出现相同 `content_hash`；人工标注中已确认多组大量或完全重复正文 | 在固定正式题集上比较去重前后 Hit@5、MRR、Top-K重复率、Rerank输入量、延迟和Token；正确证据召回不得下降 | Golden Set完成后实验 |
| RAG-ENG-01 | Embedding 批处理失败隐藏根因 | 39 条请求 HTTP 400 后继续进入存储，最终只显示 39 Chunk 对 0 Vector | API 失败时保留原始根因并停止存储；正常批次严格对齐 | 待评估 |
| RAG-ENG-02 | 评估异常可能被显示为假成功 | 曾出现有检索结果但 `(no metrics)` | 不兼容结果必须明确失败并返回非成功状态 | 已完成：正式评测启用strict errors，检索、Rerank和指标异常直接失败 |
| MCP-ENG-01 | MCP 2.x 与现实现不兼容 | 2.2.0 初始化失败，1.30.0 E2E 通过 | 新环境固定 1.x；未来迁移 2.x 必须通过 tools/list 与 tools/call 回归 | 1.x 已固定 |
| AGENT-RAG-01 | 弱相关 Top-K 可能诱发编造 | 真实答案CLI查询“什么是Azure OpenAI？”时，资料仅提到 Azure Embedding；`qwen-flash`明确使用“行业常识”补全定义，并用不支持该结论的Chunk 37作为引用，程序仍返回`answered` | 增加可校验的证据充分性判断；域外问题必须固定拒答，同时保持有答案问题质量 | 阶段四实现中 |
| RAG-CI-01 | 质量改动缺少自动回归门禁 | 当前指标主要依靠人工命令运行，后续修改可能使检索质量静默下降 | 基于稳定Golden Set建立无付费API的CI检查；阈值来自实测基线；低于阈值时CI明确失败 | 去重实验后实施 |

## 10. 已验证证据

- LLM：Qwen 模型真实调用成功；
- Embedding：`qwen3.7-text-embedding-flash`，向量维度 1024；
- 摄取：`chinese_long_doc.pdf` 生成 39 个 Chunk 并成功入库；
- 检索：Dense + BM25 + RRF 查询成功；
- 评估：人工修正5题真值后，RRF `Hit@5=1.00`、`MRR=0.90`；启用 `qwen3.7-text-rerank` 后 `Hit@5=1.00`、`MRR=1.00`，平均额外延迟371ms；
- 正式Golden Set：39题，包含29道answerable、5道partial、5道unanswerable；34题正式检索基线为`Hit@5=0.9412`、`MRR=0.9118`；
- 生成评测基础设施：Ragas 0.2.15兼容百炼OpenAI-compatible接口；39题批量评测脚本已就绪；相关Ruff、依赖检查和46项无API测试通过；
- 正式生成基线：39题中34题回答、5题拒答；状态级准确率、引用覆盖率、引用结构有效率均为1.0；34题平均Faithfulness为0.9485；
- 可观测性：Dashboard 与 Trace 已使用；
- MCP：初始化、工具发现和 `query_knowledge_hub` 调用 E2E 通过；
- 规格：`RAG_ANSWER_SPEC.md` 已定义回答、拒答、引用和异常行为。

## 11. 当前唯一任务

```text
当前阶段：阶段四——把 RAG 做成完整应用
当前任务：C1 课程交付收口
本次只做：整理README、运行命令、架构说明、正式评测指标、已知限制和演示流程
本次不做：新增Agent、扩展MCP Tool、重复Chunk去重、继续堆叠单元测试
下一产出：可复现的课程项目交付包
进入下一步条件：新环境能够按README完成摄取、查询、答案生成和评测；每项核心能力都有代码、测试、Trace或指标证据
```
