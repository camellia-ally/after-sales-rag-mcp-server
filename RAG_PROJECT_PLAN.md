# RAG 项目落地计划表

> 状态说明：本文件保留早期规划供回顾，阶段进度和简历模板不作为当前实现证据。实际运行命令及售后场景结果见 [README.md](README.md) 与 [JD_AFTER_SALES_EVAL.md](JD_AFTER_SALES_EVAL.md)。

**项目名称**：客服知识库 RAG 检索服务  
**项目目标**：将通用 RAG 框架落地到客服知识库检索场景，支持 PDF/Word/Markdown 多格式文档，建立多 Collection 管理，完成评估测试，生成简历数据。

---

## 📋 总体规划

| Phase | 任务 | 预计时间 | 状态 |
|-------|------|---------|------|
| Phase 1 | 数据准备与环境配置 | 0.5-1 天 | ⬜ 未开始 |
| Phase 2 | 多 Collection 架构实现 | 1-2 天 | ⬜ 未开始 |
| Phase 3 | Word 文档支持扩展 | 0.5-1 天 | ⬜ 未开始 |
| Phase 4 | 准备测试数据集 | 1 天 | ⬜ 未开始 |
| Phase 5 | 基线对比实验 | 1-2 天 | ⬜ 未开始 |
| Phase 6 | 人工评分与质量验证 | 0.5-1 天 | ⬜ 未开始 |
| Phase 7 | Dashboard 完善与可视化 | 0.5-1 天 | ⬜ 未开始 |
| Phase 8 | 文档与简历整理 | 0.5 天 | ⬜ 未开始 |

**总计时间**：约 1-1.5 周

---

## 🎯 Phase 1：数据准备与环境配置

**目标**：准备客服领域文档数据，配置 RAG 项目环境

### 任务清单
- [ ] 1.1 收集客服文档（5-10 份 PDF）
  - 客服 FAQ（退款、物流、账户）
  - 产品使用手册
  - 政策条款文档
  - 来源：网上公开资料或自己编写
  
- [ ] 1.2 配置 RAG 项目环境
  - 安装依赖：`pip install -r requirements.txt`
  - 配置 `.env`（API keys, 模型选择）
  - 测试 MCP Server 启动
  
- [ ] 1.3 验证基础功能
  - 测试 PDF 摄取（Ingestion）
  - 测试向量检索
  - 查看 Streamlit Dashboard

### 产出
- ✅ 客服文档集：`data/customer_service/`
- ✅ 环境配置完成
- ✅ 基础功能验证通过

### 完成标记
- [ ] **Phase 1 完成**

---

## 🎯 Phase 2：多 Collection 架构实现

**目标**：实现多知识库隔离，支持不同 Agent 查询不同 Collection

### 任务清单
- [ ] 2.1 创建多个 Collection
  - `customer_service_faq`（通用 FAQ）
  - `technical_docs`（技术问题文档）
  - `billing_policy`（账单/退款政策）
  - `product_manual`（产品使用手册）
  
- [ ] 2.2 文档分类与摄取
  - 将收集的文档按类型分配到不同 Collection
  - 运行 Ingestion Pipeline
  - 验证每个 Collection 的文档数量
  
- [ ] 2.3 修改 MCP Server 接口
  - 支持 `collection_name` 参数
  - 测试跨 Collection 查询
  - 更新 MCP 工具定义

### 产出
- ✅ 多 Collection 数据库
- ✅ MCP Server 支持指定 Collection
- ✅ Dashboard 显示多个 Collection

### 完成标记
- [ ] **Phase 2 完成**

---

## 🎯 Phase 3：Word 文档支持扩展

**目标**：添加 Word (.docx) 文档解析能力

### 任务清单
- [ ] 3.1 实现 Word Loader
  - 创建 `src/libs/loader/docx_loader.py`
  - 使用 `python-docx` 库解析
  - 提取文本、表格、图片描述
  
- [ ] 3.2 集成到 Ingestion Pipeline
  - 在配置中注册 Word Loader
  - 测试摄取 Word 文档
  - 验证解析结果
  
- [ ] 3.3 准备测试文档
  - 收集或创建 2-3 份 Word 格式的客服文档
  - 测试不同格式（表格、列表、图片）

### 产出
- ✅ `docx_loader.py` 实现
- ✅ Word 文档摄取成功
- ✅ 单元测试通过

### 完成标记
- [ ] **Phase 3 完成**

---

## 🎯 Phase 4：准备测试数据集

**目标**：构建客服问答测试集，用于后续评估

### 任务清单
- [ ] 4.1 编写测试问题（30-50 条）
  - 通用问题 10 条（FAQ）
    - 例如："如何申请退款？"
  - 技术问题 10 条（登录、错误码）
    - 例如："登录显示 401 错误怎么办？"
  - 账单问题 10 条（退款、支付）
    - 例如："退款多久到账？"
  - 产品问题 10 条（功能使用）
    - 例如："如何修改收货地址？"
  
- [ ] 4.2 标注预期答案
  - 为每个问题编写 Golden Answer
  - 标注答案应该来自哪个文档
  - 标注难度级别（简单/中等/困难）
  
- [ ] 4.3 构造难度梯度
  - 简单问题：直接命中（关键词精确匹配）
  - 中等问题：需要语义匹配（同义词理解）
  - 困难问题：需要多文档综合

### 产出
- ✅ 测试数据集：`test_data/customer_service_qa.json`
- ✅ 格式示例：
```json
{
  "questions": [
    {
      "id": 1,
      "question": "如何申请退款？",
      "golden_answer": "...",
      "source_doc": "billing_policy.pdf",
      "collection": "billing_policy",
      "difficulty": "easy"
    }
  ]
}
```

### 完成标记
- [ ] **Phase 4 完成**

---

## 🎯 Phase 5：基线对比实验

**目标**：运行对比实验，量化混合检索的提升效果

### 任务清单
- [ ] 5.1 纯向量检索 Baseline
  - 关闭 BM25（只用 Dense Retrieval）
  - 运行测试集
  - 记录指标：
    - 召回率（Recall）
    - Top-5 命中率（Hit Rate@5）
    - 平均排名（MRR）
  
- [ ] 5.2 混合检索实验
  - 开启 Dense + BM25 + RRF Fusion
  - 运行测试集
  - 记录相同指标
  - 计算提升百分比
  
- [ ] 5.3 Rerank 效果验证
  - 关闭 Rerank（只用 Fusion）
  - 开启 Cross-Encoder Rerank
  - 对比 Top-5 准确率变化
  
- [ ] 5.4 统计分析
  - 计算各项指标提升
  - 分难度级别统计
  - 生成对比图表

### 产出
- ✅ 实验报告：`reports/customer_service_baseline.json`
- ✅ 关键数据（示例）：
  - 混合检索 vs 纯向量：召回率提升 15-20%
  - Rerank vs 无 Rerank：Top-5 准确率提升 10-15%
- ✅ 对比图表（柱状图、折线图）

### 完成标记
- [ ] **Phase 5 完成**

---

## 🎯 Phase 6：人工评分与质量验证

**目标**：人工评估答案质量，补充自动化指标

### 任务清单
- [ ] 6.1 抽样评估
  - 随机抽取 20 条问答结果
  - 人工打分（1-5 分）
    - 1 分：完全不相关
    - 2 分：相关但严重不完整
    - 3 分：部分相关但不完整
    - 4 分：相关且基本完整
    - 5 分：准确且完整
  
- [ ] 6.2 对比评分
  - 纯向量检索结果评分
  - 混合检索结果评分
  - 加 Rerank 后结果评分
  - 计算平均分提升
  
- [ ] 6.3 Bad Case 分析
  - 记录失败案例
  - 分析失败原因：
    - 文档不包含答案
    - 切块不合理
    - 查询改写失败
    - Rerank 错误降权

### 产出
- ✅ 人工评分表格
- ✅ 平均分对比：
  - 纯向量：X.X/5
  - 混合检索：Y.Y/5（提升 Z%）
  - +Rerank：W.W/5（提升 M%）
- ✅ Bad Case 分析报告

### 完成标记
- [ ] **Phase 6 完成**

---

## 🎯 Phase 7：Dashboard 完善与可视化

**目标**：完善 Streamlit Dashboard，展示评估结果

### 任务清单
- [ ] 7.1 多 Collection 可视化
  - 首页显示每个 Collection 的文档数量
  - 支持下拉选择 Collection 查询
  - 显示 Collection 的文档列表
  
- [ ] 7.2 评估结果展示
  - 添加"评估报告"页面
  - 展示对比图表（召回率、准确率、人工评分）
  - 显示各难度级别的表现
  - 展示 Bad Case 列表
  
- [ ] 7.3 Query Trace 优化
  - 显示 Dense/Sparse/Fusion 各阶段结果
  - 高亮 Rerank 前后排名变化
  - 显示每个结果的得分和来源文档

### 产出
- ✅ 完善的 Streamlit Dashboard
- ✅ 评估结果可视化页面
- ✅ Demo 演示材料（截图/录屏）

### 完成标记
- [ ] **Phase 7 完成**

---

## 🎯 Phase 8：文档与简历整理

**目标**：整理项目文档，准备简历描述

### 任务清单
- [ ] 8.1 更新 README
  - 添加客服场景介绍
  - 补充多 Collection 使用说明
  - 添加评估结果摘要
  - 更新快速开始指南
  
- [ ] 8.2 撰写简历描述
  - 基于实验数据生成简历文案
  - 突出技术亮点：
    - 混合检索 + 重排
    - 多 Collection 管理
    - MCP 协议封装
    - 可插拔架构
  - 量化收益（提升百分比）
  
- [ ] 8.3 准备 Demo 演示
  - 录制 Dashboard 操作视频（3-5 分钟）
  - 准备面试讲解要点
  - 整理关键代码片段

### 产出
- ✅ 更新的 README.md
- ✅ 简历描述文案
- ✅ Demo 演示视频
- ✅ 面试准备材料

### 简历描述模板
```latex
\datedline{\textbf{【客服知识库 RAG 检索服务】}}{【2025.09 - 2025.12】}
技术栈：RAG, Hybrid Search, BM25, RRF, Cross-Encoder, ChromaDB, MCP, Streamlit

\begin{itemize}
  \item \textbf{RAG链路优化：}针对客服场景检索不准确问题，设计向量 + BM25混合召回与Cross-Encoder重排；在50条客服问答测试集上，混合检索相比纯向量召回命中率提升18\%，Cross-Encoder重排后Top-5准确率提升12\%。
  
  \item \textbf{多集合知识库管理：}实现FAQ、技术文档、账单政策、产品手册四类知识库隔离，支持不同Agent按需查询指定集合；基于可插拔架构，LLM/Embedding/Reranker每个环节通过配置文件零代码切换。
  
  \item \textbf{MCP协议封装：}将RAG封装为MCP Server供Agent按需调用；基于Streamlit构建可视化管理平台，支持文档摄取追踪、查询链路分析、评估结果展示。
\end{itemize}
```

### 完成标记
- [ ] **Phase 8 完成**

---

## 📊 进度跟踪

### 当前进度
- **当前阶段**：Phase 1
- **完成百分比**：0%
- **下次开始**：Phase 1 - 数据准备与环境配置

### 时间记录
| Phase | 开始时间 | 完成时间 | 实际用时 |
|-------|---------|---------|---------|
| Phase 1 | - | - | - |
| Phase 2 | - | - | - |
| Phase 3 | - | - | - |
| Phase 4 | - | - | - |
| Phase 5 | - | - | - |
| Phase 6 | - | - | - |
| Phase 7 | - | - | - |
| Phase 8 | - | - | - |

---

## 🎯 最终目标

### 技术目标
- ✅ 支持 PDF/Word/Markdown 多格式文档
- ✅ 实现多 Collection 知识库管理
- ✅ 混合检索 + 重排完整链路
- ✅ MCP 协议集成
- ✅ 可视化 Dashboard

### 评估目标
- ✅ 完成 30-50 条问答测试
- ✅ 混合检索提升 15%+ 召回率
- ✅ Rerank 提升 10%+ Top-5 准确率
- ✅ 人工评分验证质量提升

### 简历目标
- ✅ 量化数据支撑
- ✅ 技术亮点突出
- ✅ 场景落地完整
- ✅ 与 EchoMind 形成互补

---

## 📝 注意事项

1. **职责分离**
   - RAG 项目：专注检索质量优化
   - EchoMind：专注 Agent 协作与工具调用
   - 两个项目通过 MCP 协议解耦

2. **数据类型**
   - 知识类文档（PDF/Word）→ RAG 检索
   - 结构化数据（订单/物流）→ 数据库查询 Tool（在 EchoMind 中）

3. **评估原则**
   - 数据驱动，拒绝"凭感觉"
   - 多维度指标（自动化 + 人工评分）
   - 对比实验（有 baseline）

4. **面试准备**
   - 能讲清楚技术选型原因
   - 能展示量化提升效果
   - 能演示完整 Demo
   - 能回答深度技术问题

---

## 🚀 开始实施

准备好开始了吗？

**下一步**：Phase 1 - 数据准备与环境配置

执行命令：
```bash
# 1. 进入项目目录
cd D:/PyCharm/pytest/实习/MODULAR-RAG-MCP-SERVER-main/

# 2. 创建数据目录
mkdir -p data/customer_service

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env  # 然后编辑 .env

# 5. 启动 MCP Server 测试
python -m src.mcp_server
```

**记得每完成一个任务就勾选 ✅，每完成一个 Phase 就更新进度跟踪表！**
