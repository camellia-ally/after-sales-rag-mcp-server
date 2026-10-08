# 京东售后问答：一次运行三项对照试验

在项目根目录、已激活 `.venv` 的终端运行：

```powershell
python scripts/evaluate_jd_full.py
```

## 测什么

1. **双路召回**：一套固定的 50 道可回答题，分为关键词 17 题、语义改写 17 题、关键词与语义结合 16 题。同题比较 Dense only、BM25 only、Dense+BM25+RRF。RRF 使用项目当前的 `k=60`，不根据题目成绩调参。
2. **重排**：同一批题、同一批 RRF 前 10 候选，比较重排前后 Top 5。当前报告中的实际模型为 `dashscope/qwen3-rerank`；具体运行以报告的 `reranker` 字段为准。
3. **拒答**：另一套 10 道无法给出具体结论或资料不足的题，测拒答率；另附 6 道可回答对照题，测误拒答率，防止“全部拒答”取得虚高成绩。

检索题在 `tests/fixtures/jd_after_sales_retrieval_50_v3.json`；拒答题在 `tests/fixtures/jd_after_sales_refusal_10_v3.json`。每道可回答题给出来源、证据短句和入库 Chunk ID。组合题要求所有必需证据都进入前 5。60 道题的人工核对清单在 `reports/jd_after_sales_label_review_v3.md`；拒答集中的 6 道可回答对照题来自检索集，无需重复核对。

## 看哪些数

- 各路总体及分场景的证据 Chunk Hit@5、MRR@5、Top 1。
- 组合题全部证据找齐率@5；来源文件指标作为辅助，文件命中而证据 Chunk 未命中不算成功。
- 重排额外耗时；无答案题拒答率和可回答题误拒答率。
- 逐题来源、Chunk 排名及回答，便于检查失分原因。检索或重排发生回退时实验报错退出。

汇总写入 `reports/jd_after_sales_full_pilot.json`，详细结果分别写入同目录的 `_retrieval.json` 和 `_refusal.json`。命令会调用 Embedding、重排及回答模型，产生 API 费用。只查看运行顺序可用 `python scripts/evaluate_jd_full.py --dry-run`，不调用 API。

## 重排相关度门槛

答案服务可选用 `--min-rerank-score` 在生成前拦截低相关度问题。该分数来自重排模型，不是向量相似度或 RRF 分数。先用 `--mode scores` 收集分数，再用 `--mode refusal` 验证阈值；分数采集只调用 Embedding 与重排接口。

```powershell
python scripts/evaluate_jd_after_sales.py --mode scores --test-set tests/fixtures/jd_after_sales_refusal_10_v3.json --rrf-k 60 --output reports/jd_refusal_scores.json
python scripts/evaluate_jd_after_sales.py --mode refusal --test-set tests/fixtures/jd_after_sales_refusal_10_v3.json --rrf-k 60 --min-rerank-score 0.6 --output reports/jd_refusal_threshold_060.json
```

当前 0.60 阈值报告中，10/10 不可回答题拒答，6 道可回答对照题均未误拒答；4 道题由门槛提前拒答，其余 6 道由模型判断证据不足。分数有重叠，因此单一阈值无法承担全部拒答决策。

## 结果边界

这是一套有意覆盖三类需求的诊断题，不代表线上提问的自然分布。题目与证据目前标记为 `agent_draft`；报告出来后需要逐题核查失分和替代证据，不能直接把有利数字写进简历，也不能删掉失败题凑指标。拒答试验只统计 `answered/refused` 状态，不能代替回答事实正确性评估。库中 4 个旧 Chunk 有原文不存在的 `[IMAGE: id]` 标记，需要在正式答案质量评估前处理。
