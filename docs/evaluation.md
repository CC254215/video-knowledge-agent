# Evaluation and Improvement Plan

This project should be evaluated as a grounded video knowledge agent, not as a generic video summarizer. The core proof target is: for a user question, the system retrieves the right video evidence, answers only from that evidence, cites valid segment/frame evidence, and says evidence is insufficient when the video does not support the claim.

## Relevant Open Methods and Benchmarks

- VideoRAG / LongerVideos: use as the nearest long-video RAG reference. Its useful ideas for this codebase are graph-style cross-video knowledge indexing, dual-channel text plus multimodal retrieval, and explicit long-video benchmark reporting. Source: https://arxiv.org/abs/2502.01549 and https://github.com/HKUDS/VideoRAG
- EgoSchema: use for long temporal VideoQA stress tests. It contains human-curated multiple-choice questions over three-minute clips and exposes whether the system handles temporally spread evidence. Source: https://proceedings.neurips.cc/paper_files/paper/2023/hash/90ce332aff156b910b002ce4e6880dec-Abstract-Datasets_and_Benchmarks.html
- ActivityNet-QA and NExT-QA: use for open-ended QA and causal/temporal reasoning coverage. ActivityNet-QA has 58k QA pairs over 5.8k videos. Sources: https://arxiv.org/abs/1906.02467 and https://github.com/chakravarthi589/Video-Question-Answering_Resources
- TVSum / SumMe: use only for visual summary and keyframe selection sanity checks, not as the only proof of answer quality. Summary F1 can overstate quality, so pair it with grounding metrics. Sources: https://people.csail.mit.edu/yalesong/publications/SongVSJ2015CVPR.pdf and https://mayu-ot.github.io/rethinking-evaluation-of-video-summaries/

## Improvement Roadmap

1. Add graph-style memory over extracted claims.
   Current retrieval is per-video BM25/vector/RRF. Add a claim graph keyed by entities, topics, timestamps, and evidence ids, then retrieve both direct evidence and neighboring claims. This mirrors VideoRAG's graph grounding idea while preserving local-first storage.

2. Add query-conditioned evidence acquisition.
   The current FrameEvidenceRefiner is already a good base. Extend it so eval failures with low temporal IoU or missing visual evidence write a targeted refinement request, then rerun only affected ranges.

3. Add dataset adapters instead of one-off scripts.
   Normalize ActivityNet-QA, EgoSchema, NExT-QA, TVSum/SumMe, and private project videos into the JSON/JSONL schema below. The project should not hard-code benchmark formats in the core agent.

4. Add hard negative and abstention sets.
   Include questions whose answer is not in the video, visually unanswerable questions when frame captions are missing, and misleading questions that require the agent to reject the premise.

5. Report cost and latency with quality.
   Store per-case latency, model availability, evidence type coverage, and refinement counts. A method that improves recall but doubles VLM calls should be visible in the report.

## Eval Dataset Schema

Use JSONL for small human-labeled sets:

```json
{"id":"q001","video_id":"video_a","question":"核心论点是什么？","answer":"关键结论必须绑定时间戳证据","answer_contains":["时间戳证据"],"evidence_ids":["video_a:seg_0003:speech"],"time_ranges":[{"start":120,"end":180}]}
```

Supported aliases:

- `id` / `qid` -> `case_id`
- `answer` / `answers` / `gold_answers`
- `evidence_ids` / `gold_evidence_ids`
- `segment_ids` / `gold_segment_ids`
- `time_ranges` / `gold_time_ranges`
- `must_include` / `answer_contains`

## Metrics

- Retrieval: `retrieval_hit_rate`, `retrieval_mrr`, `segment_precision`, `segment_recall`, `segment_f1`, `evidence_precision`, `evidence_recall`.
- Temporal grounding: `temporal_iou` between retrieved evidence ranges and gold ranges.
- Answer quality: `answer_contains_recall` and `answer_token_f1` against accepted answers.
- Citation grounding: `grounding_precision`, `grounding_recall`, `invalid_citation_count`.
- Safety behavior: `abstention_accuracy` for unsupported questions.
- Operations: `latency_seconds`; future reports should add token counts, VLM frame count, and refinement count.

Run retrieval-only evaluation:

```powershell
py -3.11 -m app.cli eval --dataset data\eval\cases.jsonl --top-k 5 --output data\eval\report.json
```

Run full QA evaluation:

```powershell
py -3.11 -m app.cli eval --dataset data\eval\cases.jsonl --run-answers --fail-on-threshold
```

Default release thresholds are intentionally conservative until a real labeled set exists: retrieval hit rate >= 0.70, segment recall >= 0.60, grounding precision >= 0.80, answer contains recall >= 0.60.
