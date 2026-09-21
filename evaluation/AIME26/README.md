# AIME26

AIME 2026 数学竞赛评测，共 30 题（[MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026)），每题答案为 0–999 的整数。要求模型逐步推理，并在结尾输出 `{"answer": 123}`，按整数精确匹配判分。

自包含单文件评测脚本 `main.py`，通过 OpenAI 兼容 API（vLLM 等）请求推理端点。

## 运行

```bash
uv sync
uv run main.py \
  --model Qwen/Qwen3.6-27B \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY
```

常用参数：

- `--limit N`：只跑前 N 题（数据不可用时退回内置 fallback 样例，可离线冒烟测试）
- `--resume`：向已有 run 追加，跳过 `predictions.jsonl` 中已完成的题目
- `--dry-run`：只加载数据并打印首题，不发请求
- `--num-workers N`：异步并发请求数（默认 1，顺序执行）
- 采样参数：`--temperature`（默认 1.0）、`--top-p`（0.95）、`--top-k`（20，经 `extra_body` 下发）、`--max-tokens`（81920）

数据优先读取本地 `data/` 快照（已 gitignore），否则自动从 HuggingFace 下载并缓存到 `~/.cache/huggingface`。

## 输出

结果写入 `outputs/<时间戳>_aime26_<model>/`：

- `predictions.jsonl`：每题的原始输出、解析答案、gold、得分、耗时、错误
- `errors.jsonl`：请求或解析失败的记录
- `config.json` / `summary.json`：本次配置与汇总（accuracy 等）
