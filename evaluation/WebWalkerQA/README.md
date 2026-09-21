# WebWalkerQA

WebWalkerQA 包含 680 道 multi-hop 网页问答（[Paper](https://arxiv.org/abs/2501.07572)、[Code](https://github.com/Alibaba-NLP/DeepResearch/tree/main/WebAgent/WebWalker)）。Agent 从题目给定的 root_url 出发，用无头浏览器逐页点击按钮遍历网站寻找答案，由 LLM-as-a-judge 对照参考答案判分。

目录说明：

- `src/methods.py`: 三种 method 的实现逻辑，webwalker 是论文提出的 agent，react 和 reflexion 是两个 baseline 方法。
- `src/judge.py`: LLM-as-a-judge 判分 + 报告。

## 运行前配置

依赖配置：

```bash
uv sync
cp .env.example .env   # 端点配置：推理 AGENT_*、判分 JUDGE_*，两者可用不同端点
```

数据配置：首次运行自动从 HuggingFace 下载 [`callanwu/WebWalkerQA`](https://huggingface.co/datasets/callanwu/WebWalkerQA) 并缓存到 `~/.cache/huggingface`。

启动推理（128K 上下文）：

```bash
vllm serve ./_models/Qwen2.5-7B-Instruct \
  --served-model-name Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 5201 \
  --api-key "sk-vincent" \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --hf-overrides '{"rope_parameters":{"factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000,"rope_type":"yarn"}}' \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

## 运行 Methods

```bash
cd src
uv run methods.py \
  --method react \
  --model Qwen2.5-7B-Instruct \
  --output_path ../results/react_qwen2.5-7b.jsonl \
  --max_rounds 15 \
  --workers 5 \
  --limit 5
```

## 运行 Judge

```bash
cd src
uv run judge.py \
  --input_path ../results/react_qwen2.5-7b.jsonl \
  --output_path ../results/react_qwen2.5-7b_judge_gpt-4o.jsonl
```

判分结果追加写入 output_path（可中断续跑），汇总报告在同名 `_report.json`。
