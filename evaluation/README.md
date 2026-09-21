# LLM Evaluation

集成一些用于评估 LLM 能力的 benchmark。**每一个 benchmark 是 `evaluation/` 下的一个自包含子文件夹**：自带 uv 环境、独立入口、自己的 README 与数据目录。

| bench | 类型 | 入口 | 说明 |
|---|---|---|---|
| `aime26/` | 数学 | `main.py` | AIME 2026，整数答案判分 |
| `gpqa_diamond/` | 选择题 | `main.py` | GPQA-Diamond，A/B/C/D 判分 |
| `webwalkerqa/` | 网页遍历 agent | `main.py` + `score.py` | WebWalkerQA，WebWalker/ReAct/Reflexion 三 method，LLM-as-judge 判分 |
| `nl2repo/` | 代码 agent（沙箱） | `main.py` | 独立项目，见其 README |
| `scicode/` | 科学代码（沙箱） | `eval/` | 独立项目，见其 README |

Table 1. Official / Ours

|| Qwen3.6-27B | Qwen3.6-35B-A3B | GLM-5.2 (FP8) | Qwen3.5-35B-A3B | Kimi K3 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| AIME26 | 94.1 / 93.3 | 92.7 / 96.7 | - / - | - / - | - / - |
| GPQA Diamond | 87.8 / 86.4 | 86.0 / 83.6 | - / - | - / - | 93.5 / 94.9 |
| NL2Repo | 36.2 / - | 29.4 / - | - / - | - / - | - / - |
| SciCode (Sub problem) | - / - | - / - | 50.5 / 48.5 | 37.7 / 37.5 | 58.7 / 52.2 |

> - Qwen3.6-27B [Tongyi Official](https://qwen.ai/blog?id=qwen3.6-27b)
> - Qwen3.6-35B-A3B: [Tongyi Official](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
> - GLM-5.2: [Artificial Analysis](https://artificialanalysis.ai/articles/glm-5-2-is-the-new-leading-open-weights-model-on-the-artificial-analysis-intelligence-index)
> - Qwen3.5-35B-A3B: [Artificial Analysis](https://artificialanalysis.ai/models/qwen3-5-35b-a3b)
> - Kimi-K3: [Kimi Official](https://www.kimi.com/blog/kimi-k3)

## 通用用法

每个 bench 都是同样的三步：进目录 → `uv sync` → 跑 `main.py`。数据集放在各自的 `data/`（gitignore），`hf download` 到位后离线可用。结果写入 `outputs/<时间戳>_<bench>_<model>/`（config.json / predictions.jsonl / errors.jsonl / summary.json）。

```bash
cd evaluation/aime26
uv sync
hf download MathArena/aime_2026 --repo-type=dataset --local-dir data --max-workers 4

uv run main.py \
  --model Qwen3.6-27B \
  --base-url https://api.example.com/v1 \
  --api-key EMPTY \
  --temperature 1.0 --top-p 0.95 --top-k 20 --max-tokens 81920 \
  --seed 42
```

`gpqa_diamond` 同理（数据集 `fingertap/GPQA-Diamond`）；`webwalkerqa`、`nl2repo`、`scicode` 移步各自文件夹的 README。

公共 CLI 约定（各 bench 一致，详见 `AGENTS.md`）：`--model --base-url --api-key --split --limit（0=全量） --output-dir --temperature --top-p --top-k --max-tokens --num-workers --seed --resume --dry-run`。

## 断点续评

如需复用上一次中断的评测结果，删除 `predictions.jsonl` 中的无效行，然后添加 `--output-dir <path/to/run目录>` 和 `--resume` 重新运行即可：

```bash
uv run main.py \
  --model Qwen3.6-27B --base-url ... --api-key ... \
  --output-dir outputs/2026-07-02_22-32-25_aime26_Qwen-Qwen3.6-27B \
  --resume
```
