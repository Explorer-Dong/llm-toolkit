# LLM Evaluation

集成一些用于评估 LLM 能力的 benchmark。

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

## 环境配置

同步环境：

```bash
uv sync
source .venv/bin/activate
```

下载数据集：

```bash
hf download MathArena/aime_2026 \
  --repo-type=dataset \
  --local-dir data/aime26 \
  --max-workers 4

hf download fingertap/GPQA-Diamond \
  --repo-type=dataset \
  --local-dir data/gpqa_diamond \
  --max-workers 4
```

## 开始评估

所有的结果保存在 `output/<time>_<benchmark>_<model>` 文件夹。

### AIME26

```bash
python src/main.py \
  --benchmark aime26 \
  --model Qwen3.6-27B \
  --base-url <https://api.example.com>/v1 \
  --api-key <EMPTY> \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 81920 \
  --request-timeout 1800 \
  --num-workers 5 \
  --seed 42
```

### GPQA Diamond

```bash
python src/main.py \
  --benchmark gpqa_diamond \
  --model Qwen3.6-27B \
  --base-url <https://api.example.com>/v1 \
  --api-key <EMPTY> \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 81920 \
  --request-timeout 1800 \
  --num-workers 4 \
  --seed 42
```

### NL2Repo

作为独立项目维护在子目录下，移步 [sandbox/nl2repo/README.md](sandbox/nl2repo/README.md) 作进一步了解。

### SciCode

作为独立项目维护在子目录下，移步 [sandbox/scicode/README.md](sandbox/scicode/README.md) 作进一步了解。

## 断点续评

对于非独立维护的 benchmark。如果需要复用上一次中断的评测结果，只需要删除对应任务的 `predictions.jsonl` 中的无效行，然后添加运行参数 `--output-dir <path/to/log>` 和 `--resume` 重新运行即可。示例命令：

```bash
python src/main.py \
  --benchmark aime26 \
  --output-dir outputs/2026-07-02_22-32-25_aime26_Qwen-Qwen3.6-27B \
  --resume
```
