# LLM Evaluation

Evaluation for LLMs.

| bench | type | note |
|---|---|---|
| `AIME26/` | Math | verified by `int` number |
| `GPQA-Diamond/` | World Knowledge | verified by `A/B/C/D` choice |
| `NL2Repo/` | Coding Agent | Sandbox |
| `SciCode/` | Science | verified by `test cases` |
| `WebWalkerQA/` | Web Agent | Agent + LLM-as-a-judge |

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
