# Evaluation

Inference and scoring.

## Setup

From `eval/`:

```bash
uv sync --group dev
cp .env.example .env
```

| Connection | Environment variables |
| --- | --- |
| Model | `MODEL_NAME`, `BASE_URL`, `API_KEY` |
| DS search / DS fetch | `DS_TOOL_KEY` |
| LLM judge | `JUDGE_MODEL`, `JUDGE_BASE_URL`, `JUDGE_API_KEY` |
| (Optional) Serper search | `SERPER_API_KEY`; select `--search-provider serper` |
| (Optional) Crawl4AI fetch | `CRAWL4AI_URL`, `CRAWL4AI_TOKEN`; select `--fetch-provider crawl4ai` |

- Search and fetch default to DeepSeek.
- Local Crawl4AI: `docker compose up -d`.

## Inference

```bash
# single-turn
uv run eval-inference aime --year 2026
uv run eval-inference gpqa
uv run eval-inference math

# multi-turn
uv run eval-inference gaia
uv run eval-inference browsecomp
uv run eval-inference webwalkerqa

# swe-like bench (docker required)
uv run eval-inference swe_bench \
  --limit 2 \
  --sample-concurrency 2 \
  --concurrency 2

# see more options
uv run eval-inference <benchmark> --help
```

| Benchmarks | Default turns | Scoring |
| --- | --- | --- |
| AIME / GPQA / MATH | 1 | Integer / choice / symbolic equivalence |
| GAIA | 16 | Normalized answer |
| BrowseComp / WebWalkerQA | 16 | LLM judge |
| SWE-bench Verified (Easy) | 40 | Official tests |

| Option | Effect |
| --- | --- |
| `--limit N` | First N matching samples |
| `--max-turns N` | Model completion budget; `--max-steps` is an alias |
| `--task-timeout S` | Episode budget; `0` disables it |
| `--concurrency N` | Model requests across all samples |
| `--sample-concurrency N` | Active samples, including runtime setup and cleanup |
| `--output-dir PATH` | New inference directory |
| `--disable-thinking` | Disable thinking; (default `enabled`) |
| `--reasoning-effort LEVEL` | Reasoning effort; (default `medium`) |

## Scoring

```bash
# rubric-based bench
uv run eval-scoring aime outputs/aime/<run>

# judge-based bench
uv run eval-scoring browsecomp outputs/browsecomp/<run> \
  --judge-model <model>

# sandbox-based bench
uv run --extra swe_bench eval-scoring swe_bench outputs/swe_bench/<run> \
  --sample-concurrency 2
```

- LLM judges use the same `--disable-thinking` and `--reasoning-effort` options.
- Each invocation creates a new score directory; second-level timestamps use `-1`, `-2`, ... for collisions.
- Rule-based, LLM-judge, and hybrid policies live in each benchmark's `scoring.py`.
- Scoring loads the original dataset and verifies its hash and selected sample IDs.
- Inference evidence is checked before and after scoring.
- SWE-bench applies saved patches in fresh containers. Official scoring requires the `swe_bench` extra (`swebench==5.0.2`).

## Outputs

```text
outputs/<benchmark>/<model>_<timestamp>/
├── config.json                        # options, dataset hash, selected sample IDs
├── trajectories/<encoded-id>.json     # harness execution
├── artifacts/<encoded-id>/            # collected evidence, when needed
│   ├── manifest.json                  # evidence metadata and hashes
│   └── ...                            # e.g. patch.diff
└── scores/<timestamp>/
    ├── config.json
    ├── trajectories.json
    └── summary.json
```

`summary.json` groups costs under `costs.inference`, `costs.scoring`, and `costs.total`.

| Category | Totals and means |
| --- | --- |
| `tokens` | Prompt, completion, reasoning, cached, total; per task and model call |
| `time_sec` | Wall time, cumulative sample/episode time, model time, queue time, runtime call time; per task/call |
| `calls` | Model calls, calls reporting usage, runtime calls, runtime errors; per task |
| `by_tool` | Calls, errors, cumulative time and means for each runtime operation |

- `total_tokens = prompt_tokens + completion_tokens`; reasoning and cached tokens are reported separately.
- Model calls count agent/judge turns. Runtime calls count recorded operations.
- Means include failed samples. Unavailable timing values are `null`.
- Sample time includes runtime setup and cleanup; inference episode time covers the agent loop.

## Layout

```text
src/eval/
├── inference.py    # runtime lifecycle and trajectory storage
├── scoring.py      # scoring orchestration and cost summaries
├── utils.py        # shared scheduling and storage
└── benchmarks/<name>/
    ├── inference.py    # dataset loading and agent rollout
    ├── scoring.py      # scoring rules, judge calls, and scoring entrypoint
    └── data/           # dataset and benchmark assets
```

- Add a benchmark directory; use AIME or SWE-bench as an example.
- Inference: `eval-inference <name>` or `python -m eval.benchmarks.<name>.inference`.
- Scoring: `eval-scoring <name> <run_dir>` or `python -m eval.benchmarks.<name>.scoring <run_dir>`.
- Eval owns runtime lifecycles; harness saves trajectories.

## Tests

```bash
uv run --extra swe_bench python -m unittest discover -s tests
uv run --extra swe_bench python tests/smoke.py --limit 1
```

[Static checks and provider smoke tests](tests/README.md).
