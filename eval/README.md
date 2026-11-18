# Model evaluation

All commands below are from `eval/`.

## Setup

setup dependencies:

```bash
# regular benchmark
uv sync --group dev

# agent abstain benchmark
uv sync --group dev --extra agent_abstain

# webshop abstain benchmark
uv sync --group dev --extra webshop_abstain

# swe_bench benchmark
uv sync --group dev --extra swe_bench
```

>WebShop Abstain requires Java 21 and the assets described in [data/README.md](src/eval/benchmarks/webshop_abstain/data/README.md).

setup environment variables:

```bash
cp .env.example .env

# then edit the .env file with your model endpoint and tool provider
```


## Rollout

```bash
# QA
uv run rollout aime --year 2026
uv run rollout gpqa
uv run rollout math

# Web search (Deep Research)
uv run rollout gaia
uv run rollout browsecomp
uv run rollout webwalkerqa

# Abstention
uv run --extra agent_abstain rollout agent_abstain
uv run --extra webshop_abstain rollout webshop_abstain

# SWE (docker required)
uv run rollout swe_bench \
  --limit 2 \
  --concurrency 2

# see more options
uv run rollout <benchmark> --help
```

| Benchmarks | Default turns | Scoring |
| --- | --- | --- |
| AIME / GPQA / MATH | 1 | Integer / choice / symbolic equivalence |
| GAIA | 16 | Normalized answer |
| BrowseComp / WebWalkerQA | 16 | LLM judge |
| SWE-bench Verified (Easy) | 40 | Unit tests |
| Agent Abstain | 30 | Successful critical tools + LLM judge |
| WebShop Abstain | 30 | Purchase receipts + LLM judge |

## Scoring

```bash
uv run scoring <benchmark> outputs/<benchmark>/<run>

# see more options
uv run scoring <benchmark> --help
```

## Outputs

```text
outputs/<benchmark>/<model>_<timestamp>/
├── config.json                           # rollout configuration
├── trajectories/<encoded-id>.json        # agent trajectories
├── environment_outputs/<encoded-id>/     # (Optional) environment status
└── scores/<timestamp>/
    ├── config.json                       # scoring configuration
    ├── trajectories.json
    └── summary.json                      # rollout and scoring results
```

## Layout

```text
src/eval/
├── agent.py            # agent = model + harness (ReAct loop and episode runner)
├── model/              # model providers, request parameters, and CLI/param utils
├── harness/            # harness tools, environments, and CLI/storage utils
└── benchmarks/
    ├── _common/        # shared benchmark CLI, task runner, resume, storage, and metrics helpers
    └── <name>/
        ├── rollout.py      # benchmark-specific rollout
        ├── scoring.py      # benchmark-specific scoring
        └── data/           # dataset and benchmark-specific assets
```
