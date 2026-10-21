# eval

Unified evaluation harness for the LLM benchmarks.

## Run

From `eval/`:

Setup env:

```bash
uv sync --group dev
cp .env.example .env   # fill in the keys
docker compose up -d   # (Optional) needed only for the default fetch service Crawl4AI
```

Start evaluation:

```bash
# single-turn benchmark
uv run aime --year 2026 --enable-thinking
uv run gpqa --enable-thinking
uv run math --enable-thinking

# multi-turn benchmark
uv run browsecomp --enable-thinking
uv run gaia --enable-thinking --search-provider deepseek --fetch-provider deepseek
uv run webwalkerqa --enable-thinking

# see more options:
uv run <benchmark> --help
```

Results will be stored in `outputs/<benchmark>/<timestamp>_<model>/`.

## Layout

```
eval/
├── data/<benchmark>/       # dataset of each benchmark
├── src/eval/
│   ├── core/
│   │   ├── harness.py  # tools, agent paradigms
│   │   ├── model.py    # model CLI args, request parameters, streaming model calls
│   │   ├── runner.py   # before eval (prepare folder), during eval (asynchronous run, record result), after eval (summary)
│   │   └── utils.py    # shared helpers
│   ├── aime.py
│   ├── browsecomp.py
│   └── <benchmark>.py      # other benchmarks
├── tests/              # unit tests
└── docker-compose.yml  # start tool server locally
```

per benchmark module include:

- `parse_args`: arguments parser
- `load_tasks`: data loader
- `scorer`: score the prediction against the ground truth
- `solve_task`: solver of one sample
- `main`: running control
