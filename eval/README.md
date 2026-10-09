# eval

Unified evaluation harness for the LLM benchmarks.

## Run

Setup env:

```bash
uv sync --group dev
cp .env.example .env   # fill in the keys
docker compose up -d   # crawl4ai (used by web read tool)
```

Start evaluation:

```bash
uv run aime --year 2026 --enable-thinking
uv run gpqa --enable-thinking
uv run gaia --enable-thinking
uv run browsecomp --enable-thinking
uv run webwalkerqa --enable-thinking
uv run math --enable-thinking

# see more options:
uv run <bench> --help
```

Results will be stored in `outputs/<benchmark>/<timestamp>_<model>/`.

## Layout

```
eval/
├── src/eval/
│   ├── core/
│   │   ├── model.py    # streaming model call, ModelParams/Context, per-call capture
│   │   ├── harness.py  # Tool, serper key pool + search/read, ReAct loop
│   │   ├── runner.py   # run loop, run dir/config, records, summary, common CLI args
│   │   └── utils.py    # neutral helpers shared by benchmarks
│   ├── aime.py
│   ├── browsecomp.py
│   └── <bench>.py      # other bench
├── data/<bench>/       # dataset of each bench
├── outputs/<benchmark>/<timestamp>_<model>/
└── docker-compose.yml  # start tool server locally
```

per benchmark module include:

- `parse_args`: arguments parser
- `load_data`: data loader
- `scorer`: score the prediction against the ground truth
- `solve`: solver of one sample
- `amain`: (optional) manage extra async resources
- `main`: running control
