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
# single-turn bench
uv run aime --year 2026 --enable-thinking
uv run gpqa --enable-thinking
uv run math --enable-thinking

# multi-turn bench
uv run browsecomp --enable-thinking
uv run gaia --enable-thinking --search-provider deepseek --fetch-provider deepseek
uv run webwalkerqa --enable-thinking

# see more options:
uv run <bench> --help
```

Results will be stored in `outputs/<benchmark>/<timestamp>_<model>/`.

## Layout

```
eval/
├── data/<bench>/       # dataset of each bench
├── src/eval/
│   ├── core/
│   │   ├── model.py    # streaming model call, ModelParams/Context, per-call capture
│   │   ├── harness.py  # tool use and agent framework
│   │   ├── runner.py   # run loop, run dir/config, records, summary, common CLI args
│   │   └── utils.py    # neutral helpers shared by benchmarks
│   ├── aime.py
│   ├── browsecomp.py
│   └── <bench>.py      # other bench
├── tests/              # unit tests
└── docker-compose.yml  # start tool server locally
```

per benchmark module include:

- `parse_args`: arguments parser
- `load_data`: data loader
- `scorer`: score the prediction against the ground truth
- `solve`: solver of one sample
- `amain`: (optional) manage extra async resources
- `main`: running control
