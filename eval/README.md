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
uv run gaia
uv run browsecomp
uv run webwalkerqa

# see more options:
uv run <bench> --help
```

Results will store in `outputs/<benchmark>/<timestamp>_<model>/`.

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
│   ├── gaia.py
│   ├── gpqa.py
│   └── webwalkerqa.py
├── data/<bench>/
├── outputs/<benchmark>/<timestamp>_<model>/
└── docker-compose.yml
```

per benchmark module include:

- `parse_args`: arguments parser
- `load_data`: data loader
- `scorer`: judge between the prediction and the ground truth
- `solve`: solver of one sample
- `amain`: (optional) manage extra async resources
- `main`: running control
