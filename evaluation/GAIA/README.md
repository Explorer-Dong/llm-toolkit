# GAIA

Simple ReAct for [GAIA text-only 103](https://github.com/RUC-NLPIR/WebThinker/blob/main/data/GAIA/dev.json).

Native function calling: `search` and `read`.

## Run

Setup python env:

```bash
uv sync
```

Setup model endpoint (OpenAI-compatible, support FC), search engine ([Serper](https://serper.dev)) and page crawl ([Crawl4AI](https://github.com/unclecode/crawl4ai)) in `.env`:

```bash
mv .env.example .env
```

Start Crawl4AI server:

```bash
docker compose up -d
```

Run GAIA:

```bash
# without thinking
uv run main.py

# enable thinking
uv run main.py --enable-thinking --max-tokens 81920 --timeout 1800 --task-timeout 7200

# see more options with:
uv run main.py --help
```

All results will be stored in `outputs` folder.
