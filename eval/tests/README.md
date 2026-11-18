# Tests

```bash
uv run --all-extras python -m unittest discover -s tests
uv run --all-extras python -m unittest discover -s tests -p test_mcp_protocol.py
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src tests
uv run python tests/live_tools.py --all
uv run --all-extras python tests/smoke.py --limit 1
uv run rollout swe_bench --subset easy --limit 2 --sample-concurrency 2 --concurrency 2
```

The MCP audit covers all benchmark entrypoints with real protocol sessions, fake model turns, and fixture web, Docker, and WebShop backends.

Score an existing rollout:

```bash
uv run scoring aime outputs/aime/<run>
uv run --extra swe_bench scoring swe_bench outputs/swe_bench/<run> --sample-concurrency 2
```
