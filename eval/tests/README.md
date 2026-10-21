# tests

## Offline code tests

From `eval/`:

```bash
uv run python -m unittest discover -s tests
```

## Real API connectivity

Make sure the key has been prepared.

From `eval/`:

```bash
uv run python tests/live_tools.py --tool serper-search
uv run python tests/live_tools.py --tool deepseek-search --tool deepseek-fetch
uv run python tests/live_tools.py --all
```
