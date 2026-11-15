# Tests

```bash
uv run --extra swe_bench python -m unittest discover -s tests
uv run python -m unittest discover -s ../harness/tests
uv run ruff check src tests ../harness/src ../harness/tests
uv run ruff format --check src tests ../harness/src ../harness/tests
uv run ty check src tests ../harness/src ../harness/tests
uv run python tests/live_tools.py --all
uv run --extra swe_bench python tests/smoke.py --limit 1
uv run eval-inference swe_bench --subset easy --limit 2 --sample-concurrency 2 --concurrency 2
```

Score an existing rollout:

```bash
uv run eval-scoring aime outputs/aime/<run>
uv run --extra swe_bench eval-scoring swe_bench outputs/swe_bench/<run> --sample-concurrency 2
```
