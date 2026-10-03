# AIME

Simple one-shot eval for [AIME 2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024), [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) and [AIME 2026](https://huggingface.co/datasets/MathArena/aime_2026), 30 problems each with integer answers.

The reply is graded by exact integer match, parsed from the final `{"answer": 123}` JSON.

## Run

Setup python env:

```bash
uv sync
```

Setup model endpoint (OpenAI-compatible) in `.env`:

```bash
mv .env.example .env
```

Run AIME (default 2026):

```bash
uv run main.py

# other years:
uv run main.py --year 2024
uv run main.py --year 2025

# see more options with:
uv run main.py --help
```

All results will be stored in `outputs` folder.
