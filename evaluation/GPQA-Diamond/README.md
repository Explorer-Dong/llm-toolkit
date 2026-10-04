# GPQA-Diamond

Simple single-turn eval for [GPQA-Diamond](https://huggingface.co/datasets/fingertap/GPQA-Diamond), 198 four-choice science questions.

The reply is graded by the chosen letter, parsed from the final `{"answer": "A"}` JSON.

## Run

Setup python env:

```bash
uv sync
```

Setup model endpoint (OpenAI-compatible) in `.env`:

```bash
mv .env.example .env
```

Run GPQA-Diamond:

```bash
uv run main.py

# see more options with:
uv run main.py --help
```

All results will be stored in `outputs` folder.
