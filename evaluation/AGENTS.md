# LLM Evaluation System

## Goal

- 模型以 OpenAI-compatible API 的形式呈现。
- **每一个 benchmark 是 `evaluation/` 下的一个自包含子文件夹**，不再区分根目录/standalone/sandbox 类别。
- 每个 bench 自带 uv 环境（`pyproject.toml` + `uv sync`）、独立入口、自己的 README 与数据目录（`data/`，gitignore）。
- 需要执行不可信代码的 bench（如 `scicode`、`nl2repo`）在自己的文件夹内用 `docker-compose.yml` 管理沙箱；不需要的（如 `aime26`、`gpqa_diamond`、`webwalkerqa`）直接原生运行。
- 优先保证结构清晰、能跑通、易扩展。
- 不要做 Web UI、数据库、复杂分布式调度、多用户系统。

## Engineering harness

- 每个 bench 一个入口 `main.py`（如需判分等阶段可另有 `score.py`）。
- 包管理：必须使用 `uv`，各 bench 各自管理依赖。
- 进度展示：运行入口后必须显示 tqdm 进度条。
- 输出落盘：每个 run 生成 `outputs/<时间戳>_<bench>_<model>/`，含 `config.json`、`predictions.jsonl`、`errors.jsonl`、`summary.json`（或 `report.json`）。

`main.py` 至少支持：

- `--model [name]`：模型名称。
- `--base-url [url]`：模型服务 URL。
- `--api-key [key]`：模型服务 API KEY。
- `--split`：数据集划分。
- `--limit [number]`：测试样例数量（`0` = 全量）。
- `--output-dir [path]`：输出路径。
- `--temperature [number]`：温度。
- `--top-p [number]`：累计概率。
- `--top-k [number]`：预选 token 数。
- `--max-tokens [number]`：最大输出 token 数。
- `--num-workers [number]`：并发数。
- `--seed [number]`：种子数。
- `--resume`：复用已完成结果。
- `--dry-run`：空跑（加载数据、写 config、不调模型）。

bench 可按需追加专属参数（如 `webwalkerqa` 的 `--method`、`--max-rounds`）。

## Code style

- 简单。
- 显式。
- 少抽象。
- 类型标注明确。
- 不使用 Optional，使用 `T | None`。
- 每个 benchmark 一个（或少数几个）文件，逻辑内聚在 bench 文件夹内，不做跨 bench 共享抽象。
- 所有输出都落盘。
- 所有错误都记录。
- 不允许静默失败。
- 优先可读性，不要过度封装。
