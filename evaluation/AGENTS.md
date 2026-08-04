# LLM Evaluation System

## Goal

- 模型以 OpenAI-compatible API 的形式呈现。
- 需要沙箱环境的基准测试在 `sandbox` 对应的子目录，其余均在根目录维护。
- 评测者只需要运行一个 Python 入口文件 `src/main.py`。
- 不同 benchmark 通过 `--benchmark` 参数指定。
- 优先保证结构清晰、能跑通、易扩展。
- 不要做 Web UI、数据库、复杂分布式调度、多用户系统。

## Engineering harness

- 唯一入口文件：`src/main.py`
- 包管理：必须使用 `uv` 管理 Python 包。
- 沙箱环境：必须使用 `docker-compose.yml` 管理沙箱环境。
- 进度展示：运行 `src/main.py` 后，必须显示 tqdm 进度条。
- 配置管理：使用根目录的 `config.yml` 保存默认配置。

`src/main.py` 至少支持：

- `--benchmark [name]`：具体 Benchmark 名称。
- `--model [name]`：模型名称。
- `--base-url [url]`：模型服务 URL。
- `--api-key [key]`：模型服务 API KEY。
- `--split`：是否划分数据集。
- `--limit [number]`：测试样例数量。
- `--output-dir [path]`：输出路径。
- `--temperature [number]`：温度。
- `--top-p [number]`：累计概率。
- `--top-k [number]`：预选 token 数。
- `--max-tokens [number]`：最大输出 token 数。
- `--num-workers [number]`：进程数。
- `--seed [number]`：种子数。
- `--resume`：是否复用结果。
- `--dry-run`：是否空跑。

## Code style

- 简单。
- 显式。
- 少抽象。
- 类型标注明确。
- 不使用 Optional，使用 `T | None`。
- 每个 benchmark 一个文件。
- 所有输出都落盘。
- 所有错误都记录。
- 不允许静默失败。
- 优先可读性，不要过度封装。
