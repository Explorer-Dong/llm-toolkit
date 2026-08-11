# NL2Repo Bench

> 这是 [NL2Repo Bench (Official)](https://github.com/multimodal-art-projection/NL2RepoBench/) 的鲁棒性增强实现。

## ToDo

- [ ] 支持设置容器最大内存（boltons 任务会出现内存泄露问题）。
- [ ] 支持设置任务超时时间（jinja、freezegun、dbutils 任务会出现超时问题）。
- [ ] 支持 resume 运行。
- [ ] 支持多次运行，当前每次运行前需要删除上一次运行产生的 result 和 workspaces 文件夹。

## 快速开始

安装并检查 docker 和 docker-buildx-plugin：

```bash
docker version
docker buildx version
```

拉取 OpenHands 运行所需镜像：

```bash
docker pull ghcr.io/all-hands-ai/openhands:0.56
docker pull ghcr.io/all-hands-ai/runtime:0.56-nikolaik
```

安装 Python 依赖：

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

配置 `config.json` 后启动评估：

```bash
python main.py
```

运行结果将会出现在 `result/<task_id>.json` 中，同时 LLM 对每道题的作答情况将会出现在 `workspaces/<task_id>/` 中。重复运行需要删除 `result` 和 `workspaces` 文件夹。

统计评估结果：

```bash
python score.py
```
