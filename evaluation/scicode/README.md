# SciCode Bench

> 这是 [SciCode Bench (Official)](https://github.com/scicode-bench/SciCode) 的官方实现，附加了一些日志处理的逻辑。

配置环境：

```bash
uv sync
source .venv/bin/activate
```

下载测试集：

```bash
gdown https://drive.google.com/file/d/17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW/view?usp=drive_link -O eval/data/test_data.h5
```

启动评测：

```bash
cd eval/inspect_ai

inspect eval scicode.py \
  --model vllm/BigBang-v1 \
  --model-base-url http://127.0.0.1:8000/v1 \
  --temperature 1.0 \
  --max-connections 8 \
  --top-p 0.95 \
  --max-tokens 102400 \
  --log-format json \
  --time-limit 1800 \
  --reasoning-effort max \
  -T split=test \
  -T with_background=True \
  -M api_key=sk-vincent
```

冒烟测试时可以添加 `--limit 4` 来限制 main problem 的数量。

> `inspect` 的更多用法见其官网：<https://inspect.aisi.org.uk/>。
>
> 评测后的日志处理移步 [eval/inspect_ai](./eval/inspect_ai/README.md) 作进一步了解。
