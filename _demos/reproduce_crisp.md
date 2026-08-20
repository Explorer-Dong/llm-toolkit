# CRISP 复现手册

> 学生和教师使用相同的基座模型，通过给教师模型添加约束提示词获得指导信号，来指导学生减少推理内容。

克隆代码：

```bash
git clone https://github.com/HJSang/CRISP_Reasoning_Compression.git crisp
cd crisp
```

安装依赖：

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r workspace/requirements.txt

# H 系列显卡
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=4 \
  uv pip install flash-attn==2.8.3 --no-build-isolation
```

接下来根据项目的 README.md 完成数据的配置即可开始蒸馏。

## 2026-08-20 crisp2 全新机器复现

已在 hd03-gpu2-0020 的 crisp2 独立目录实测。版本固定为 Python 3.10.12、CUDA 12.8、Torch 2.9.1+cu128、SGLang 0.5.9、sgl-kernel 0.3.21、VERL commit deec5d02ba679266f5f0293530f387186908238d、Ray 2.57.0、Transformers 4.57.1；完整清单见 crisp2/workspace/requirements.txt。

关键版本陷阱：sglang 0.5.9 要求 Transformers 4.57.1；cuda-python 12.9.0、cuda-bindings 12.9.7、nvidia-cutlass-dsl 4.3.5 配 Torch 2.9.1/cu128；Python 3.10 需要固定 grpcio-health-checking/grpcio-reflection 1.83.0 和 protobuf 7.35.1。不要自由解析 quack-kernels，否则可能升级到 Torch 2.13/CUDA 13。FlashAttention 未验证成功时使用 SDPA，并传 actor_rollout_ref.model.use_remove_padding=false 与 +actor_rollout_ref.model.override_config.attn_implementation=sdpa。

在 crisp2 执行一键准备：

```bash
PYTHON_BIN=python3.10 ./reproduce.sh
```

脚本创建隔离 .venv、安装依赖、执行 pip check、处理数据并生成四个 length-prune 变体。预期 DAPO 训练 17,398 条，MATH-500 500 条，AIME24/25 各 30 条；每个变体训练/验证为 13,918/3,480 条。

单步训练冒烟：

```bash
MODEL_PATH=/kwkj-k8s/llm_team/hanjiaxuan/EOPD/models/Qwen3-8B \\
CUDA_VISIBLE_DEVICES=4,5,6,7 RUN_SMOKE=1 ./reproduce.sh
```

固定 4 卡 FSDP、TP=2、8 条样本、1 step、512 token、GPU_MEM_UTIL=0.4；成功判据是日志出现 training/global_step=1、opsd/loss、opsd/grad_norm、opsd/num_tokens。本次实测为 loss 0.03173787659034133、grad_norm 3.513873338699341、num_tokens 1026，验证 JSONL 在 outputs/fresh_smoke2/detailed_logs/val_generations/step_000001.jsonl。

训练前脚本只停止原本正在运行的 sglang-kat-coder-v2.5-dev，并在退出时恢复；容器不存在时不会猜测创建。只使用 GPU 4--7。step 1 指标完成后 Ray teardown 可能出现 DataLoader worker SIGKILL；若 step 前出现则不算成功，小实验可将 data.dataloader_num_workers=0。
