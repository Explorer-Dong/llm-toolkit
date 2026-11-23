# Model serving

Serve models with engine for high-performance inference.

## Prepare model weights

Download pre-trained model weights:

```bash
# from huggingface
uvx hf download Qwen/Qwen3.5-9B --local-dir /path/to/Qwen3.5-9B

# from modelscope
uvx modelscope download Qwen/Qwen3.5-9B --local_dir /path/to/Qwen3.5-9B
```

## Serve with docker

### (Optional) Persist image

> [!tip]
>
> Skip this step if you have a good network and a fixed developing environment.
>
> Otherwise, we can persist [vllm](https://hub.docker.com/r/vllm/vllm-openai/tags) or [sglang](https://hub.docker.com/r/lmsysorg/sglang/tags) image.

non-compression:

```bash
# pull image
docker pull vllm/vllm-openai:v0.30.0-cu129
docker pull lmsysorg/sglang:v0.5.19-cu129

# save image
docker save vllm/vllm-openai:v0.30.0-cu129 -o vllm-openai-v0.30.0-cu129.tar
docker save lmsysorg/sglang:v0.5.19-cu129 -o sglang-v0.5.19-cu129.tar

# load image
docker load -i vllm-openai-v0.30.0-cu129.tar
docker load -i sglang-v0.5.19-cu129.tar
```

compression (with limited storage):

```bash
# pull image
docker pull vllm/vllm-openai:v0.30.0-cu129
docker pull lmsysorg/sglang:v0.5.19-cu129

# save image
docker save vllm/vllm-openai:v0.30.0-cu129 | zstd -T0 -o vllm-openai-v0.30.0-cu129.tar.zst
docker save lmsysorg/sglang:v0.5.19-cu129 | zstd -T0 -o sglang-v0.5.19-cu129.tar.zst

# load image
zstd -dc vllm-openai-v0.30.0-cu129.tar.zst | docker load
zstd -dc sglang-v0.5.19-cu129.tar.zst | docker load
```

### Start inference engine

```bash
export API_KEY=sk-xxx
export MODEL_PATH=/path/to/Qwen3.5-9B
export MODEL_FOLDER=Qwen3.5-9B
export DOCKER_CONTAINER_NAME=local_qwen3.5-9b
export MODEL_NAME=Qwen3.5-9B
export TOOL_CALL_PARSER=qwen3_coder
export REASONING_PARSER=qwen3
```

vLLM:

```bash
docker run -d \
  --name "$DOCKER_CONTAINER_NAME" \
  --gpus '"device=0,1"' \
  -v $MODEL_PATH:/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:v0.30.0-cu129 \
    --model /models/$MODEL_FOLDER \
    --served-model-name $MODEL_NAME \
    --api-key $API_KEY \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --tool-call-parser $TOOL_CALL_PARSER \
    --reasoning-parser $REASONING_PARSER
```

SGLang:

```bash
docker run -d \
  --name "$DOCKER_CONTAINER_NAME" \
  --gpus '"device=4,5,6,7"' \
  -v $MODEL_PATH:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  lmsysorg/sglang:v0.5.19-cu129 \
  sglang serve \
    --model-path /models/$MODEL_FOLDER \
    --served-model-name $MODEL_NAME \
    --api-key $API_KEY \
    --host 0.0.0.0 \
    --port 8000 \
    --tp-size 4 \
    --allow-auto-truncate \
    --dtype bfloat16 \
    --mem-fraction-static 0.90 \
    --trust-remote-code \
    --tool-call-parser $TOOL_CALL_PARSER \
    --reasoning-parser $REASONING_PARSER
```

## Serve with binary

### Install binary

> [!tip]
>
> Remember, engine depends on torch and cuda, torch depend on cuda, and cuda depend on nvidia driver and glibc. We need to verify the highest supported cuda version on your nvidia driver and glibc, then install engine with targeted torch backend.

For example, we need to install vLLM on Ubuntu-22.04. First, check the nvidia driver version `570.xxx.xxx`, then check the glibc version `glibc=2.35`, the highest supported cuda version is `12.9`, so we should install vLLM with:

```bash
uv venv --python 3.12
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" --torch-backend=cu129

# you can also persist the wheel, then
uv pip install ./vllm-0.30.0+cu129-*.whl --torch-backend=cu129
```

Verify installation:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import vllm; print(vllm.__version__)"
```

### Start inference engine

```bash
export API_KEY=sk-xxx
export MODEL_PATH=/path/to/Qwen3.5-9B
export MODEL_NAME=Qwen3.5-9B
export TOOL_CALL_PARSER=qwen3_coder
export REASONING_PARSER=qwen3
```

vLLM:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --api-key $API_KEY \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --tool-call-parser $TOOL_CALL_PARSER \
  --reasoning-parser $REASONING_PARSER
```

SGLang:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
sglang serve \
  --model-path $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --api-key $API_KEY \
  --host 0.0.0.0 \
  --port 8000 \
  --tp-size 2 \
  --allow-auto-truncate \
  --dtype bfloat16 \
  --mem-fraction-static 0.95 \
  --trust-remote-code \
  --tool-call-parser $TOOL_CALL_PARSER \
  --reasoning-parser $REASONING_PARSER
```

## Verify service

```bash
# verify model list
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY" | jq

# verify simple request
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
       "model": "'$MODEL_NAME'",
       "messages": [
          {"role": "system", "content": "Reply in Chinese."},
          {"role": "user", "content": "Introduce yourself briefly"}
        ]
      }' | jq
```
