# LLM Serving

Serve models for high-performance inference.

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
export API_KEY=sk-vincent
export MODEL_PATH=/data/dongwenjie/llm-toolkit/_models/Qwen3.5-9B
export MODEL_FOLDER=Qwen3.5-9B
export DOCKER_CONTAINER_NAME=local-qwen3.5-9b
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
> Let agent help you setup binary is a good choice, as runtime-environment is very complex.

Example from official docs:

```bash
uv venv --python 3.12 --seed --managed-python
source .vene/bin/activate

# vllm
uv pip install vllm --torch-backend=auto

# sglang
uv pip install sglang --prerelease=allow
```

### Start inference engine

```bash
export API_KEY=sk-vincent
export MODEL_PATH=/data/dongwenjie/llm-toolkit/_models/Qwen3.5-9B
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
  --gpu-memory-utilization 0.90 \
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
  --mem-fraction-static 0.90 \
  --trust-remote-code \
  --tool-call-parser $TOOL_CALL_PARSER \
  --reasoning-parser $REASONING_PARSER
```

## Check server

```bash
# check model list
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY" | jq

# check simple request
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
