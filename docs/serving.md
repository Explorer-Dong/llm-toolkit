# LLM Serving

We serve models with vLLM / SGLang in Docker.

## Start server

```bash
export API_KEY=sk-vincent
export MODEL_PATH=/kwkj-k8s/llm_team/ztj/MULTI_DISTILL_8.10/MOPD_GOPD/eval/models
export MODEL_FOLDER=hybrid_tp8_sp4_40k_moe2048_sync_step103
export DOCKER_CONTAINER_NAME=local-hybrid_tp8_sp4_40k_moe2048_sync_step103
export MODEL_NAME=hybrid_tp8_sp4_40k_moe2048_sync_step103
export TOOL_CALL_PARSER=qwen3_coder
export REASONING_PARSER=qwen3
```

vLLM:

```bash
docker run -d \
  --name "$DOCKER_CONTAINER_NAME" \
  --gpus '"device=4,5,6,7"' \
  -v $MODEL_PATH:/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
    --model /models/$MODEL_FOLDER \
    --served-model-name $MODEL_NAME \
    --api-key $API_KEY \
    --tensor-parallel-size 4 \
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
  lmsysorg/sglang:v0.5.16 \
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
