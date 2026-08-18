# LLM Serving

LLM 推理依赖复杂的环境，这里选择直接基于 Docker 容器进行，避免所有环境配置。

## vLLM

示例配置（路径以及其余参数视本地情况自行修改）：

```bash
export API_KEY=sk-vincent
export MODEL_NAME=BigBang-v1

docker run -d \
  --name bigbang-v1 \
  --gpus '"device=4,5,6,7"' \
  -v /kwkj-k8s/llm_team/dwj/models:/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
    --model /models/BigBang-v1 \
    --served-model-name $MODEL_NAME \
    --api-key $API_KEY \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

## SGLang

示例配置（路径以及其余参数视本地情况自行修改）：

```bash
export API_KEY=sk-vincent

# Qwen3.5
export MODEL_PATH=/kwkj-k8s/llm_team/dwj/my-llm-toolkit/_models
export MODEL_FOLDER=KAT-Coder-V2.5-Dev
export DOCKER_CONTAINER_NAME=sglang-kat-coder-v2.5-dev
export MODEL_NAME=KAT-Coder-V2.5-Dev
export TOOL_CALL_PARSER=qwen3_coder
export REASONING_PARSER=qwen3

docker run -d \
  --name "$DOCKER_CONTAINER_NAME" \
  --gpus '"device=4,5,6,7"' \
  -v $MODEL_PATH:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  lmsysorg/sglang:v0.5.16 \
  sglang serve \
    --model-path "/models/$MODEL_FOLDER" \
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

## 检查推理服务

```bash
# 检查模型列表
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $API_KEY" | jq

# 简单请求
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
