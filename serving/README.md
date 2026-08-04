# LLM Serving

LLM 推理依赖复杂的环境，这里选择直接基于 Docker 容器进行，避免所有环境配置。

## vLLM

示例配置（路径以及其余参数视本地情况自行修改）：

```bash
export API_KEY=sk-vincent
export MODEL_NAME=MiniMind3

docker run -d \
  --name minimind3 \
  --runtime nvidia \
  --gpus '"device=0,1"' \
  -v /cpfs01/llm_team/dwj/.cache/huggingface:/root/.cache/huggingface \
  -v /cpfs01/llm_team/dwj/llm-mini/saves:/models \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /models/minimind-3 \
  --served-model-name $MODEL_NAME \
  --api-key $API_KEY \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code
```

## SGLang

示例配置（路径以及其余参数视本地情况自行修改）：

```bash
export API_KEY=sk-vincent

# GLM-5.2
# export MODEL_PATH=/cpfs01/llm_team/models
# export MODEL_FOLDER=GLM-5.2-FP8
# export DOCKER_CONTAINER_NAME=sglang-glm52-fp8
# export MODEL_NAME=GLM-5.2-FP8
# export TOOL_CALL_PARSER=glm47
# export REASONING_PARSER=glm45

# Qwen3.5-35B-A3B
# export MODEL_PATH=/cpfs01/llm_team/models
# export MODEL_FOLDER=Qwen3.5-35B-A3B
# export DOCKER_CONTAINER_NAME=sglang-qwen3.5-35b-a3b
# export MODEL_NAME=Qwen3.5-35B-A3B
# export TOOL_CALL_PARSER=qwen3_coder
# export REASONING_PARSER=qwen3

# Qwen3.5-35B-A3B (SFT)
export MODEL_PATH=/cpfs01/llm_team/dwj/data_filter/LlamaFactory/saves/qwen3.5-35b-a3b/sft/lora-0729-scicode-2
export MODEL_FOLDER=lora_sft_merged
export DOCKER_CONTAINER_NAME=sglang-qwen-scicode
export MODEL_NAME=Qwen3.5-35B-A3B-scicode-2
export TOOL_CALL_PARSER=qwen3_coder
export REASONING_PARSER=qwen3

docker run -d \
  --name "$DOCKER_CONTAINER_NAME" \
  --runtime nvidia \
  --gpus '"device=0,1,2,3"' \
  --platform linux/arm64 \
  -v $MODEL_PATH:/models \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  lmsysorg/sglang:v0.5.15.post1-cu130 \
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
       "model": "$MODEL_NAME",
       "messages": [
          {"role": "system", "content": "Reply in Chinese."},
          {"role": "user", "content": "Introduce yourself briefly"}
        ]
      }' | jq
```
