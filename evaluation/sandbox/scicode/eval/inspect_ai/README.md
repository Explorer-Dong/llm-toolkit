# 数据分析与处理

## Stage 1. Data leak

通过数据泄露的方式，直接让目标模型学习更强模型的正确解题方式。

### 轨迹转换：Inspect -> OpenAI

将 inspect 格式转换为 llama-factory 支持的 OpenAI 格式，以支持 SFT：

```bash
python inspect2lmf.py \
  --input logs/2026-07-27T15-30-23-00-00_scicode_djMFUYadahin3S6GitvvHw.json \
  --output logs-lmf/2026-07-27T15-30-23-00-00_scicode_djMFUYadahin3S6GitvvHw.json
```

### 运行报告

将 `logs-lmf/` 目录下的一份转换后的 LlamaFactory 日志追加到固定报告 `eval/inspect_ai/tmp/run_report.md`：

```bash
cd eval/inspect_ai

python analysis.py \
  --log logs-lmf/2026-07-27T15-30-23-00-00_GLM-5.2-FP8.json
```

脚本只接受 `logs-lmf/` 目录直接包含的 JSON 文件，并从每条记录的 `source.model` 读取模型名称，在报告中添加或更新对应列。每行对应一个完整的小题编号，例如 `11.1`；转换日志中出现的 `pass` 小题显示为 `✅`，未出现在该日志中的小题显示为 `❌`。转换日志未来若包含 `fail` 或 `time out` 状态，报告会分别显示 `❌` 或 `time out`。

## Stage 2. Data generalize

TODO

通过分析更强模型的正确解题方式，合成泛化数据以提升模型的真实能力。
