# LLM SFT Pipeline

SFT 来提升模型的 Coding 和 Code Agent 能力。

| Method | SWE-bench Lite (Pass@3) | SciCode |
| :--- | :---: | :---: |
| Qwen3.5-35B-A3B Base | 69.3 | - |
| LoRA on [Nemotron-SFT-SWE-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-SWE-v2) Agentless | 66.7 | - |
| LoRA on Nemotron-SFT-SWE-v2 SWE | 56.0 | - |
| LoRA on GLM-5.2 trajectories of scicode (leak data) | - | - |
| LoRA on GLM-5.2 trajectories of scicode (generalize data) | - | - |
