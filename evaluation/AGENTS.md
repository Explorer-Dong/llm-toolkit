# Evaluation

## One-shot benchmark

AIME 子文件夹下的代码风格与文件组织是最符合我开发习惯的 One-shot 评测代码。所有 One-shot 类型的 benchmark 在复现时需要借鉴其实现。

## Agentic benchmark

基本代码风格与 One-shot benchmark 一致，额外部分包括但不限于：

- 所有的网络请求都需要有指数退让时间的重试机制。
- 工具调用逻辑作为函数参数传入而非提示词约束。
