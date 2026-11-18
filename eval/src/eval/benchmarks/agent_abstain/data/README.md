# Agent Abstain

`agent_abstain.json` contains 526 tasks, including prompts, initial states, labels, and tool overrides.
`environments/` and `runtime/` retain the 42 original simulators; only import prefixes change.
Source: [AgentAbstain](https://github.com/AntiQuality/agentabstain), dataset `antiquality/agentabstain`
at revision `842228426c2a703347396501af61c7890972c7ee`. See `LICENSE`.

```bash
uv run --extra agent_abstain eval-inference agent_abstain \
  --category critical_tool_failure --task-id preview_001
uv run eval-scoring agent_abstain outputs/agent_abstain/<run>
```
