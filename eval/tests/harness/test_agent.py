"""The ReAct core normalizes runtime failures and retains metadata."""

import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from eval import agent
from eval.agent import Task
from eval.harness.tool import ToolBox
from eval.model import AssistantTurn

__all__ = []
PARAMS: dict[str, Any] = {"model": "model", "temperature": 0, "top_p": 1, "max_tokens": 1024, "seed": 42}


class ReactCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_error_records_partial_turn_and_status(self):
        turn = AssistantTurn("partial", error=RuntimeError("request failed"))
        with (
            self.assertLogs("eval.agent", level="ERROR"),
            patch.object(agent, "call_model", new_callable=AsyncMock, return_value=turn),
        ):
            trace = await agent.react(Task("one", "system", "prompt"), MagicMock(), ToolBox(), PARAMS)
        self.assertEqual(trace.episode_status, "error")
        self.assertEqual(trace.error, "RuntimeError: request failed")
        self.assertEqual(trace.messages[-1]["content"], "partial")
        self.assertTrue(trace.messages[-1]["incomplete"])

    async def test_default_task_is_one_turn_and_retains_input_metadata(self):
        task = Task("one", "system", "prompt", meta={"tags": ["a"]})
        with patch.object(agent, "call_model", new_callable=AsyncMock, return_value=AssistantTurn("answer")):
            trace = await agent.react(task, MagicMock(), ToolBox(), PARAMS)
        self.assertEqual((trace.turns, trace.episode_status, trace.final_answer), (1, "final_answer", "answer"))
        trace.meta["tags"].append("b")
        self.assertEqual(task.meta, {"tags": ["a"]})


if __name__ == "__main__":
    unittest.main()
