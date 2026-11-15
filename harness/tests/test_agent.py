"""The ReAct core borrows resources and propagates runtime failures."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from harness import Task, Trajectory, agent, model
from harness.tool import ToolBox

__all__ = []


class ReactCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_error_propagates_with_partial_record(self):
        trace = Trajectory("one")
        ctx = model.Context(MagicMock(), model.ModelParams("model", 0, 1, 1024, 42), asyncio.Semaphore(1))
        turn = model.AssistantTurn("partial", error=RuntimeError("request failed"))
        with (
            patch.object(agent, "call_model", new_callable=AsyncMock, return_value=turn),
            self.assertRaisesRegex(RuntimeError, "request failed"),
        ):
            await agent.react(Task("one", "system", "prompt"), ctx, ToolBox(), trajectory=trace)
        self.assertEqual(trace.messages[-1]["content"], "partial")
        self.assertTrue(trace.messages[-1]["incomplete"])
        self.assertIsNone(trace.error)

    async def test_default_task_is_one_turn_and_retains_input_metadata(self):
        task = Task("one", "system", "prompt", meta={"tags": ["a"]})
        ctx = MagicMock()
        with patch.object(agent, "call_model", new_callable=AsyncMock, return_value=model.AssistantTurn("answer")):
            trace = await agent.react(task, ctx, ToolBox())
        self.assertEqual((trace.turns, trace.stop_reason, trace.final_answer), (1, "final_answer", "answer"))
        trace.meta["tags"].append("b")
        self.assertEqual(task.meta, {"tags": ["a"]})


if __name__ == "__main__":
    unittest.main()
