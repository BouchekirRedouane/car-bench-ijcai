import json
import unittest

from purple_car_bench_agent_codex_planner.planner_agent import (
    PlannerExecutorCARBenchAgentExecutor,
    _build_fallback_private_plan,
    _should_create_private_plan,
    parse_private_plan,
)
from purple_car_bench_agent_codex.codex_client import CodexTurnResult


class _NoopLogger:
    def debug(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass


class _FakeExecutorOnlyClient:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, **kwargs) -> CodexTurnResult:
        self.calls.append(kwargs)
        return CodexTurnResult(
            text=json.dumps(
                {"action": "respond", "content": "Done.", "tool_calls": []}
            ),
            duration_ms=12.0,
            model=kwargs.get("model"),
            reasoning_effort=kwargs.get("reasoning_effort"),
        )


class PlannerAgentStateTest(unittest.TestCase):
    def test_private_plan_is_created_only_after_user_turn(self) -> None:
        self.assertTrue(
            _should_create_private_plan(
                [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "Please close the sunshade."},
                ]
            )
        )
        self.assertFalse(
            _should_create_private_plan(
                [
                    {"role": "assistant", "tool_calls": []},
                    {"role": "tool", "name": "open_close_sunshade"},
                ]
            )
        )

    def test_fallback_plan_matches_private_plan_schema(self) -> None:
        plan = _build_fallback_private_plan(
            [{"role": "tool", "name": "open_close_sunshade"}]
        )
        self.assertEqual(parse_private_plan(json.dumps(plan)), plan)
        self.assertIn("missing_cached_private_plan", plan["risk_flags"])

    def test_tool_result_turn_reuses_active_plan_without_planner_call(self) -> None:
        executor = PlannerExecutorCARBenchAgentExecutor()
        fake_client = _FakeExecutorOnlyClient()
        executor.client = fake_client
        context_id = "ctx-test"
        executor._active_private_plans_by_context[context_id] = (
            _build_fallback_private_plan([])
        )

        parsed, elapsed_ms = executor._call_codex_with_retries(
            context_id=context_id,
            messages=[{"role": "tool", "name": "open_close_sunshade"}],
            tools=[],
            ctx_logger=_NoopLogger(),
        )

        self.assertEqual(parsed, {"action": "respond", "content": "Done."})
        self.assertEqual(elapsed_ms, 12.0)
        self.assertEqual(len(fake_client.calls), 1)
        self.assertNotIn(context_id, executor._active_private_plans_by_context)


if __name__ == "__main__":
    unittest.main()
