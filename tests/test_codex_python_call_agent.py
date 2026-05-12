import json
import unittest

from purple_car_bench_agent_codex.codex_client import CodexMalformedResponseError
from purple_car_bench_agent_codex_python.python_call_agent import (
    extract_python_call_code,
    parse_python_calls,
    parse_python_next_action_output,
)


class PythonCallParserTest(unittest.TestCase):
    def test_respond_call(self) -> None:
        self.assertEqual(
            parse_python_calls('respond("Done.")'),
            {"action": "respond", "content": "Done."},
        )
        self.assertEqual(
            parse_python_next_action_output(
                json.dumps({"python_code": 'respond("Done.")'})
            ),
            {"action": "respond", "content": "Done."},
        )
        self.assertEqual(
            parse_python_next_action_output('respond("Done.")'),
            {"action": "respond", "content": "Done."},
        )

    def test_fenced_python_with_private_prose(self) -> None:
        text = (
            "I need to ask for a specific setting before changing the shade.\n\n"
            "```python\n"
            'respond("What percentage should I set it to?")\n'
            "```"
        )
        self.assertEqual(
            extract_python_call_code(text),
            'respond("What percentage should I set it to?")',
        )
        self.assertEqual(
            parse_python_next_action_output(text),
            {
                "action": "respond",
                "content": "What percentage should I set it to?",
            },
        )

    def test_fenced_python_tool_call(self) -> None:
        text = (
            "Action:\n"
            "```python\n"
            "open_close_sunshade(percentage=50)\n"
            "```"
        )
        self.assertEqual(
            parse_python_next_action_output(text),
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            },
        )

    def test_single_tool_call(self) -> None:
        self.assertEqual(
            parse_python_calls("open_close_sunshade(percentage=50)"),
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            },
        )

    def test_tool_call_with_premature_respond_recovers_as_tool_call(self) -> None:
        text = (
            "```python\n"
            "open_close_sunshade(percentage=50)\n"
            'respond("Done, I set the sunshade to 50%.")\n'
            "```"
        )
        self.assertEqual(
            parse_python_next_action_output(text),
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            },
        )

    def test_multiple_tool_calls(self) -> None:
        parsed = parse_python_calls(
            "get_user_preferences("
            'preference_categories={"vehicle_settings": {"vehicle_settings": True}}'
            ")\n"
            "open_close_sunshade(percentage=50)"
        )
        self.assertEqual(
            parsed,
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "get_user_preferences",
                        "arguments": {
                            "preference_categories": {
                                "vehicle_settings": {"vehicle_settings": True}
                            }
                        },
                    },
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    },
                ],
            },
        )

    def test_unknown_tools_and_parameters_pass_through(self) -> None:
        self.assertEqual(
            parse_python_calls('unknown_removed_tool(foo="bar")'),
            {
                "action": "tool_calls",
                "tool_calls": [
                    {"tool_name": "unknown_removed_tool", "arguments": {"foo": "bar"}}
                ],
            },
        )
        self.assertEqual(
            parse_python_calls("open_close_sunshade(removed_parameter=True)"),
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"removed_parameter": True},
                    }
                ],
            },
        )

    def test_rejects_non_call_python(self) -> None:
        invalid_examples = [
            "import os",
            "x = open_close_sunshade(percentage=50)",
            "for x in [1]:\n    open_close_sunshade(percentage=x)",
            "tools.open_close_sunshade(percentage=50)",
            "open_close_sunshade(percentage=x)",
            "open_close_sunshade(50)",
        ]
        for example in invalid_examples:
            with self.subTest(example=example):
                with self.assertRaises(CodexMalformedResponseError):
                    parse_python_calls(example)

    def test_rejects_ambiguous_or_non_python_markdown(self) -> None:
        invalid_examples = [
            (
                "```python\nrespond(\"One\")\n```\n"
                "```python\nrespond(\"Two\")\n```"
            ),
            "```javascript\nrespond(\"Done\")\n```",
            (
                "Some prose without a fence.\n"
                "respond(\"Done\")"
            ),
        ]
        for example in invalid_examples:
            with self.subTest(example=example):
                with self.assertRaises(CodexMalformedResponseError):
                    parse_python_next_action_output(example)

    def test_legacy_json_fence_still_works(self) -> None:
        text = '```json\n{"python_code": "open_close_sunshade(percentage=50)"}\n```'
        self.assertEqual(
            parse_python_next_action_output(text),
            {
                "action": "tool_calls",
                "tool_calls": [
                    {
                        "tool_name": "open_close_sunshade",
                        "arguments": {"percentage": 50},
                    }
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
