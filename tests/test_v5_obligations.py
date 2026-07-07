"""Tests for the v5 obligation engine, ledger stage, and voting logic.

No LLM calls. Run: .venv/bin/python tests/test_v5_obligations.py  (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "track_1_agent_under_test"))

from harness import obligations as O
from harness import verify as V
from harness.config import HarnessConfig
from harness.provenance import ProvenanceLedger

WINDOW_TOOLS = [
    {"function": {"name": "get_vehicle_window_positions",
                  "parameters": {"type": "object", "properties": {}}}},
    {"function": {"name": "set_air_conditioning",
                  "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}}}},
    {"function": {"name": "open_close_window",
                  "parameters": {"type": "object", "properties": {
                      "window": {"type": "string",
                                 "enum": ["DRIVER", "PASSENGER", "DRIVER_REAR",
                                          "PASSENGER_REAR", "ALL"]},
                      "percentage": {"type": "number"}}}}},
]

WINDOW_RULE = [{
    "id": "011", "type": "auto_action", "trigger_tools": ["set_air_conditioning"],
    "requirement": "When turning the AC on, close all windows open more than 20%.",
    "exec": {"read": "get_vehicle_window_positions", "field_pattern": "window_*_position",
             "op": ">", "value": 20,
             "obligation": {"tool": "open_close_window",
                            "args": {"window": "<item>", "percentage": 0}}},
}]


def _msgs(window_result: str):
    return [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Turn on the AC."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vehicle_window_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vehicle_window_positions", "content": window_result},
    ]


STATE = ('{"status": "SUCCESS", "result": {"description": "0=closed", '
         '"window_driver_position": 25, "window_passenger_position": 100, '
         '"window_driver_rear_position": 0, "window_passenger_rear_position": 10}}')

AC_DRAFT = {"tool_calls": [{"function": {"name": "set_air_conditioning",
                                         "arguments": {"on": True}}}]}


def test_engine_computes_exact_obligations():
    """25% and 100% qualify (>20), rears don't -> exactly two computed calls
    with enum-aligned window names."""
    ev = O.evaluate(WINDOW_RULE, WINDOW_TOOLS, _msgs(STATE), {"set_air_conditioning"})
    got = sorted((ob["tool"], ob["args"]["window"], ob["args"]["percentage"])
                 for ob in ev["obligations"])
    assert got == [("open_close_window", "DRIVER", 0),
                   ("open_close_window", "PASSENGER", 0)], got
    assert ev["total_matched"]["011"] == 4


def test_missing_obligations_yield_exact_calls_finding():
    findings = O.check_obligations(AC_DRAFT, _msgs(STATE), WINDOW_TOOLS, WINDOW_RULE)
    assert len(findings) == 1, findings
    assert '"window": "DRIVER"' in findings[0] and '"window": "PASSENGER"' in findings[0]
    assert "DRIVER_REAR" not in findings[0]  # non-qualifying items never demanded


def test_covered_draft_is_clean():
    draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "DRIVER", "percentage": 0}}},
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "PASSENGER", "percentage": 0}}},
    ]}
    assert O.check_obligations(draft, _msgs(STATE), WINDOW_TOOLS, WINDOW_RULE) == []


def test_all_scope_flagged_when_subset_qualifies():
    draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "ALL", "percentage": 0}}},
    ]}
    findings = O.check_obligations(draft, _msgs(STATE), WINDOW_TOOLS, WINDOW_RULE)
    assert any("ALL" in f and "per-item" in f for f in findings), findings


def test_all_accepted_when_every_item_qualifies():
    state = ('{"result": {"window_driver_position": 30, "window_passenger_position": 100, '
             '"window_driver_rear_position": 25, "window_passenger_rear_position": 90}}')
    draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "ALL", "percentage": 0}}},
    ]}
    assert O.check_obligations(draft, _msgs(state), WINDOW_TOOLS, WINDOW_RULE) == []


def test_missing_read_demanded_first():
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Turn on the AC."}]
    findings = O.check_obligations(AC_DRAFT, msgs, WINDOW_TOOLS, WINDOW_RULE)
    assert len(findings) == 1 and "get_vehicle_window_positions" in findings[0], findings


def test_unknown_values_never_produce_obligations():
    state = ('{"result": {"window_driver_position": "unknown", '
             '"window_passenger_position": "unknown", "window_driver_rear_position": 0, '
             '"window_passenger_rear_position": 10}}')
    ev = O.evaluate(WINDOW_RULE, WINDOW_TOOLS, _msgs(state), {"set_air_conditioning"})
    assert ev["obligations"] == [] and len(ev["unknown_fields"]) == 2


def test_invalid_exec_and_unavailable_tools_degrade_silently():
    bad = [dict(WINDOW_RULE[0], exec={"read": "x"})]  # structurally invalid
    assert O.check_obligations(AC_DRAFT, _msgs(STATE), WINDOW_TOOLS, bad) == []
    ghost = [dict(WINDOW_RULE[0],
                  exec=dict(WINDOW_RULE[0]["exec"],
                            obligation={"tool": "no_such_tool", "args": {}}))]
    assert O.check_obligations(AC_DRAFT, _msgs(STATE), WINDOW_TOOLS, ghost) == []


def test_alien_domain_smart_home():
    """Hidden-benchmark proof: an oven-door policy from a fictional smart home
    interprets identically — nothing car-specific in the engine."""
    tools = [
        {"function": {"name": "get_oven_state",
                      "parameters": {"type": "object", "properties": {}}}},
        {"function": {"name": "set_oven_power",
                      "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}}}},
        {"function": {"name": "close_oven_door",
                      "parameters": {"type": "object", "properties": {
                          "door": {"type": "string", "enum": ["MAIN", "TOP"]}}}}},
    ]
    rules = [{"id": "k1", "type": "auto_action", "trigger_tools": ["set_oven_power"],
              "requirement": "When powering the oven, close any door that is open.",
              "exec": {"read": "get_oven_state", "field_pattern": "door_*_state",
                       "op": "==", "value": "open",
                       "obligation": {"tool": "close_oven_door", "args": {"door": "<item>"}}}}]
    msgs = [{"role": "user", "content": "Turn the oven on."},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "get_oven_state", "arguments": "{}"}}]},
            {"role": "tool", "name": "get_oven_state",
             "content": '{"result": {"door_main_state": "open", "door_top_state": "closed"}}'}]
    draft = {"tool_calls": [{"function": {"name": "set_oven_power", "arguments": {"on": True}}}]}
    findings = O.check_obligations(draft, msgs, tools, rules)
    assert len(findings) == 1 and '"door": "MAIN"' in findings[0], findings
    assert "TOP" not in findings[0]


def test_ledger_stage_is_reply_only_candidate():
    cfg = HarnessConfig()
    draft = {"content": "All done, enjoy!"}
    sink = {}
    V.run_verification(draft, [{"role": "user", "content": "hi"}], WINDOW_TOOLS, [],
                       ProvenanceLedger(), cfg, stage_sink=sink, skip_llm=True,
                       ledger=["turn on the AC", "close the windows"])
    # candidate tier: present in the sink, but skip_llm returns hard-only
    assert sink.get("ledger") and "close the windows" in sink["ledger"][0]
    sink2 = {}
    V.run_verification(AC_DRAFT, [{"role": "user", "content": "hi"}], WINDOW_TOOLS, [],
                       ProvenanceLedger(), cfg, stage_sink=sink2, skip_llm=True,
                       ledger=["turn on the AC"])
    assert not sink2.get("ledger")  # tool-call turns are not report turns


def test_vote_key_canonicalization():
    from harness.orchestrator import CoVeOrchestrator
    o = CoVeOrchestrator(HarnessConfig())
    calls = {"n": 0}
    drafts = [
        {"tool_calls": [{"function": {"name": "a", "arguments": {"x": 1}}}]},
        {"tool_calls": [{"function": {"name": "b", "arguments": {}}}]},
        {"tool_calls": [{"function": {"name": "a", "arguments": '{"x": 1}'}}]},  # same as #1
    ]
    def fake_student(messages, tools, record):
        d = drafts[calls["n"] % 3]
        calls["n"] += 1
        return d
    o._student = fake_student
    o.cfg.vote_n = 3
    winner, passes = o._student_vote([], [], None)
    assert passes == 3
    assert (winner.get("tool_calls")[0]["function"]["name"]) == "a"  # modal set wins


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL  {name}: {e}")
    print("ALL TESTS PASS" if not failed else f"{failed} FAILURE(S)")
    sys.exit(1 if failed else 0)
