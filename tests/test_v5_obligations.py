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


def test_parenthesized_names_from_live_compiler_are_normalized():
    """Live flash output included 'get_vehicle_window_positions()' — names with
    parentheses (copied from the signature format) must still evaluate."""
    rules = [dict(WINDOW_RULE[0], exec=dict(WINDOW_RULE[0]["exec"],
                                            read="get_vehicle_window_positions()"))]
    findings = O.check_obligations(AC_DRAFT, _msgs(STATE), WINDOW_TOOLS, rules)
    assert len(findings) == 1 and '"window": "DRIVER"' in findings[0], findings



def test_stale_read_demands_refresh_not_redundant_obligations():
    """Live base_32 failure: ALL windows were closed AFTER the positions read;
    the engine then demanded redundant per-item closes from the stale values.
    v6: the close is the rule's own obligation tool, so the engine SIMULATES
    it (all positions := 0) instead of invalidating the read — no redundant
    obligations, and no re-read roundtrip either."""
    msgs = _msgs(STATE) + [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "open_close_window",
                          "arguments": {"window": "ALL", "percentage": 0}}}]},
        {"role": "tool", "name": "open_close_window",
         "content": '{"result": {"window": "ALL", "percentage": 0}}'},
    ]
    findings = O.check_obligations(AC_DRAFT, msgs, WINDOW_TOOLS, WINDOW_RULE)
    assert findings == [], findings


def test_executed_obligations_count_as_covered():
    """Live base_40 failure: the missing-finding made the student re-send
    already-executed writes. Obligations satisfied earlier in the task are
    covered."""
    msgs = _msgs(STATE) + [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "open_close_window",
                          "arguments": {"window": "DRIVER", "percentage": 0}}},
            {"function": {"name": "open_close_window",
                          "arguments": {"window": "PASSENGER", "percentage": 0}}}]},
    ]
    findings = O.check_obligations(AC_DRAFT, msgs, WINDOW_TOOLS, WINDOW_RULE)
    # both computed obligations were already executed -> nothing missing...
    # but the closes themselves make the positions read stale -> re-read demand
    assert all("DRIVER" not in f or "get_vehicle" in f for f in findings), findings



def test_inverse_scope_flags_writes_on_non_qualifying_items():
    """Live base_94 failure: closed the 10%-open window under a >20% rule.
    A write on a matched-but-non-qualifying item is flagged — unless the user
    explicitly mentioned the subject."""
    draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "DRIVER", "percentage": 0}}},      # qualifying (25)
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "DRIVER_REAR", "percentage": 0}}}, # NON-qualifying (0)
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "PASSENGER", "percentage": 0}}},   # qualifying (100)
    ]}
    findings = O.check_obligations(draft, _msgs(STATE), WINDOW_TOOLS, WINDOW_RULE)
    extra = [f for f in findings if "NOT required" in f]
    assert len(extra) == 1 and "DRIVER_REAR" in extra[0], findings
    # explicit user mention of the subject disables the inverse check
    msgs2 = _msgs(STATE)
    msgs2[1] = {"role": "user", "content": "Turn on the AC and close all the windows please."}
    findings2 = O.check_obligations(draft, msgs2, WINDOW_TOOLS, WINDOW_RULE)
    assert not any("NOT required" in f for f in findings2), findings2


AIRFLOW_TOOLS = WINDOW_TOOLS + [
    {"function": {"name": "get_climate_settings",
                  "parameters": {"type": "object", "properties": {}}}},
    {"function": {"name": "set_fan_airflow_direction",
                  "parameters": {"type": "object", "properties": {
                      "direction": {"type": "string"}}}}},
]
AIRFLOW_RULE = [{
    "id": "010_air", "type": "auto_action",
    "trigger_tools": ["set_window_defrost", "set_fan_airflow_direction", "set_air_conditioning"],
    "requirement": "Set airflow to WINDSHIELD if the current direction does not include WINDSHIELD.",
    "exec": {"read": "get_climate_settings", "field_pattern": "fan_airflow_direction",
             "op": "not_contains", "value": "WINDSHIELD",
             "obligation": {"tool": "set_fan_airflow_direction",
                            "args": {"direction": "WINDSHIELD"}}},
}]


def _climate_msgs(direction):
    return [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Turn on the AC."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_climate_settings", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_climate_settings",
         "content": '{"result": {"fan_airflow_direction": "%s"}}' % direction},
    ]


def test_not_contains_condition_and_unneeded_call():
    """Live base_32 failure: airflow already included WINDSHIELD but the agent
    changed it anyway. With not_contains the engine knows the condition is NOT
    met and flags the unnecessary call; when the condition IS met, the
    obligation is demanded."""
    extra_draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "set_fan_airflow_direction",
                      "arguments": {"direction": "WINDSHIELD"}}}]}
    f1 = O.check_obligations(extra_draft, _climate_msgs("WINDSHIELD_HEAD_FEET"),
                             AIRFLOW_TOOLS, AIRFLOW_RULE)
    assert any("NOT required" in x for x in f1), f1
    plain_draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}}]}
    f2 = O.check_obligations(plain_draft, _climate_msgs("FEET"),
                             AIRFLOW_TOOLS, AIRFLOW_RULE)
    assert any("set_fan_airflow_direction" in x and "REQUIRES" in x for x in f2), f2
    # condition met and the agent (correctly) did not call it -> silent
    f3 = O.check_obligations(plain_draft, _climate_msgs("WINDSHIELD_HEAD_FEET"),
                             AIRFLOW_TOOLS, AIRFLOW_RULE)
    assert not any("set_fan_airflow_direction" in x for x in f3), f3



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
