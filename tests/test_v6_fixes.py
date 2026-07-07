"""Tests for the v6 disambiguation-split fixes.

Fix 1 — future-condition gate: future/arrival-time constraints must not be
         answered with current-moment filters or current-only refusals.
Fix 2 — refusal-read gate: a can't-do reply made without ever reading the
         subsystem state must first perform the (score-free) read.
Fix 3 — obligation engine v6: field-level staleness with write simulation,
         post-draft-state evaluation, covered() argument comparison, and
         canonical ALL-aggregation.

All deterministic, no LLM calls. Alien-domain variants prove nothing is tied
to the CAR-bench tool vocabulary.
Run: .venv/bin/python tests/test_v6_fixes.py  (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "track_1_agent_under_test"))

from harness import obligations as O
from harness import verify as V

# --------------------------------------------------------------------------- #
# Fix 1 — future-condition gate
# --------------------------------------------------------------------------- #
POI_TOOLS = [
    {"function": {"name": "search_poi_along_the_route",
                  "parameters": {"type": "object", "properties": {
                      "route_id": {"type": "string"},
                      "filters": {"type": "array"},
                      "at_kilometer": {"type": "number"}}}}},
]


def _future_msgs(user_text):
    return [{"role": "system", "content": "p"},
            {"role": "user", "content": user_text}]


def test_future_gate_fires_on_currently_filter_with_time_window():
    draft = {"tool_calls": [{"function": {
        "name": "search_poi_along_the_route",
        "arguments": {"route_id": "r1", "filters": ["any::currently_open"],
                      "at_kilometer": 100}}}]}
    msgs = _future_msgs("I need a fast food place open between 7 PM and 7:45 PM.")
    fs = V.check_future_condition(draft, msgs, POI_TOOLS)
    assert len(fs) == 1 and "FUTURE" in fs[0], fs


def test_future_gate_fires_on_current_only_refusal():
    draft = {"content": "Sorry, I can only search for places that are "
                        "currently open, I cannot filter a future time range."}
    msgs = _future_msgs("Find a supermarket that is still open when I arrive.")
    fs = V.check_future_condition(draft, msgs, POI_TOOLS)
    assert len(fs) == 1, fs


def test_future_gate_silent_without_future_expression():
    """dis_26-style ask: the user EXPLICITLY wants 'currently available' —
    the same filter must not be flagged."""
    draft = {"tool_calls": [{"function": {
        "name": "search_poi_along_the_route",
        "arguments": {"filters": ["any::currently_open"]}}}]}
    msgs = _future_msgs("How long would I need to charge at a currently available charger?")
    assert V.check_future_condition(draft, msgs, POI_TOOLS) == []


def test_future_gate_silent_on_plain_refusal_without_present_anchor():
    """A genuine capability refusal with no 'currently' justification is the
    capability gate's territory, not this one's."""
    draft = {"content": "I'm sorry, I don't have a tool for restaurant bookings."}
    msgs = _future_msgs("Book a table for 8 pm tonight.")
    assert V.check_future_condition(draft, msgs, POI_TOOLS) == []


def test_future_time_arg_candidate_on_timed_read():
    """dis_53: weather checked at the CURRENT hour for a condition that
    applies at arrival -> teacher candidate naming the timed arguments."""
    draft = {"tool_calls": [{"function": {
        "name": "get_weather",
        "arguments": {"location_or_poi_id": "loc_1", "time_hour_24hformat": 16,
                      "day": 13, "month": 3}}}]}
    msgs = _future_msgs("Navigate there if it does not rain when I arrive.")
    fs = V.check_future_time_arg(draft, msgs)
    assert len(fs) == 1 and "get_weather" in fs[0], fs
    # and silent when the user constraint has no future expression
    assert V.check_future_time_arg(draft, _future_msgs("What's the weather like?")) == []


# --------------------------------------------------------------------------- #
# Fix 2 — refusal-read gate (alien domain: nothing CAR-specific)
# --------------------------------------------------------------------------- #
OVEN_TOOLS = [
    {"function": {"name": "get_oven_rack_positions",
                  "parameters": {"type": "object", "properties": {}}}},
    {"function": {"name": "set_oven_rack",
                  "parameters": {"type": "object", "properties": {
                      "rack": {"type": "string"}, "height": {"type": "number"}},
                      "required": ["rack", "height"]}}},
]


def test_refusal_read_demands_the_read_first():
    """dis_16 twin: 'sync/equalize' refused with ZERO tool calls while a
    zero-parameter read and a write for the subject exist."""
    draft = {"content": "I'm sorry, I can't equalize the oven rack positions for you."}
    msgs = [{"role": "user", "content": "Can you equalize the oven rack positions?"}]
    fs = V.check_refusal_read(draft, msgs, OVEN_TOOLS)
    assert len(fs) == 1 and "get_oven_rack_positions" in fs[0], fs


def test_refusal_read_silent_once_state_was_read():
    draft = {"content": "I'm sorry, I can't equalize the oven rack positions."}
    msgs = [
        {"role": "user", "content": "Can you equalize the oven rack positions?"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_oven_rack_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_oven_rack_positions", "content": "{}"},
    ]
    assert V.check_refusal_read(draft, msgs, OVEN_TOOLS) == []


def test_refusal_read_silent_without_matching_write():
    """Hallucination-split safety: refusing a subject with NO write tool
    (a truly absent capability) must never be challenged with a read demand."""
    tools = [OVEN_TOOLS[0]]  # read exists, no write
    draft = {"content": "I'm sorry, I can't equalize the oven rack positions."}
    msgs = [{"role": "user", "content": "Can you equalize the oven rack positions?"}]
    assert V.check_refusal_read(draft, msgs, tools) == []


def test_refusal_read_silent_when_read_needs_parameters():
    """Only zero-required-parameter reads are demanded — the gate must never
    push the student into guessing arguments."""
    tools = [
        {"function": {"name": "get_oven_rack_positions",
                      "parameters": {"type": "object",
                                     "properties": {"oven_id": {"type": "string"}},
                                     "required": ["oven_id"]}}},
        OVEN_TOOLS[1],
    ]
    draft = {"content": "I'm sorry, I can't equalize the oven rack positions."}
    msgs = [{"role": "user", "content": "Can you equalize the oven rack positions?"}]
    assert V.check_refusal_read(draft, msgs, tools) == []


def test_refusal_read_silent_on_non_refusal_reply():
    draft = {"content": "The oven racks are now at the same height."}
    msgs = [{"role": "user", "content": "Can you equalize the oven rack positions?"}]
    assert V.check_refusal_read(draft, msgs, OVEN_TOOLS) == []


# --------------------------------------------------------------------------- #
# Fix 3 — obligation engine v6
# --------------------------------------------------------------------------- #
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
    "id": "011_a", "type": "auto_action", "trigger_tools": ["set_air_conditioning"],
    "requirement": "When turning the AC on, close all windows open more than 20%.",
    "exec": {"read": "get_vehicle_window_positions", "field_pattern": "window_*_position",
             "op": ">", "value": 20,
             "obligation": {"tool": "open_close_window",
                            "args": {"window": "<item>", "percentage": 0}}},
}]

CLIMATE_TOOLS = [
    {"function": {"name": "get_climate_settings",
                  "parameters": {"type": "object", "properties": {}}}},
    {"function": {"name": "set_air_conditioning",
                  "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}}}},
    {"function": {"name": "set_climate_temperature",
                  "parameters": {"type": "object", "properties": {
                      "temperature": {"type": "number"}, "seat_zone": {"type": "string"}}}}},
    {"function": {"name": "set_fan_speed",
                  "parameters": {"type": "object", "properties": {"level": {"type": "number"}}}}},
]

FAN_RULE = [{
    "id": "011_b", "type": "auto_action", "trigger_tools": ["set_air_conditioning"],
    "requirement": "When turning the AC on, set fan speed to 1 if it is 0.",
    "exec": {"read": "get_climate_settings", "field_pattern": "fan_speed",
             "op": "==", "value": 0,
             "obligation": {"tool": "set_fan_speed", "args": {"level": 1}}},
}]


def test_draft_created_condition_yields_obligation():
    """dis_32: the AC is already on (earlier write) and the draft OPENS all
    windows to 50%%. The post-write state violates the rule -> the engine
    demands the remediation, aggregated to the canonical single ALL call."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Open all the windows a bit and turn on the AC."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vehicle_window_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"result": {"window_driver_position": 0, "window_passenger_position": 0, '
                    '"window_driver_rear_position": 0, "window_passenger_rear_position": 0}}'},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}}]},
        {"role": "tool", "name": "set_air_conditioning", "content": '{"status": "SUCCESS"}'},
    ]
    draft = {"tool_calls": [{"function": {"name": "open_close_window",
                                          "arguments": {"window": "ALL", "percentage": 50}}}]}
    findings = O.check_obligations(draft, msgs, WINDOW_TOOLS, WINDOW_RULE)
    assert len(findings) == 1, findings
    assert '"window": "ALL"' in findings[0] and '"percentage": 0' in findings[0], findings


def test_open_all_50_does_not_cover_close_to_zero():
    """covered() bug: an ALL-argument call used to cover ANY obligation of the
    same tool regardless of the other arguments."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "AC on please."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vehicle_window_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"result": {"window_driver_position": 60, "window_passenger_position": 60, '
                    '"window_driver_rear_position": 60, "window_passenger_rear_position": 60}}'},
    ]
    draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "open_close_window",
                      "arguments": {"window": "ALL", "percentage": 50}}}]}
    findings = O.check_obligations(draft, msgs, WINDOW_TOOLS, WINDOW_RULE)
    assert len(findings) == 1 and '"percentage": 0' in findings[0], findings


def test_field_level_staleness_keeps_unrelated_fields_alive():
    """dis_34 root cause: set_climate_temperature after the climate read used
    to invalidate the WHOLE read, silencing the fan rule. Field-level
    staleness only drops the temperature fields — fan_speed stays valid and
    the obligation fires."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Warm up the car."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_climate_settings", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_climate_settings",
         "content": '{"result": {"fan_speed": 0, "climate_temperature_driver": 18}}'},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "set_climate_temperature",
                          "arguments": {"temperature": 22, "seat_zone": "ALL_ZONES"}}}]},
        {"role": "tool", "name": "set_climate_temperature", "content": '{"status": "SUCCESS"}'},
    ]
    draft = {"tool_calls": [{"function": {"name": "set_air_conditioning",
                                          "arguments": {"on": True}}}]}
    findings = O.check_obligations(draft, msgs, CLIMATE_TOOLS, FAN_RULE)
    assert len(findings) == 1 and "set_fan_speed" in findings[0], findings
    assert '"level": 1' in findings[0], findings


def test_draft_that_satisfies_obligation_is_clean():
    """The same dis_34 draft WITH the fan call: post-draft simulation sets
    fan_speed := 1 -> condition false -> nothing demanded."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Warm up the car."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_climate_settings", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_climate_settings",
         "content": '{"result": {"fan_speed": 0}}'},
    ]
    draft = {"tool_calls": [
        {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
        {"function": {"name": "set_fan_speed", "arguments": {"level": 1}}}]}
    assert O.check_obligations(draft, msgs, CLIMATE_TOOLS, FAN_RULE) == []


def test_failed_write_invalidates_instead_of_simulating():
    """A write whose tool result is an ERROR changed nothing — simulating it
    would hide a still-standing obligation; the field must go unknown and a
    fresh read be demanded."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "AC on."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vehicle_window_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"result": {"window_driver_position": 60, "window_passenger_position": 60, '
                    '"window_driver_rear_position": 60, "window_passenger_rear_position": 60}}'},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "open_close_window",
                          "arguments": {"window": "ALL", "percentage": 0}}}]},
        {"role": "tool", "name": "open_close_window", "content": '{"status": "ERROR"}'},
    ]
    draft = {"tool_calls": [{"function": {"name": "set_air_conditioning",
                                          "arguments": {"on": True}}}]}
    findings = O.check_obligations(draft, msgs, WINDOW_TOOLS, WINDOW_RULE)
    assert len(findings) == 1 and "get_vehicle_window_positions" in findings[0], findings


def test_trigger_from_earlier_turn_still_binds():
    """The AC write happened in a PREVIOUS turn; this turn's window write
    re-creates the condition — the rule must still trigger."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Turn on the AC."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}}]},
        {"role": "tool", "name": "set_air_conditioning", "content": '{"status": "SUCCESS"}'},
        {"role": "user", "content": "Now open all the windows halfway."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vehicle_window_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"result": {"window_driver_position": 0, "window_passenger_position": 0, '
                    '"window_driver_rear_position": 0, "window_passenger_rear_position": 0}}'},
    ]
    draft = {"tool_calls": [{"function": {"name": "open_close_window",
                                          "arguments": {"window": "ALL", "percentage": 50}}}]}
    findings = O.check_obligations(draft, msgs, WINDOW_TOOLS, WINDOW_RULE)
    assert len(findings) == 1 and '"percentage": 0' in findings[0], findings


def test_read_only_draft_never_checked():
    """Obligations are enforced at write time — a gather draft must pass
    freely even when a trigger already executed."""
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "AC on."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}}]},
        {"role": "tool", "name": "set_air_conditioning", "content": '{"status": "SUCCESS"}'},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vehicle_window_positions", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"result": {"window_driver_position": 60}}'},
    ]
    draft = {"tool_calls": [{"function": {"name": "get_climate_settings",
                                          "arguments": "{}"}}]}
    assert O.check_obligations(draft, msgs, WINDOW_TOOLS + CLIMATE_TOOLS[:1], WINDOW_RULE) == []


def test_alien_domain_simulation():
    """Nothing window-specific: a greenhouse vent rule simulates and
    aggregates identically."""
    tools = [
        {"function": {"name": "get_vent_openings",
                      "parameters": {"type": "object", "properties": {}}}},
        {"function": {"name": "start_heating",
                      "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}}}},
        {"function": {"name": "set_vent",
                      "parameters": {"type": "object", "properties": {
                          "vent": {"type": "string", "enum": ["NORTH", "SOUTH", "ALL"]},
                          "opening": {"type": "number"}}}}},
    ]
    rules = [{
        "id": "g1", "type": "auto_action", "trigger_tools": ["start_heating"],
        "requirement": "When heating starts, close vents open more than 10.",
        "exec": {"read": "get_vent_openings", "field_pattern": "vent_*_opening",
                 "op": ">", "value": 10,
                 "obligation": {"tool": "set_vent",
                                "args": {"vent": "<item>", "opening": 0}}},
    }]
    msgs = [
        {"role": "system", "content": "p"},
        {"role": "user", "content": "Start heating."},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "get_vent_openings", "arguments": "{}"}}]},
        {"role": "tool", "name": "get_vent_openings",
         "content": '{"result": {"vent_north_opening": 40, "vent_south_opening": 40}}'},
    ]
    draft = {"tool_calls": [{"function": {"name": "start_heating",
                                          "arguments": {"on": True}}}]}
    findings = O.check_obligations(draft, msgs, tools, rules)
    assert len(findings) == 1 and '"vent": "ALL"' in findings[0], findings
    # and a draft containing the aggregate remediation is clean
    draft2 = {"tool_calls": [
        {"function": {"name": "start_heating", "arguments": {"on": True}}},
        {"function": {"name": "set_vent", "arguments": {"vent": "ALL", "opening": 0}}}]}
    assert O.check_obligations(draft2, msgs, tools, rules) == []


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
