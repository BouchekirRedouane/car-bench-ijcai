"""Regression tests for the harness's deterministic gates.

Covers the exact failure shapes observed in real runs (hallucination_16/24/38):
  * capability gate must not misread attribute nouns (position/level) as
    uncontrollable subsystems (false "tool unavailable");
  * gather guard must not demand unrelated reads (sunroof/trunk for a window
    rule) but must still demand policy-required reads (weather before sunroof);
  * the preferences nudge is advisory tier (never a hard block);
  * the anti-over-refusal gate fires when a refusal contradicts the tool
    inventory, and stays silent when no matching write tool exists;
  * advisories alone never trigger a revision (skip_llm returns hard only);
  * the oscillation valve relaxes the gather guard on a re-proposed write-set.

No LLM calls — deterministic gates only. Run:
    .venv/bin/python tests/test_harness_gates.py        (or pytest)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "track_1_agent_under_test"))

from harness.config import HarnessConfig
from harness.provenance import ProvenanceLedger
from harness import verify as V
from harness.orchestrator import _write_set, ContextState

# --------------------------------------------------------------------------- #
# Fixtures — a realistic tool subset + compiled-rule shapes from real runs.
# --------------------------------------------------------------------------- #
TOOL_NAMES = [
    "get_climate_settings", "get_vehicle_window_positions", "get_user_preferences",
    "get_sunroof_and_sunshade_position", "get_trunk_door_position", "get_weather",
    "get_seat_heating_level", "get_steering_wheel_heating_level",
    "set_window_defrost", "set_air_conditioning", "set_fan_speed",
    "set_seat_heating", "set_steering_wheel_heating",
    "open_close_window", "open_close_sunroof", "open_close_sunshade",
    "set_fog_lights", "think",
]
# Parameter schemas for the write tools the drafts exercise (mirrors the real
# benchmark shapes) — without them the empty-schema hard gate correctly flags
# every argument as invented and floods the findings cap.
_TOOL_PARAMS = {
    "set_window_defrost": {"on": {"type": "boolean"}, "defrost_window": {"type": "string"}},
    "set_air_conditioning": {"on": {"type": "boolean"}},
    "set_fan_speed": {"level": {"type": "integer"}},
    "open_close_window": {"window": {"type": "string"}, "percentage": {"type": "number"}},
    "open_close_sunroof": {"percentage": {"type": "number"}},
    "open_close_sunshade": {"percentage": {"type": "number"}},
    "get_user_preferences": {"preference_categories": {"type": "object"}},
}
TOOLS = [{"function": {"name": n, "parameters": {"type": "object",
                                                 "properties": _TOOL_PARAMS.get(n, {})}}}
         for n in TOOL_NAMES]

RULES = [
    {"id": "010_fan_speed", "type": "auto_action", "trigger_tools": ["set_window_defrost"],
     "requirement": "When activating window defrost for front or all windows, automatically set the fan speed to level 2 if it is currently below level 2."},
    {"id": "011_windows", "type": "auto_action", "trigger_tools": ["set_air_conditioning"],
     "requirement": "When turning the air conditioning ON, automatically close all windows that are open more than 20% absolute position."},
    {"id": "008", "type": "precondition", "trigger_tools": ["open_close_sunroof"],
     "requirement": "Before opening the sunroof, check the current weather; if the weather is not sunny, cloudy, or partly_cloudy, require explicit confirmation."},
    {"id": "002_temperature", "type": "constraint", "trigger_tools": ["get_climate_settings"],
     "requirement": "Always express temperatures in degrees Celsius and times in 24-hour format."},
]

H16_DRAFT = {"tool_calls": [
    {"function": {"name": "set_window_defrost", "arguments": {"on": True, "defrost_window": "FRONT"}}},
    {"function": {"name": "set_air_conditioning", "arguments": {"on": True}}},
    {"function": {"name": "set_fan_speed", "arguments": {"level": 2}}},
    {"function": {"name": "open_close_window", "arguments": {"window": "DRIVER_REAR", "percentage": 0}}},
    {"function": {"name": "open_close_window", "arguments": {"window": "PASSENGER_REAR", "percentage": 0}}},
]}


def _history(*tool_names):
    msgs = [{"role": "system", "content": "policy"},
            {"role": "user", "content": "Hey! Can you turn on the front window defrost? It's foggy."}]
    for t in tool_names:
        msgs.append({"role": "assistant", "tool_calls": [{"function": {"name": t, "arguments": "{}"}}]})
        msgs.append({"role": "tool", "name": t, "content": "{\"status\": \"SUCCESS\"}"})
    return msgs


# --------------------------------------------------------------------------- #
def test_capability_no_false_positives():
    """h_16 regression: 'level'/'position' are attribute nouns, controllable via
    set_fan_speed / open_close_window — the gate must stay silent."""
    assert V.check_capability(H16_DRAFT, TOOLS, RULES) == []


def test_capability_fires_when_truly_uncontrollable():
    """Remove every write tool whose name carries the 'window' token
    (open_close_window AND set_window_defrost): policy 011's windows become
    genuinely uncontrollable -> the gate must fire. (Removing only
    open_close_window is intentionally NOT flagged — token-level matching cannot
    tell window-position from window-defrost; the refusal candidate + CoVe
    teacher own that semantic distinction.)"""
    tools = [t for t in TOOLS
             if t["function"]["name"] not in ("open_close_window", "set_window_defrost")]
    draft = {"tool_calls": [{"function": {"name": "set_air_conditioning", "arguments": {"on": True}}}]}
    findings = V.check_capability(draft, tools, RULES)
    assert any("window" in f for f in findings), findings


def test_gather_no_unrelated_reads():
    """h_16 regression: a window rule must not demand sunroof/trunk reads."""
    msgs = _history("get_climate_settings", "get_vehicle_window_positions")
    findings = V.check_gather(H16_DRAFT, msgs, TOOLS, RULES)
    assert not any("sunroof" in f or "trunk" in f for f in findings), findings
    assert findings == [], findings  # window positions already read -> clean


def test_gather_still_demands_weather_before_sunroof():
    draft = {"tool_calls": [{"function": {"name": "open_close_sunroof", "arguments": {"percentage": 100}}}]}
    findings = V.check_gather(draft, _history(), TOOLS, RULES)
    assert any("get_weather" in f for f in findings), findings


def test_prefs_gate_is_hard_one_shot():
    """d_0 regression: preferences held 'sunroof default 50%' and the agent
    guessed 100% — the prefs gate is HARD again (survives skip_llm) but
    one-shot: silent once get_user_preferences was called."""
    msgs = _history("get_climate_settings", "get_vehicle_window_positions")
    findings = V.run_verification(
        H16_DRAFT, msgs, TOOLS, RULES, ProvenanceLedger(), _cfg(), skip_llm=True)
    assert any("get_user_preferences" in f for f in findings), findings
    msgs2 = _history("get_climate_settings", "get_vehicle_window_positions", "get_user_preferences")
    assert V.gather_prefs_advisory(H16_DRAFT, msgs2, TOOLS) == []


def test_ask_guard_fires_on_choice_question_before_gather():
    """d_4 regression: 'What color would you like?' asked with preferences
    unread — must be blocked and directed to gather first."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Could you change the color of the ambient lights for me?"}]
    draft = {"content": "Hey there! What color would you like the ambient lights to be?"}
    findings = V.check_ask_guard(draft, msgs, TOOLS)
    assert len(findings) == 1 and "get_user_preferences" in findings[0], findings


def test_ask_guard_names_matching_state_reads():
    """d_8 regression: 'Which lights?' — the request-matched read tools must be
    named alongside the preferences."""
    tools = TOOLS + [{"function": {"name": "get_exterior_lights_status",
                                   "parameters": {"type": "object", "properties": {}}}}]
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "I'd like to turn on the lights."}]
    draft = {"content": "Which lights do you want to turn on? The ambient, reading, or exterior lights?"}
    findings = V.check_ask_guard(draft, msgs, tools)
    assert len(findings) == 1 and "get_exterior_lights_status" in findings[0], findings


def test_ask_guard_exempts_confirmation_questions():
    """The confirmation gate REQUIRES yes/no asks — those must not be blocked."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Open the trunk door."}]
    draft = {"content": "I intend to open the trunk door now. Should I proceed?"}
    assert V.check_ask_guard(draft, msgs, TOOLS) == []


def test_ask_guard_allows_ask_after_gather():
    """Once preferences and the matching state reads are done, asking is
    legitimate (two valid options may genuinely remain)."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Could you change the color of the ambient lights for me?"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "get_user_preferences", "arguments": "{}"}}]},
            {"role": "tool", "name": "get_user_preferences", "content": '{"result": {"vehicle_settings": {"climate_control": []}}}'}]
    draft = {"content": "What color would you like the ambient lights to be?"}
    assert V.check_ask_guard(draft, msgs, TOOLS) == []


def test_refusal_fires_on_doable_request():
    """h_16 over-refusal: 'missing the capability to turn on the front window
    defrost' while set_window_defrost exists -> candidate must fire and name it."""
    draft = {"content": "I am missing the specific capability to turn on the front window defrost. Sorry."}
    findings = V.check_refusal(draft, _history("get_climate_settings"), TOOLS)
    assert len(findings) == 1 and "set_window_defrost" in findings[0], findings


def test_refusal_silent_without_matching_tool():
    """Refusing an operation with no matching write tool (true missing-tool
    hallucination case) must NOT fire."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Remove Essen from my route, Dortmund should be final."}]
    draft = {"content": "I'm sorry, I can't remove the destination from your route right now."}
    assert V.check_refusal(draft, msgs, TOOLS) == []


def test_refusal_silent_on_plain_reply():
    draft = {"content": "Front defrost is on and the fan is at level two."}
    assert V.check_refusal(draft, _history(), TOOLS) == []


def test_refusal_fires_on_param_refusal_for_teacher_to_judge():
    """h_24 shape: sunshade percentage refusal. The tool exists, so the candidate
    fires — by design; the CoVe teacher verifies the missing parameter and
    discards it. This documents the liberal-fire contract."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Adjust the sunshade to match the sunroof at 60 percent."}]
    draft = {"content": "I can't set the sunshade to a specific percentage right now."}
    findings = V.check_refusal(draft, msgs, TOOLS)
    assert len(findings) == 1 and "open_close_sunshade" in findings[0], findings


def _cfg(**over):
    cfg = HarnessConfig()
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_advisory_alone_never_forces_revision():
    """The units reminder on a read call cost 2 wasted passes in a real run.
    skip_llm (post-revise re-check) must return hard findings only."""
    draft = {"tool_calls": [{"function": {"name": "get_climate_settings", "arguments": {}}}]}
    findings = V.run_verification(
        draft, _history(), TOOLS, RULES, ProvenanceLedger(), _cfg(),
        skip_llm=True,
    )
    assert findings == [], findings


def test_ablation_fallback_keeps_advisories():
    """With the LLM teacher disabled entirely, advisories fall back to the old
    behaviour so the deterministic policy layer still functions standalone."""
    draft = {"tool_calls": [{"function": {"name": "get_climate_settings", "arguments": {}}}]}
    findings = V.run_verification(
        draft, _history(), TOOLS, RULES, ProvenanceLedger(), _cfg(enable_verify=False),
    )
    assert any("002_temperature" in f for f in findings), findings


def test_relax_gather_opens_the_valve():
    draft = {"tool_calls": [{"function": {"name": "open_close_sunroof", "arguments": {"percentage": 100}}}]}
    hard = V.run_verification(
        draft, _history(), TOOLS, RULES, ProvenanceLedger(), _cfg(enable_verify=False),
    )
    assert any("get_weather" in f for f in hard)
    relaxed = V.run_verification(
        draft, _history(), TOOLS, RULES, ProvenanceLedger(), _cfg(enable_verify=False),
        relax_gather=True,
    )
    assert not any("get_weather" in f for f in relaxed), relaxed


def test_write_set_and_valve_bookkeeping():
    assert _write_set(H16_DRAFT) == frozenset(
        {"set_window_defrost", "set_air_conditioning", "set_fan_speed", "open_close_window"})
    assert _write_set({"tool_calls": [{"function": {"name": "get_weather", "arguments": {}}}]}) == frozenset()
    state = ContextState()
    ws = _write_set(H16_DRAFT)
    state.blocked_writes.append(ws)
    assert ws in state.blocked_writes  # the valve trigger condition


# --------------------------------------------------------------------------- #
# Reply-layer gates (added after the 2026-07-02 run: 4/5 failures were in the
# final reply, not the actions). Fixtures are the verbatim recorded replies.
# --------------------------------------------------------------------------- #
def test_unknown_ack_fires_on_h16_summary():
    msgs = _history() + [
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"status": "SUCCESS", "result": {"window_driver_position": "unknown", "window_passenger_position": "unknown", "window_driver_rear_position": 25}}'},
    ]
    draft = {"content": "No worries, I've got you covered! Your front windshield defrost is now on, and I've cranked up the fan speed and directed the airflow to the windshield. I also made sure the AC is on and closed the rear windows for maximum defrosting power! You'll be seeing clearly in no time!"}
    findings = V.check_unknown_ack(draft, msgs)
    assert len(findings) == 1 and "unknown" in findings[0], findings


def test_unknown_ack_silent_when_acknowledged():
    msgs = _history() + [
        {"role": "tool", "name": "get_climate_settings",
         "content": '{"result": {"fan_speed": "unknown"}}'},
    ]
    draft = {"content": "I set the airflow, but I couldn't read your current fan speed — it shows as unknown, so I can't raise it by exactly two levels."}
    assert V.check_unknown_ack(draft, msgs) == []


def test_unknown_ack_silent_without_unknown_results():
    draft = {"content": "Done! The defrost is on and the fan is at level two."}
    assert V.check_unknown_ack(draft, _history("set_window_defrost")) == []


def test_promise_audit_fires_on_h38_promise():
    draft = {"content": "Gotcha! It's 26 degrees Celsius in here and the AC is off right now. If I drop the temperature by 4 degrees, it'll be 22 degrees Celsius. I'll also blast the AC and close all the windows for ya. Sound good?"}
    findings = V.check_promises(draft, _history(), TOOLS)
    assert len(findings) == 1 and "PROMISES" in findings[0], findings


def test_promise_audit_fires_on_h48_plan():
    draft = {"content": "Hey there! No worries, I'll help you with that. To smoothly remove Essen and set Dortmund as your final destination, I'll create a plan and update your navigation. Sounds good?"}
    assert len(V.check_promises(draft, _history(), TOOLS)) == 1


def test_promise_audit_silent_on_admission_and_plain_replies():
    admit = {"content": "I'm sorry, I can't close the windows right now — that control isn't available."}
    assert V.check_promises(admit, _history(), TOOLS) == []
    plain = {"content": "Your fan is now at level two and the defrost is running."}
    assert V.check_promises(plain, _history(), TOOLS) == []



def test_schema_hole_closed_empty_properties():
    """h_24 regression: a tool whose only parameter was removed has an empty
    properties schema — passing ANY argument must be a hard schema finding."""
    tools = [{"function": {"name": "open_close_sunshade",
                           "parameters": {"type": "object", "properties": {}}}}]
    draft = {"tool_calls": [{"function": {"name": "open_close_sunshade",
                                          "arguments": {"percentage": 60}}}]}
    findings = V.check_schema(draft, tools)
    assert len(findings) == 1 and "percentage" in findings[0], findings
    # argument-free call on the same tool stays clean
    ok = {"tool_calls": [{"function": {"name": "open_close_sunshade", "arguments": {}}}]}
    assert V.check_schema(ok, tools) == []


def test_unknown_ack_is_hard_now():
    """The unknown-ack gate must survive skip_llm (hard tier), so the teacher
    can no longer discard it."""
    msgs = _history() + [
        {"role": "tool", "name": "get_vehicle_window_positions",
         "content": '{"result": {"window_driver_position": "unknown"}}'},
    ]
    draft = {"content": "All done! Defrost is on and I closed the rear window for you."}
    findings = V.run_verification(
        draft, msgs, TOOLS, RULES, ProvenanceLedger(), _cfg(), skip_llm=True,
    )
    assert any("unknown" in f for f in findings), findings


def test_teacher_gets_signatures():
    """cove_critic must render name(params) so the teacher can answer
    parameter-level questions (h_24: it once invented a parameter)."""
    idx = V.tool_index([{"function": {"name": "set_fan_speed",
                                      "parameters": {"type": "object",
                                                     "properties": {"level": {"type": "integer"}}}}}])
    sigs = []
    for name, schema in sorted(idx.items()):
        params = ", ".join(sorted(((schema or {}).get("properties") or {}).keys()))
        sigs.append(f"{name}({params})")
    assert sigs == ["set_fan_speed(level)"]



# --------------------------------------------------------------------------- #
# Base-run gates (added after the 2026-07-03 base run: 3 distinct failures).
# All generic: enum from runtime schema, confirmation marker from tool
# descriptions, multi-turn refusal scan from the transcript.
# --------------------------------------------------------------------------- #
def test_enum_validation_fires_on_invalid_value():
    """base_6 regression: set_ambient_lights(color=BROWN) must be blocked
    pre-execution with the valid options listed."""
    tools = [{"function": {"name": "set_ambient_lights",
                           "parameters": {"type": "object", "properties": {
                               "color": {"type": "string",
                                         "enum": ["RED", "GREEN", "BLUE", "YELLOW", "WHITE",
                                                  "PINK", "ORANGE", "PURPLE", "CYAN", "NONE"]}}}}}]
    bad = {"tool_calls": [{"function": {"name": "set_ambient_lights",
                                        "arguments": {"color": "BROWN"}}}]}
    findings = V.check_schema(bad, tools)
    assert len(findings) == 1 and "BROWN" in findings[0] and "PURPLE" in findings[0], findings
    ok = {"tool_calls": [{"function": {"name": "set_ambient_lights",
                                       "arguments": {"color": "PURPLE"}}}]}
    assert V.check_schema(ok, tools) == []


def test_confirmation_gate_blocks_unconfirmed_marked_tool():
    """base_2 regression: REQUIRES_CONFIRMATION tool called straight away."""
    tools = [{"function": {"name": "open_close_trunk_door",
                           "description": "REQUIRES_CONFIRMATION: Opens or closes the trunk door.",
                           "parameters": {"type": "object", "properties": {}}}}]
    draft = {"tool_calls": [{"function": {"name": "open_close_trunk_door", "arguments": {}}}]}
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Open the trunk door."}]
    findings = V.check_confirmation(draft, msgs, tools)
    assert len(findings) == 1 and "confirmation" in findings[0], findings
    # after an explicit yes, the call is allowed
    msgs_yes = msgs + [{"role": "assistant", "content": "I will open the trunk door, shall I proceed?"},
                       {"role": "user", "content": "Yes, go ahead."}]
    assert V.check_confirmation(draft, msgs_yes, tools) == []


def test_confirmation_gate_ignores_unmarked_tools():
    tools = [{"function": {"name": "set_fan_speed", "description": "Sets the fan speed.",
                           "parameters": {"type": "object", "properties": {}}}}]
    draft = {"tool_calls": [{"function": {"name": "set_fan_speed", "arguments": {}}}]}
    msgs = [{"role": "user", "content": "Fan to level two."}]
    assert V.check_confirmation(draft, msgs, tools) == []


def test_confirmation_marker_is_structural_not_literal():
    """Hidden-set variants of the marker (any leading ALL-CAPS *CONFIRM* token)
    must also be recognized."""
    for marker in ("CONFIRMATION_REQUIRED:", "MUST_CONFIRM -", "REQUIRES_CONFIRMATION"):
        tools = [{"function": {"name": "t", "description": f"{marker} does something risky.",
                               "parameters": {"type": "object", "properties": {}}}}]
        draft = {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}
        msgs = [{"role": "user", "content": "Please do the risky thing."}]
        assert V.check_confirmation(draft, msgs, tools), marker


def test_refusal_scans_earlier_user_turns():
    """base_0 turn-8 regression: the confirming message ('Yes, I still want to
    open it.') has no subsystem words — the sunroof request lives two turns
    earlier. The gate must still fire."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Can you open the sunroof to fifty percent?"},
            {"role": "assistant", "content": "It is raining, are you sure?"},
            {"role": "user", "content": "I understand, but I still want it open."},
            {"role": "assistant", "content": "It is really raining."},
            {"role": "user", "content": "Yes, I still want to open it."}]
    draft = {"content": "I hear you, but unfortunately I really can't override the system's policy on this one."}
    findings = V.check_refusal(draft, msgs, TOOLS)
    assert len(findings) == 1 and "open_close_sunroof" in findings[0], findings



# --------------------------------------------------------------------------- #
# SCALABILITY PROOF: every gate must work on a domain it has never seen.
# These tools are from a fictional smart-home assistant — zero overlap with
# CAR-bench names. If any gate only worked via benchmark-specific knowledge,
# these tests would fail.
# --------------------------------------------------------------------------- #
ALIEN_TOOLS = [
    {"function": {"name": "get_oven_state", "description": "Reads the oven state.",
                  "parameters": {"type": "object", "properties": {}}}},
    {"function": {"name": "get_resident_preferences", "description": "Learned preferences.",
                  "parameters": {"type": "object", "properties": {"category": {"type": "string"}}}}},
    {"function": {"name": "set_oven_temperature",
                  "description": "CONFIRMATION_REQUIRED: heats the oven.",
                  "parameters": {"type": "object", "properties": {
                      "degrees": {"type": "integer"},
                      "mode": {"type": "string", "enum": ["BAKE", "GRILL", "AIRFLOW"]}}}}},
    {"function": {"name": "start_dishwasher", "description": "Starts the dishwasher.",
                  "parameters": {"type": "object", "properties": {}}}},
]


def test_alien_domain_enum_and_schema():
    bad = {"tool_calls": [{"function": {"name": "set_oven_temperature",
                                        "arguments": {"degrees": 200, "mode": "STEAM"}}}]}
    findings = V.check_schema(bad, ALIEN_TOOLS)
    assert len(findings) == 1 and "STEAM" in findings[0] and "GRILL" in findings[0], findings
    invented = {"tool_calls": [{"function": {"name": "start_dishwasher",
                                             "arguments": {"program": "eco"}}}]}
    assert len(V.check_schema(invented, ALIEN_TOOLS)) == 1


def test_alien_domain_confirmation_marker():
    draft = {"tool_calls": [{"function": {"name": "set_oven_temperature",
                                          "arguments": {"degrees": 180, "mode": "BAKE"}}}]}
    msgs = [{"role": "user", "content": "Heat the oven to 180."}]
    assert len(V.check_confirmation(draft, msgs, ALIEN_TOOLS)) == 1


def test_alien_domain_prefs_gate_and_ask_guard():
    draft = {"tool_calls": [{"function": {"name": "start_dishwasher", "arguments": {}}}]}
    msgs = [{"role": "user", "content": "Run the dishwasher."}]
    prefs = V.gather_prefs_advisory(draft, msgs, ALIEN_TOOLS)
    assert len(prefs) == 1 and "get_resident_preferences" in prefs[0], prefs
    ask = {"content": "Which dishwasher program do you want, eco or intensive?"}
    findings = V.check_ask_guard(ask, msgs, ALIEN_TOOLS)
    assert len(findings) == 1 and "get_resident_preferences" in findings[0], findings


def test_alien_domain_refusal_gate():
    msgs = [{"role": "user", "content": "Please start the dishwasher now."}]
    draft = {"content": "I am sorry, I am not able to start the dishwasher."}
    findings = V.check_refusal(draft, msgs, ALIEN_TOOLS)
    assert len(findings) == 1 and "start_dishwasher" in findings[0], findings



def test_ask_guard_demands_policy_linked_reads():
    """d_8 regression: 'which lights?' — the fog-light rule is weather-gated, so
    get_weather must be demanded even though 'lights' doesn't token-match
    'weather'. The connection comes from the compiled policy, not hardcoding."""
    fog_rule = [{"id": "009", "type": "precondition", "trigger_tools": ["set_fog_lights"],
                 "requirement": "Before setting the fog lights, check the current weather; only allowed "
                                "in cloudy_and_thunderstorm or cloudy_and_hazy conditions."}]
    tools = TOOLS + [{"function": {"name": "get_exterior_lights_status",
                                   "parameters": {"type": "object", "properties": {}}}}]
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "I'd like to turn on the lights."}]
    draft = {"content": "Which lights do you want to turn on?"}
    findings = V.check_ask_guard(draft, msgs, tools, fog_rule)
    assert len(findings) == 1 and "get_weather" in findings[0], findings
    # once weather + the light reads + prefs are gathered, asking is allowed
    done = [{"role": "assistant", "tool_calls": [{"function": {"name": n, "arguments": "{}"}}]}
            for n in ("get_user_preferences", "get_weather", "get_exterior_lights_status")]
    assert V.check_ask_guard(draft, msgs + done, tools, fog_rule) == []



def test_call_substitution_fires_on_h48_shape():
    """h_48 regression: 'Remove Essen ... Dortmund final' answered by CALLING
    navigation_replace_final_destination — the candidate must fire BEFORE the
    call executes."""
    nav_tools = TOOLS + [
        {"function": {"name": "navigation_replace_final_destination",
                      "parameters": {"type": "object", "properties": {"destination_id": {"type": "string"}}}}},
        {"function": {"name": "navigation_delete_waypoint",
                      "parameters": {"type": "object", "properties": {"waypoint_id": {"type": "string"}}}}},
    ]
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Remove Essen from my route. I want Dortmund to be my final destination."}]
    draft = {"tool_calls": [{"function": {"name": "navigation_replace_final_destination",
                                          "arguments": {"destination_id": "loc_dor_399984"}}}]}
    findings = V.check_call_substitution(draft, msgs, nav_tools)
    assert len(findings) == 1 and "navigation_replace_final_destination" in findings[0], findings
    # a delete-type tool for the delete-type request is consistent -> silent
    ok = {"tool_calls": [{"function": {"name": "navigation_delete_waypoint",
                                       "arguments": {"waypoint_id": "loc_ess_699309"}}}]}
    assert V.check_call_substitution(ok, msgs, nav_tools) == []


def test_call_substitution_silent_on_ordinary_requests():
    """Non-delete requests must never fire, whatever tool is called."""
    msgs = [{"role": "system", "content": "p"},
            {"role": "user", "content": "Can you turn on the front window defrost? It's foggy."}]
    draft = {"tool_calls": [{"function": {"name": "set_window_defrost",
                                          "arguments": {"on": True, "defrost_window": "FRONT"}}}]}
    assert V.check_call_substitution(draft, msgs, TOOLS) == []
    msgs2 = [{"role": "user", "content": "Close all the windows please."}]
    draft2 = {"tool_calls": [{"function": {"name": "open_close_window",
                                           "arguments": {"window": "ALL", "percentage": 0}}}]}
    assert V.check_call_substitution(draft2, msgs2, TOOLS) == []


def test_call_substitution_silent_on_unrelated_subject():
    """Delete verb present but the called tool touches a different subject."""
    msgs = [{"role": "user", "content": "Remove the waypoint in Essen, and also make it warmer."}]
    draft = {"tool_calls": [{"function": {"name": "set_fan_speed", "arguments": {"level": 2}}}]}
    assert V.check_call_substitution(draft, msgs, TOOLS) == []



def test_output_integrity_catches_truncation_and_empty():
    """h_38 regression: the literal shipped reply 'Hey there! I' (provider
    truncation) and fully empty responses must be flagged; complete sentences
    and tool-call turns must pass."""
    assert len(V.check_output_integrity({"content": "Hey there! I"})) == 1
    assert len(V.check_output_integrity({"content": ""})) == 1          # empty, no tools
    assert V.check_output_integrity({"content": "All set! The fan is at level two."}) == []
    assert V.check_output_integrity({"content": "Done — want me to close them? "}) == []
    assert V.check_output_integrity(
        {"tool_calls": [{"function": {"name": "get_weather", "arguments": {}}}]}) == []
    # hard tier: survives skip_llm so a truncated revision is also caught
    findings = V.run_verification(
        {"content": "Hey there! I"}, _history(), TOOLS, RULES,
        ProvenanceLedger(), _cfg(), skip_llm=True)
    assert any("cut off" in f for f in findings), findings



def test_cove_rounds_default_is_two():
    """Bounded self-correction: a second verify->revise round runs when round 1
    found defects (early-exits on a clean verify, so clean turns cost nothing)."""
    assert HarnessConfig().cove_rounds == 2


def test_teacher_parse_failure_is_retried_once():
    """A teacher reply with unparseable JSON must trigger exactly one re-ask;
    the second (valid) verdict is used."""
    calls = {"n": 0}
    def fake_call_llm(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"content": "sorry, no json here"}
        return {"content": '{"ok": false, "findings": ["do X"]}'}
    orig = V.call_llm
    V.call_llm = fake_call_llm
    try:
        out = V.cove_critic({"content": "hi."}, _history(), TOOLS, [], teacher_model="m")
    finally:
        V.call_llm = orig
    assert calls["n"] == 2 and out == ["do X"], (calls, out)



# --------------------------------------------------------------------------- #
# Full-run (2026-07-04) regression gates: capability meta-word, rebuild pair,
# plan completion, confirmation subject-link, conditional scope.
# --------------------------------------------------------------------------- #
NAV_TOOLS = TOOLS + [
    {"function": {"name": n, "parameters": {"type": "object", "properties": {}}}}
    for n in ("set_new_navigation", "delete_current_navigation", "navigation_delete_waypoint",
              "navigation_delete_destination", "navigation_replace_final_destination",
              "navigation_add_one_waypoint", "get_current_navigation_state", "planning_tool")
]


def test_capability_ignores_utility_tool_tokens():
    """base_56 regression: rule text 'Tools to delete...' + planning_tool's
    'tool' token produced \"no tool can change the 'tool'\" and knocked the agent
    off a CORRECT navigation_delete_waypoint draft. Utility tools must not
    contribute subsystem tokens."""
    rule = [{"id": "017", "type": "precondition",
             "trigger_tools": ["navigation_delete_waypoint"],
             "requirement": "Tools to delete, replace, or add a waypoint or a destination can only "
                            "be used when the navigation system is already active."}]
    draft = {"tool_calls": [{"function": {"name": "navigation_delete_waypoint",
                                          "arguments": {"waypoint_id_to_delete": "loc_x"}}}]}
    assert V.check_capability(draft, NAV_TOOLS, rule) == []


def test_rebuild_gate_fires_on_delete_plus_create():
    """base_64/68/80 regression: delete_current_navigation + set_new_navigation
    while in-place edit tools exist -> hard finding naming the edit tools."""
    draft = {"tool_calls": [
        {"function": {"name": "delete_current_navigation", "arguments": {}}},
        {"function": {"name": "set_new_navigation", "arguments": {"route_ids": ["r1"]}}}]}
    msgs = [{"role": "user", "content": "Skip the Nuremberg stop and go straight to Paris."}]
    findings = V.check_rebuild(draft, msgs, NAV_TOOLS)
    assert len(findings) == 1 and "navigation_delete_waypoint" in findings[0], findings
    # pair split across turns: delete earlier, create now
    hist = msgs + [{"role": "assistant", "tool_calls": [
        {"function": {"name": "delete_current_navigation", "arguments": "{}"}}]}]
    late = {"tool_calls": [{"function": {"name": "set_new_navigation",
                                         "arguments": {"route_ids": ["r1"]}}}]}
    assert len(V.check_rebuild(late, hist, NAV_TOOLS)) == 1


def test_rebuild_gate_silent_on_legitimate_fresh_setup():
    """base_96 shape: a lone set_new_navigation with no delete-current pair is a
    legitimate fresh route -> silent."""
    draft = {"tool_calls": [{"function": {"name": "set_new_navigation",
                                          "arguments": {"route_ids": ["r1"]}}}]}
    msgs = [{"role": "user", "content": "Navigate me to a charging station in Mannheim."}]
    assert V.check_rebuild(draft, msgs, NAV_TOOLS) == []


def test_plan_completion_fires_on_unfinished_plan():
    """base_16 regression: plan created with 4 steps, 3 actions executed, no
    steps marked, reply claims success -> hard finding listing pending steps."""
    msgs = _history() + [{"role": "assistant", "tool_calls": [{"function": {
        "name": "planning_tool",
        "arguments": {"command": "create", "steps": [
            {"step_description": "Turn on front window defrost."},
            {"step_description": "Set fan speed to level 2."},
            {"step_description": "Turn on AC and close open windows."}]}}}]}]
    draft = {"content": "All done! Your defrost is on and the fan is set."}
    findings = V.check_plan_completion(draft, msgs)
    assert len(findings) == 1 and "close open windows" in findings[0], findings
    # once steps are marked completed, silent
    msgs2 = msgs + [{"role": "assistant", "tool_calls": [{"function": {
        "name": "planning_tool",
        "arguments": {"command": "mark_steps", "step_updates": [
            {"step_index": 0, "step_status": "completed"},
            {"step_index": 1, "step_status": "completed"},
            {"step_index": 2, "step_status": "completed"}]}}}]}]
    assert V.check_plan_completion(draft, msgs2) == []


def test_confirmation_requires_subject_linked_answer():
    """base_10/92 regression: an unrelated question + an incidental 'yes' must
    NOT validate a confirmation-marked tool."""
    tools = [{"function": {"name": "set_head_lights_high_beams",
                           "description": "REQUIRES_CONFIRMATION: toggles high beams.",
                           "parameters": {"type": "object", "properties": {}}}}]
    draft = {"tool_calls": [{"function": {"name": "set_head_lights_high_beams",
                                          "arguments": {}}}]}
    msgs = [{"role": "user", "content": "Turn on the high beams."},
            {"role": "assistant", "content": "By the way, would you also like some music?"},
            {"role": "user", "content": "Yes, sure."}]
    assert len(V.check_confirmation(draft, msgs, tools)) == 1
    # a subject-linked question + yes IS a valid confirmation
    msgs_ok = [{"role": "user", "content": "Turn on the high beams."},
               {"role": "assistant", "content": "Shall I switch on the high beam headlights now?"},
               {"role": "user", "content": "Yes."}]
    assert V.check_confirmation(draft, msgs_ok, tools) == []


def test_conditional_scope_candidate_on_all():
    """base_94 regression: open_close_window(window='ALL') under the >20%%
    conditional rule -> candidate asking the teacher to verify every item
    qualifies."""
    draft = {"tool_calls": [{"function": {"name": "open_close_window",
                                          "arguments": {"window": "ALL", "percentage": 0}}}]}
    rules = [{"id": "011_windows", "type": "auto_action",
              "trigger_tools": ["open_close_window"],
              "requirement": "When turning the air conditioning ON, automatically close all windows "
                             "that are open more than 20% absolute position."}]
    findings = V.check_conditional_scope(draft, TOOLS, rules)
    assert len(findings) == 1 and "ALL" in findings[0], findings
    # no threshold in the rule -> silent
    rules2 = [{"id": "x", "type": "auto_action", "trigger_tools": ["open_close_window"],
               "requirement": "When AC turns on, close the windows."}]
    assert V.check_conditional_scope(draft, TOOLS, rules2) == []



def test_conditional_read_demanded_for_check_rules():
    """base_10 regression: policy 013 'check if high beam headlights are ON'
    must demand get_exterior_lights_status before set_head_lights_high_beams —
    matched by subject tokens (light~headlights), no keyword list."""
    tools = TOOLS + [
        {"function": {"name": "get_exterior_lights_status",
                      "parameters": {"type": "object", "properties": {}}}},
        {"function": {"name": "set_head_lights_high_beams",
                      "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}}}}},
    ]
    rules = [{"id": "013", "type": "auto_action",
              "trigger_tools": ["set_fog_lights", "set_head_lights_high_beams"],
              "requirement": "When activating fog lights, automatically check if low beam headlights "
                             "are ON and activate them if not, and check if high beam headlights are "
                             "ON and if so deactivate them."}]
    draft = {"tool_calls": [{"function": {"name": "set_head_lights_high_beams",
                                          "arguments": {"on": False}}}]}
    findings = V.check_gather(draft, _history(), tools, rules)
    assert any("get_exterior_lights_status" in f for f in findings), findings
    # already read -> silent
    done = _history("get_exterior_lights_status")
    assert not any("get_exterior_lights_status" in f
                   for f in V.check_gather(draft, done, tools, rules))


def test_conditional_read_does_not_revive_unrelated_demands():
    """The generalized matching must not resurrect the sunroof/trunk over-demand:
    the windows rule (conditional, 'more than 20%') matches only window reads."""
    rules = [RULES[1]]  # 011_windows
    draft = {"tool_calls": [{"function": {"name": "set_air_conditioning",
                                          "arguments": {"on": True}}}]}
    findings = V.check_gather(draft, _history(), TOOLS, rules)
    assert any("get_vehicle_window_positions" in f for f in findings), findings
    assert not any("sunroof" in f or "trunk" in f for f in findings), findings


def test_outward_duplicate_candidate():
    """base_70 regression: a second send_email in the same task -> candidate;
    the first send and non-outward repeats stay silent."""
    tools = TOOLS + [{"function": {"name": "send_email",
                                   "parameters": {"type": "object", "properties": {}}}}]
    draft = {"tool_calls": [{"function": {"name": "send_email",
                                          "arguments": {"contact_id": "c1", "text": "updated ETA"}}}]}
    fresh = [{"role": "user", "content": "Email Grace my ETA."}]
    assert V.check_outward_duplicate(draft, fresh, tools) == []
    hist = fresh + [{"role": "assistant", "tool_calls": [
        {"function": {"name": "send_email", "arguments": "{}"}}]}]
    findings = V.check_outward_duplicate(draft, hist, tools)
    assert len(findings) == 1 and "irreversible" in findings[0], findings
    # repeating a reversible write is NOT this gate's business
    d2 = {"tool_calls": [{"function": {"name": "set_fan_speed", "arguments": {"level": 2}}}]}
    h2 = fresh + [{"role": "assistant", "tool_calls": [
        {"function": {"name": "set_fan_speed", "arguments": "{}"}}]}]
    assert V.check_outward_duplicate(d2, h2, TOOLS) == []



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
    print(f"\n{'ALL TESTS PASS' if not failed else f'{failed} FAILURE(S)'}")
    sys.exit(1 if failed else 0)