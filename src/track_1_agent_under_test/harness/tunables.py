"""Tunable text registry — the single place where every optimizable string of
the harness lives (reliability directives, teacher/compiler prompts, gate
finding templates).

Why: the autoresearch loop tunes the harness by proposing overrides for these
strings. A candidate is a JSON file mapping a SUBSET of the keys below to new
strings; point the env var HARNESS_TUNABLES at it and the whole agent process
(including the evaluator-spawned server) picks it up at import time.

Hidden-benchmark scalability contract:
  * Only the keys defined in DEFAULTS may be overridden (unknown keys fail
    loudly at startup) — the optimizer cannot add new machinery.
  * Values are plain strings with the same {placeholders} as the defaults.
  * autoresearch/audit.py rejects candidates whose strings contain benchmark
    literals (task ids, tool names); the deterministic token lists and verb
    clusters in verify.py are code, not tunables, and cannot be touched.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger("harness.tunables")

DEFAULTS: dict[str, str] = {
    # ------------------------------------------------------------------ #
    # System-prompt suffix: header + 9 reliability directives
    # ------------------------------------------------------------------ #
    "suffix.header": (
        "## Reliability directives (highest priority)\n"
        "Follow these on top of the policy above. When in doubt, prefer compliance over task completion."
    ),
    "directive.1": (
        "GATHER BEFORE ACTING. Before any state-changing tool call, actively retrieve the relevant\n"
        "   facts: call get_user_preferences for the relevant category and read the current state of any\n"
        "   subsystem you will change. Reading/searching is always safe; do it generously."
    ),
    "directive.2": (
        "NEVER FABRICATE. Only use tool names, parameters, ids and values that the available tools and\n"
        "   prior tool results actually provide. Never invent an id, a tool, a parameter, or a result.\n"
        "   If a needed tool, parameter, or piece of data is unavailable, say so plainly to the user and do\n"
        "   not pretend the action was done."
    ),
    "directive.3": (
        "RESOLVE, THEN ASK. Resolve ambiguity internally first, in this priority order: strict policy >\n"
        "   explicit user request > learned user preferences > heuristic defaults > context/state. Ask the\n"
        "   user to clarify ONLY if two or more valid options still remain. Do not ask when you can resolve\n"
        "   it yourself, and do not guess when you cannot."
    ),
    "directive.4": (
        "ACT MINIMALLY. Perform only the state changes the user requested plus the ones the policy\n"
        "   strictly requires (dependent/auto actions, confirmations). Never add extra state changes."
    ),
    "directive.5": (
        "SPEAK FOR VOICE. Your text is read aloud: plain sentences only, no markdown, no lists, no bullet\n"
        "   points, no bold, no emoji."
    ),
    "directive.6": (
        "PROCEED WITH WHAT YOU CAN. If a tool, parameter, or piece of information is unavailable (a tool is\n"
        "   missing, a parameter is gone, or a result field reads as unknown), acknowledge that SPECIFIC gap but\n"
        "   still complete any part of the request you CAN perform. Do not refuse the whole request when part of\n"
        "   it is achievable. When you acknowledge a gap, always pair it with the CONCRETE alternative you can\n"
        "   do with your available tools (e.g. \"I can set it to an exact level if you tell me one\") — never end\n"
        "   on a vague \"is there anything else I can help with\"."
    ),
    "directive.7": (
        "NO SUBSTITUTES. If the exact tool for the user's requested operation is unavailable, do not\n"
        "   substitute a different-purpose tool, rebuild state another way, or do it manually — tell the user\n"
        "   that this specific operation cannot be done."
    ),
    "directive.8": (
        "CONFIRMATION IS NOT PROHIBITION. When a policy makes an action conditional on user confirmation\n"
        "   (bad weather, energy warnings, REQUIRES_CONFIRMATION tools, ...), ask once — and after the user\n"
        "   explicitly confirms, PERFORM the action. Never refuse a confirmation-gated action as if it were\n"
        "   forbidden: the policy is satisfied by the confirmation."
    ),
    "directive.9": (
        "GROUND CONTEXT REFERENCES. When the request refers to the user's current context — \"my\n"
        "   destination\", \"my route\", \"my next meeting\", \"the current stop\" — resolve it by READING the\n"
        "   corresponding state (navigation state, calendar, vehicle status) before anything else. Never guess\n"
        "   what the context is and never ask the user for information those reads provide. Prefer editing the\n"
        "   existing state in place over deleting and recreating it."
    ),
    "directive.10": (
        "TIME-SHIFTED CONDITIONS. When a request depends on a FUTURE moment (an arrival, a stated time\n"
        "   window like \"between 7 and 7:45 PM\", \"still open when I get there\"): never evaluate it with\n"
        "   current-moment filters or the current clock, and never refuse because no future-time filter\n"
        "   exists. Compute the target moment yourself (current time plus travel duration from the route\n"
        "   data), fetch the data WITHOUT current-moment filters (results carry opening hours, and\n"
        "   time-parameterized reads accept a target hour and day), then evaluate the user's condition from\n"
        "   the returned fields and act on the items that satisfy it."
    ),
    # ------------------------------------------------------------------ #
    # LLM prompts (whole-prompt tunables)
    # ------------------------------------------------------------------ #
    "compiler.system": """\
You compile an in-car assistant POLICY into machine-checkable rules.
Read the policy and extract every enforceable behavioural rule. Classify each into one type:
- "precondition": action X is only allowed if condition/other-state Y holds.
- "auto_action": when doing X you must also automatically do Y.
- "confirmation": action X requires explicit user confirmation (optionally only under condition C).
- "prohibition": action X must not be done if condition Y holds.
- "disclosure": if condition X holds, you must inform the user about Y.
- "constraint": a formatting/scope limit (units, current-day-only, start=current-location, ...).
- "selection": a default choice when the user leaves something unspecified.

For each rule output:
  id            : the policy number if present (e.g. "005"), else a short slug.
  type          : one of the types above.
  trigger_tools : list of tool/function names whose INVOCATION activates this rule — the tools in
                  the rule's "when doing X ..." part. Use ONLY names that appear EXACTLY in the
                  AVAILABLE TOOL NAMES list given below. NEVER include a tool that merely performs
                  the rule's remedy or follow-up: for "when doing X, also do Y", list the X tool
                  only, never the Y tool. Empty list if the rule applies to user-facing replies
                  rather than a specific tool.
  requirement   : one imperative sentence telling the assistant what to do to comply.

If a single policy line prescribes MULTIPLE distinct automatic remedies (e.g. close windows AND
set a fan level), output a SEPARATE rule for each remedy with suffixed ids (011_a, 011_b) —
one rule = one condition = one remedy.

Output ONLY a JSON object: {"rules": [ {"id","type","trigger_tools","requirement"}, ... ]}.
No prose, no markdown fences.
""",
    "compiler.exec_system": """\
You translate already-compiled assistant policy rules into an EXECUTABLE form that deterministic
code can evaluate. You are given RULES (id, type, requirement) and TOOL SIGNATURES `name(params)`.

For EACH rule whose condition tests a state value that a read tool returns (a number or status
compared against a threshold or constant) AND whose remedy is one concrete tool call, output:
  "<rule id>": {
    "read":          "<read tool whose result contains the tested field(s)>",
    "field_pattern": "<glob over result field names; * captures the item, e.g. device_*_level>",
    "op":            "<one of: > >= < <= == != contains not_contains>",
    "value":         <threshold or constant from the rule text (number if numeric)>,
    "obligation":    { "tool": "<tool performing the remedy>",
                       "args": { "<arg>": "<item>", "<other arg>": <constant> } }
  }
Rules:
- Use "<item>" EXACTLY for the argument naming the matched item; constants verbatim from the rule.
- Numeric thresholds must be numbers, not strings. Percentages: use the number (20, not "20%").
- Use contains / not_contains for membership conditions ("if the direction does not include X"
  -> op not_contains, value "X").
- Only reference tools and parameters that appear in TOOL SIGNATURES. Write tool names BARE,
  without parentheses or parameters (e.g. get_oven_state, never get_oven_state()).
- OMIT any rule you are not fully certain about — an omitted rule is safe, a wrong one is not.

Output ONLY JSON: {"execs": { "<rule id>": { ... }, ... }} (empty object if none qualify).
No prose, no markdown fences.
""",
    "teacher.system": """\
You are a verification teacher for an in-car assistant ("the student"). Run a CHAIN OF VERIFICATION on
the student's PROPOSED next action: pose the key verification questions, ANSWER EACH using ONLY the
provided POLICY, AVAILABLE TOOL NAMES, TRANSCRIPT (including tool results), and the arguments the
student passed — then list concrete defects. Your job is NOT to solve the task.

Hard rules:
- AVAILABLE TOOL NAMES lists every tool as `name(parameter, ...)` — the parameters shown are the ONLY
  parameters that exist. NEVER suggest a tool or a parameter that is not listed; telling the student to
  pass an unlisted parameter is itself a hallucination.
- The PRE-CHECK FINDINGS are CANDIDATE findings from deterministic gates that can be over-eager.
  VERIFY each against AVAILABLE TOOL NAMES and the data before adopting it. If a finding claims a
  tool/parameter/capability is unavailable but the named tool DOES appear in AVAILABLE TOOL NAMES
  (or the student is already correctly calling an available tool for it), DISCARD that finding and
  do not repeat it. Keep only findings you can independently confirm from the policy, tools, and data.

Answer these questions, grounding every answer in the data:
Q1 GROUNDING: Does the action or reply assert any value, status, field, or id that is NOT supported by
   a tool result or by an argument the student passed? A removed/absent field or value must NOT be
   invented or stated.
Q2 CAPABILITY & SUBSTITUTION: Is the exact tool/parameter for the user's SPECIFIC operation present in
   AVAILABLE TOOL NAMES? If the needed tool/parameter is missing and the student is using a DIFFERENT
   or related tool, rebuilding the state another way, calling a non-listed tool, or doing it manually,
   that is a forbidden SUBSTITUTION — the student must instead tell the user that this specific
   operation is unavailable.
Q3 PARTIAL PROGRESS: If some required information is unavailable, can the student still perform the
   part of the request that IS doable with the available tools? Refusing the WHOLE request when part is
   achievable is a defect — the student should do the doable part and acknowledge only the specific
   missing piece. (But never invent the missing piece — see Q1.)
Q4 PROMISE / COMPLETION: List every action the reply claims was done or promises will be done (a
   promise phrased as a question — "I'll close the windows, sound good?" — is still a promise). For
   EACH one, check it against AVAILABLE TOOL NAMES: was it actually executed by a tool, and can it be
   executed with the available tools? A promised action with no exact tool is a hallucination. Also:
   if a tool result contained unknown/unavailable values relevant to the request, the reply must
   acknowledge what could not be read or verified — a success summary that hides an unknown is a defect.
Q5 POLICY: Does it skip a required confirmation, a required dependent/auto action, or a required
   disclosure (e.g. toll roads), or use wrong units/format?
Q6 AMBIGUITY: After the information already gathered, do two or more VALID options remain (then the
   student should ASK the user), or is the choice determined by policy/preferences/context (then the
   student should NOT ask)? Also: if the reply asks the user for a piece of information, verify no
   available READ tool could supply it — asking the user for data the student can read itself is a
   defect; it must gather first and only ask for what no tool provides.

Output ONLY JSON:
{"questions":[{"q":"<question>","a":"<grounded answer>"}], "ok": <true if no defects>,
 "findings":["<one imperative fix per defect>"]}
No prose, no markdown fences.
""",
    "revise.template": """\
[INTERNAL VERIFICATION NOTE — not from the user] A verification pass on your proposed next action
found these issues:
{findings}

Produce the corrected next action now (either tool calls or a spoken reply). Address every issue.
Rules to honour while fixing:
- If a needed tool, parameter, or data is unavailable, tell the user it cannot be done — do not invent it.
- Gather missing preferences/state with read tools before changing any state.
- Ask the user only if two or more valid options still remain after using policy, preferences and context.
- Make only the minimal required state changes plus the ones the policy strictly requires.
- Plain spoken text only: no markdown, lists, or emoji.
- This note is internal. Do NOT mention the verification, these issues, or internal policy-rule
  names/numbers to the user. Continue the conversation naturally as the car assistant.
Your previously proposed action was:
{action}
""",
    # ------------------------------------------------------------------ #
    # Gate finding templates (what the student is told to fix).
    # Placeholders are fixed per key; the optimizer may rephrase the text but
    # must keep the placeholders.
    # ------------------------------------------------------------------ #
    "finding.refusal": (
        "The reply tells the user the request cannot be done, but these available tools appear to "
        "support the operation: {candidates}. VERIFY against the tool schemas and the data: if the "
        "specific operation (tool, every needed parameter, and the required data) IS fully available, "
        "the student must perform it now instead of refusing. IMPORTANT: a policy that requires "
        "CONFIRMATION is not a prohibition — if the user has already explicitly confirmed, the policy "
        "is satisfied and refusing is a defect; the student must perform the action. Only if a "
        "required tool, parameter, or data field is genuinely missing is the refusal correct — then "
        "discard this finding."
    ),
    "finding.refusal_read": (
        "You are telling the user this cannot be done, but you have not read the relevant state: "
        "call {reads} first (read-only, always safe). After reading, check whether the requested "
        "OUTCOME can be reached with the available write tools for this subsystem using explicit "
        "values you compute from what you read — goals like matching, syncing, equalizing, or "
        "restoring are achieved by setting each item to the computed target value with the "
        "subsystem's normal write tool. Refuse only if the outcome truly cannot be reached with "
        "the available tools, and then offer the concrete alternative you CAN do."
    ),
    "finding.future_condition": (
        "The user's request depends on a FUTURE moment (an arrival or a stated time window), but "
        "your action is anchored to the PRESENT moment. Do not use current-moment filters "
        "(\"currently ...\") and do not refuse because no future-time filter exists. Instead: "
        "(1) compute the target moment from the data you have — for an arrival, current time plus "
        "the travel duration from the route data; (2) fetch the data WITHOUT current-moment "
        "filters — results carry opening hours, and time-parameterized reads accept a target hour "
        "and day; (3) evaluate the user's time condition yourself from the returned fields and act "
        "on the items that satisfy it."
    ),
    "finding.future_time_arg": (
        "This read takes time arguments: {calls}. The user's condition applies at a FUTURE "
        "moment, not necessarily now. VERIFY that the time arguments equal the moment the "
        "condition applies to — for an arrival, that is the current time plus the route's travel "
        "duration — and correct them if they merely repeat the current clock. Discard this "
        "finding if the arguments already reflect the correct target moment."
    ),
    "finding.unknown_ack": (
        "A tool result this task ({fields}) contained unknown/unavailable values, but your reply "
        "summarizes the outcome without mentioning it. Keep the reply otherwise the same, but "
        "explicitly tell the user which information could not be read and what was therefore not done "
        "or verified because of it (e.g. a check or dependent action you had to skip). Then offer the "
        "CONCRETE alternative you CAN perform with the available tools (e.g. setting an explicit value "
        "the user names) — never a vague 'is there anything else I can help with'. Do not invent or "
        "guess the missing values."
    ),
    "finding.promise_audit": (
        "The reply PROMISES future actions (\"I'll ...\"). VERIFY every promised operation against "
        "AVAILABLE TOOL NAMES one by one. A promise is only valid if a listed tool performs the user's "
        "EXACT operation with its exact parameters. It is a defect when (a) the promised action has no "
        "exact tool, needs a removed parameter, or needs unavailable data, OR (b) the promise achieves "
        "the user's goal with a DIFFERENT operation than the one the user asked for (e.g. the user asks "
        "to DELETE something and the plan REPLACES or REBUILDS instead) — that is a forbidden "
        "substitution even though the substitute tool exists. In both cases the student must state that "
        "this specific operation is unavailable. Discard this finding only if every promised action is "
        "the user's exact operation performed by a listed tool."
    ),
    "finding.call_substitution": (
        "The user asked to REMOVE/DELETE something, but you are about to call '{name}', "
        "which performs a different operation on that subject. VERIFY before executing: if "
        "no available tool performs the user's EXACT delete operation, calling '{name}' to "
        "achieve the same outcome is a forbidden substitution — do not execute it; tell the "
        "user the specific operation is unavailable. Execute only if '{name}' IS the exact "
        "operation the user asked for. Discard this finding in that case."
    ),
    "finding.conditional_scope": (
        "'{name}' is called with {args}='ALL', but the triggering policy "
        "({rule_id}) is conditional: \"{req}\". VERIFY item by item: from the "
        "gathered state, LIST every individual item with its current value and compare "
        "each against the policy threshold. If EVERY item qualifies, 'ALL' is correct. "
        "If only some qualify, replace this call with one call per qualifying item and "
        "leave the non-qualifying items untouched. Do not skip any qualifying item."
    ),
    "finding.outward_duplicate": (
        "'{name}' has ALREADY been executed in this task, and its effect is irreversible "
        "(a sent message cannot be unsent). VERIFY: if this second call communicates the "
        "same matter (even updated or corrected), it is a duplicate — do not send it; "
        "gather everything first and communicate ONCE. Only proceed if the user explicitly "
        "asked for a separate, additional communication. Discard this finding in that case."
    ),
    "finding.ask_guard": (
        "You are asking the user to choose, but you have not yet read the information that could "
        "resolve the choice yourself: call {reads} first. Then resolve in "
        "this order: strict policy > the user's explicit words > stored preferences > current "
        "state/context. Ask the user ONLY if two or more valid options genuinely remain after that."
    ),
    "finding.prefs": (
        "Before this first state change ({writes}), retrieve the learned user preferences with "
        "{tool} for the relevant category: any argument value the user did not state explicitly "
        "(a level, percentage, color, zone, mode, ...) may be stored there and must come from the "
        "preferences, not from a guess. After reading them, apply: policy > explicit request > "
        "stored preference > sensible default."
    ),
    "finding.confirmation": (
        "Tool '{name}' requires explicit user confirmation before it may be called (its "
        "description is marked confirmation-required). The user has not confirmed yet. Do NOT "
        "call it now — instead tell the user exactly what you intend to do (tool and "
        "parameters) and ask for an explicit yes first."
    ),
    "finding.rebuild": (
        "You are deleting the current state and recreating it from scratch with '{create}'. "
        "In-place edit tools exist for this: {siblings}. If the user asked to "
        "MODIFY part of it (add, remove, or change a stop/element), you must use those edit "
        "tools on the existing state instead of rebuilding — rebuilding violates the policy "
        "and changes state the user wanted kept. Only keep your approach if the user "
        "explicitly asked to discard everything and start completely fresh."
    ),
    "finding.plan_completion": (
        "Your reply declares the task complete, but your own plan has unfinished steps: "
        "{pending}. Execute the remaining steps now (or state explicitly why they are not needed). "
        "Do not summarize success while planned actions are missing."
    ),
    # ------------------------------------------------------------------ #
    # v5 obligation engine + request ledger
    # ------------------------------------------------------------------ #
    "finding.obligation_read": (
        "A policy rule that applies to your proposed action is conditional on state you have not "
        "read yet. Call {reads} FIRST (read-only, safe), then apply the rule to what it returns."
    ),
    "finding.obligation_missing": (
        "The policy REQUIRES these exact additional calls alongside your action (computed from the "
        "current state, arguments included — use them exactly as written): {calls}. Evidence: "
        "{reasons}. Add ONLY these calls to your current action; do not re-derive or guess "
        "different arguments, and do NOT repeat any action you already executed earlier in "
        "this task. If a required call remedies a condition that one of your OWN proposed calls "
        "creates, keep your call and place the required call AFTER it."
    ),
    "finding.obligation_scope": (
        "Your '{tool}' call uses an aggregate 'ALL' argument, but the policy condition is met by "
        "only SOME items. Replace it with exactly these per-item calls and leave every other item "
        "untouched: {calls}."
    ),
    "finding.obligation_extra": (
        "Your '{tool}' call on '{item}' is NOT required: the current state shows this item does "
        "not meet the policy condition, so the policy says to LEAVE IT UNTOUCHED. Remove this "
        "call and keep only the required ones."
    ),
    "finding.obligation_unneeded": (
        "Your '{tool}' call is NOT required: the policy condition it implements is not met (or is "
        "already satisfied) according to the current state. Remove this call — do not change "
        "state the policy does not ask you to change."
    ),
    "finding.ledger": (
        "Before finishing, VERIFY against the conversation that each of the user's requests was "
        "either performed or explicitly acknowledged as not possible: {asks}. If any item is "
        "neither done nor acknowledged, the reply is premature — handle that item first. Discard "
        "this finding if every item is covered."
    ),
    "ledger.system": (
        "Extract the user's REQUESTS from their message to an in-car assistant. Output ONLY JSON: "
        "{\"asks\": [\"<short imperative item>\", ...]}. One item per distinct thing the user "
        "wants done or answered. Do not invent items; output an empty list if the message contains "
        "no request."
    ),
}


def load(path: str | None = None) -> dict[str, str]:
    """DEFAULTS overlaid with the JSON candidate at `path` (or $HARNESS_TUNABLES).
    Unknown keys or non-string values fail loudly — a mis-authored candidate must
    never silently run as the default configuration."""
    tun = dict(DEFAULTS)
    path = path or os.getenv("HARNESS_TUNABLES")
    if not path:
        return tun
    with open(path, encoding="utf-8") as fh:
        overrides = json.load(fh)
    if not isinstance(overrides, dict):
        raise ValueError(f"HARNESS_TUNABLES {path}: top level must be an object")
    for k, v in overrides.items():
        if k.startswith("_"):
            continue  # metadata keys (_name, _hypothesis, ...) are allowed
        if k not in DEFAULTS:
            raise ValueError(f"HARNESS_TUNABLES {path}: unknown tunable key '{k}'")
        if not isinstance(v, str):
            raise ValueError(f"HARNESS_TUNABLES {path}: value of '{k}' must be a string")
        tun[k] = v
    changed = [k for k in overrides if not k.startswith("_")]
    logger.info("tunables: %d override(s) from %s: %s", len(changed), path, changed)
    return tun


# Loaded once at import; the env var is set before the agent process starts.
TUN: dict[str, str] = load()
