"""TEACHER verification (CoVe stages S2/S3).

Given the student's proposed action, produce a list of grounded findings:
  * Tier A (deterministic): tool/param existence + id grounding + gather guard.
  * Tier A (deterministic, dynamic): compiled-policy advisories.
  * Tier B/C (LLM critic): soft policy + ambiguity, grounded in the transcript.

Returns plain strings (imperative fixes). An empty list means "verified".
Every sub-check is fail-safe: an internal error contributes no findings rather
than crashing the turn.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from . import policy as policy_mod
from .policy import resolved_triggers, tokens
from .llm import call_llm, parse_json_object
from .provenance import ProvenanceLedger, extract_ids
from .prompts import (
    TEACHER_SYSTEM, TEACHER_USER_TEMPLATE,
    CLAIM_GROUNDING_SYSTEM, CLAIM_GROUNDING_TEMPLATE,
)

# Verbs in a precondition requirement that imply a subsystem must be *changed*
# (and therefore need a write tool to achieve). Generic, not tool-specific.
_CHANGE_VERBS = {
    "open", "opened", "close", "closed", "set", "activate", "activated", "deactivate",
    "turn", "enable", "enabled", "disable", "disabled", "adjust", "raise", "lower",
    "increase", "decrease", "fully",
}

logger = logging.getLogger("harness.verify")

# Heuristic: tool-name prefixes that only READ state (safe / free to call).
_READ_PREFIXES = (
    "get_", "search_", "list_", "find_", "lookup_", "retrieve_", "calculate_",
    "compute_", "check_", "read_",
)
_READ_EXACT = {"planning_tool", "think", "note_intermediate_result", "datetime", "math"}


def _is_read_tool(name: str) -> bool:
    if not name:
        return False
    if name in _READ_EXACT:
        return True
    return name.startswith(_READ_PREFIXES)


def _norm_tok(t: str) -> str:
    """Plural-tolerant token: 'positions' -> 'position', 'windows' -> 'window'.
    Lets us compare requirement nouns to tool-name tokens despite singular/plural
    differences (the tokenizer does no stemming)."""
    return t[:-1] if len(t) > 4 and t.endswith("s") else t


def tool_index(tools: Optional[list]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if name:
            idx[name] = fn.get("parameters") or {}
    return idx


def _calls(draft: dict) -> list[dict]:
    return draft.get("tool_calls") or []


def _args(tc: dict) -> dict:
    fn = tc.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


def describe_action(draft: dict) -> str:
    calls = _calls(draft)
    if calls:
        parts = []
        for tc in calls:
            fn = tc.get("function") or {}
            parts.append(f"{fn.get('name')}({json.dumps(_args(tc), ensure_ascii=False)})")
        return "TOOL CALLS: " + "; ".join(parts)
    content = draft.get("content") or ""
    return "REPLY TO USER: " + content.strip()


def _recent_context(messages: list[dict], last_n: int = 6) -> str:
    """Recent user text + tool-result contents, for relevance filtering of
    reply-level policy advisories."""
    chunks: list[str] = []
    for m in messages[-last_n:]:
        if m.get("role") in ("user", "tool") and m.get("content"):
            chunks.append(str(m["content"]))
    return " ".join(chunks)


def _render_transcript(messages: list[dict], max_chars: int = 12000) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue  # policy is passed separately
        if role == "assistant" and m.get("tool_calls"):
            names = ", ".join(
                (tc.get("function") or {}).get("name", "?") for tc in m["tool_calls"]
            )
            lines.append(f"ASSISTANT(tool_calls): {names}")
            if m.get("content"):
                lines.append(f"ASSISTANT: {m['content']}")
        elif role == "tool":
            lines.append(f"TOOL_RESULT[{m.get('name','?')}]: {m.get('content','')}")
        elif role == "user":
            lines.append(f"USER: {m.get('content','')}")
        elif role == "assistant":
            lines.append(f"ASSISTANT: {m.get('content','')}")
    text = "\n".join(lines)
    if len(text) > max_chars:  # keep the most recent context
        text = "...\n" + text[-max_chars:]
    return text


# --------------------------------------------------------------------------- #
# Tier A — deterministic checks
# --------------------------------------------------------------------------- #
def check_schema(draft: dict, tools: Optional[list]) -> list[str]:
    """Hallucination gate: called tool + params must exist in the provided tools."""
    findings: list[str] = []
    idx = tool_index(tools)
    if not idx:
        return findings  # no tool list -> cannot judge; stay silent
    for tc in _calls(draft):
        name = (tc.get("function") or {}).get("name")
        if name not in idx:
            findings.append(
                f"The tool '{name}' is not in the available tools, so you cannot use it. "
                f"Tell the user this capability is unavailable instead of calling it."
            )
            continue
        schema = idx[name] or {}
        props = set((schema.get("properties") or {}).keys())
        required = schema.get("required") or []
        args = _args(tc)
        # NOTE: no `if props:` guard — when a tool's schema lists NO properties
        # (e.g. its only parameter was removed), every supplied argument is
        # invented. The old guard silently skipped exactly that case (h_24:
        # open_close_sunshade(percentage=60) sailed through and insta-failed).
        prop_schemas = schema.get("properties") or {}
        for k, v in args.items():
            if k not in props:
                findings.append(
                    f"Parameter '{k}' does not exist on tool '{name}'. Do not invent parameters; "
                    f"if you need it and it is unavailable, tell the user."
                )
                continue
            # Value-level validation, driven entirely by the provided schema:
            # an argument outside a declared enum WILL fail at execution (a
            # scored error) — catch it pre-execution and let the agent tell
            # the user the real options instead.
            pschema = prop_schemas.get(k) or {}
            enum = pschema.get("enum")
            if enum and not isinstance(v, (dict, list)) and v not in enum:
                findings.append(
                    f"Value '{v}' is not a valid option for '{name}.{k}'. The only valid values "
                    f"are: {', '.join(str(e) for e in enum)}. If the user asked for something not "
                    f"in this list, tell them it is unavailable and offer these options."
                )
        for req in required:
            if req not in args:
                findings.append(f"Required parameter '{req}' is missing for tool '{name}'.")
    return findings


def check_grounding(draft: dict, prov: ProvenanceLedger) -> list[str]:
    """Hallucination gate: id-like argument values must trace to provided data."""
    findings: list[str] = []
    for tc in _calls(draft):
        name = (tc.get("function") or {}).get("name")
        for k, v in _args(tc).items():
            if not isinstance(v, str):
                continue
            for tok in extract_ids(v):
                if not prov.is_grounded(tok):
                    findings.append(
                        f"The id '{tok}' used in '{name}.{k}' was never returned by a tool. "
                        f"Look it up with a search/get tool instead of inventing it."
                    )
    return findings


# Maps a concept appearing in a compiled policy rule's requirement text to the
# substring(s) of the read tool that supplies it. Used to derive, dynamically,
# which read must precede a state change (e.g. a weather-gated sunroof needs
# get_weather first). Driven by the compiled rules + the available tools, so it
# generalizes to policies/tools we have not seen.
_GATHER_KEYWORDS = {
    "weather": ("weather",),
    "preference": ("preference",),
    "position": ("position",),
    "state of charge": ("charge", "battery"),
    "battery": ("charge", "battery"),
    "calendar": ("calendar",),
    "contact": ("contact",),
    "location": ("location",),
    "charging": ("charg",),
}


def _read_tool_names(tools) -> set[str]:
    return {n for n in tool_index(tools) if _is_read_tool(n)}


def _required_reads(action_name: str, rules: Optional[list], read_tools: set[str], all_tools: set[str]) -> set[str]:
    """Reads that a compiled policy rule makes a precondition of `action_name`
    (rule triggers are fuzzy-resolved to real tool names)."""
    required: set[str] = set()
    for rule in rules or []:
        if action_name not in resolved_triggers(rule, all_tools):
            continue
        text = (rule.get("requirement") or "").lower()
        req_tok = {_norm_tok(t) for t in tokens(text) if len(t) > 3}
        for kw, patterns in _GATHER_KEYWORDS.items():
            if kw not in text:
                continue
            patt_norm = {_norm_tok(p) for p in patterns} | {_norm_tok(kw)}
            for rt in read_tools:
                if not any(p in rt for p in patterns):
                    continue
                # The read tool's subject = its tokens minus the (attribute)
                # keyword. Demand it only if it has no distinct subject (a
                # single-purpose read like get_weather) or its subject actually
                # matches the rule's requirement — so a window-position rule does
                # NOT pull in sunroof/trunk position reads.
                subject = {_norm_tok(t) for t in tokens(rt) if len(t) > 3} - patt_norm
                if not subject or (subject & req_tok):
                    required.add(rt)
    return required


def check_gather(draft: dict, messages: list[dict], tools=None, rules=None) -> list[str]:
    """Gather guard: don't change state before retrieving the information the
    request and the applicable policy rules depend on (preferences + any
    policy-required read such as get_weather for weather-gated actions)."""
    calls = _calls(draft)
    state_changes = [
        name
        for tc in calls
        if (name := (tc.get("function") or {}).get("name", "")) and not _is_read_tool(name)
    ]
    if not state_changes:
        return []

    history_tools = {
        (tc.get("function") or {}).get("name")
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    read_tools = _read_tool_names(tools)
    all_tools = set(tool_index(tools).keys())
    findings: list[str] = []

    # Policy-required reads (deterministic, dynamic from the compiled rules).
    for name in state_changes:
        for rt in sorted(_required_reads(name, rules, read_tools, all_tools)):
            if rt not in history_tools:
                findings.append(
                    f"Before calling '{name}', you must first call '{rt}': the policy makes this "
                    f"action conditional on that information. Gather it now, then apply the policy "
                    f"(ask the user for confirmation if the policy requires it)."
                )

    # NOTE: the get_user_preferences reminder is advisory tier now — see
    # gather_prefs_advisory(). Hard gather findings are only the reads a compiled
    # policy rule makes a precondition of the action.
    return findings


# General completion-claim vs admission language (not tool/task specific).
_DONE_PATTERNS = (
    "done", "there you go", "all set", "you got it", "i've", "i have ", "taken care",
    "consider it", "opened", "closed", "removed", "deleted", "cancelled", "canceled",
    " set ", "changed", "updated", "adjusted", "activated", "deactivated", "completed",
    "turned on", "turned off", "switched",
)
_INTENT_PATTERNS = ("i'll ", "i will ", "going to ", "let me ")
_ADMIT_PATTERNS = (
    "can't", "cannot", "can not", "unable", "isn't available", "is not available",
    "not available", "unavailable", "not possible", "couldn't", "could not", "no tool",
    "don't have", "do not have", "not able", "not supported", "won't be able",
    "i'm sorry", "i am sorry", "unfortunately", "missing the", "am missing",
    "not equipped", "no way to", "lack the",
)


def _any_state_change_in_history(messages: list[dict]) -> bool:
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if not _is_read_tool((tc.get("function") or {}).get("name", "")):
                return True
    return False


def check_completion(draft: dict, messages: list[dict]) -> list[str]:
    """Anti-fabrication gate: a user-facing reply that declares an action was/
    will be done while NO state-changing tool has ever run for this task is a
    hallucinated completion. Conservative: skips questions/proposals (they end
    with '?') and skips replies that honestly admit incapability. Purely a
    language+structure heuristic — no tool/task-specific knowledge."""
    if _calls(draft):
        return []  # it is taking an action, not just claiming one
    content = (draft.get("content") or "").strip()
    if not content:
        return []
    low = content.lower()
    if "?" in content:
        return []  # a question / proposal, not a claim of completion
    if any(p in low for p in _ADMIT_PATTERNS):
        return []  # correctly telling the user it cannot be done
    if _any_state_change_in_history(messages):
        return []  # an action really was performed earlier this task
    if any(p in low for p in _DONE_PATTERNS + _INTENT_PATTERNS):
        return [
            "Your reply says an action was or will be done, but no tool that performs it has been "
            "called for this request. If the required tool is unavailable, tell the user you cannot "
            "do it; otherwise call the real tool now. Never claim something is done when it was not."
        ]
    return []


def check_loops(draft: dict, messages: list[dict], repeat_threshold: int = 2) -> list[str]:
    """Retry-loop / no-progress detector. If the proposed call has already been
    made identically `repeat_threshold` times (i.e. this is the 3rd+ attempt), the
    agent is stuck — it should stop retrying and tell the user it cannot be done.
    Deterministic, general, no tool/task knowledge."""
    if not _calls(draft):
        return []
    seen: dict = {}
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            a = fn.get("arguments")
            if isinstance(a, (dict, list)):
                a = json.dumps(a, sort_keys=True)
            seen[(fn.get("name"), a)] = seen.get((fn.get("name"), a), 0) + 1
    findings: list[str] = []
    flagged: set = set()
    for tc in _calls(draft):
        fn = tc.get("function") or {}
        a = fn.get("arguments")
        if isinstance(a, (dict, list)):
            a = json.dumps(a, sort_keys=True)
        name = fn.get("name")
        if name and seen.get((name, a), 0) >= repeat_threshold and name not in flagged:
            flagged.add(name)
            findings.append(
                f"You have already called '{name}' with the same arguments {seen[(name, a)]} times "
                f"without resolving the request. Do not retry it again — if it is not working or the "
                f"needed data/capability is unavailable, tell the user it cannot be done."
            )
    return findings


# Marker: the tool description STARTS with an ALL-CAPS token containing
# "CONFIRM" (REQUIRES_CONFIRMATION, CONFIRMATION_REQUIRED, MUST_CONFIRM, ...).
# Structural, derived from the runtime tool descriptions — not a literal
# benchmark string — so it generalizes to hidden-set variants of the marker.
_CONFIRM_MARKER_RE = re.compile(r"^\s*[A-Z_]*CONFIRM[A-Z_]*\b")

# Explicit-affirmation cues in the user's latest message. Generic conversational
# English, not benchmark-specific.
_AFFIRM_PATTERNS = (
    "yes", "yeah", "yep", "sure", "okay", "ok,", "ok.", "ok!", " ok", "confirm",
    "go ahead", "proceed", "do it", "please do", "that's fine", "thats fine",
    "sounds good", "sound good", "correct", "affirmative", "still want", "i do",
)


def _tool_descriptions(tools) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if name:
            out[name] = str(fn.get("description") or "")
    return out


def check_confirmation(draft: dict, messages: list[dict], tools) -> list[str]:
    """Confirmation gate (hard): a tool whose description carries a
    confirmation-required marker may only be called after the user explicitly
    affirmed in their LATEST message. Otherwise the agent must first present
    the intended action + parameters and ask. Fully generic — the requirement
    is read from the runtime tool descriptions, the affirmation from the
    transcript. (base_2 failure: trunk door opened with no confirmation.)"""
    calls = _calls(draft)
    if not calls:
        return []
    # An affirmation only counts when it ANSWERS a question: the user's latest
    # message contains an affirmative cue AND the assistant's preceding
    # user-facing message asked one. A first request that merely contains
    # "please do ..." has confirmed nothing yet.
    last_user = ""
    asked_before = False
    seen_user = False
    for m in reversed(messages):
        if not seen_user and m.get("role") == "user" and m.get("content"):
            last_user = str(m["content"]).lower()
            seen_user = True
            continue
        if seen_user and m.get("role") == "assistant" and m.get("content"):
            asked_before = "?" in str(m["content"])
            break
    user_affirmed = asked_before and any(p in last_user for p in _AFFIRM_PATTERNS)
    if user_affirmed:
        return []
    descs = _tool_descriptions(tools)
    findings: list[str] = []
    for tc in calls:
        name = (tc.get("function") or {}).get("name") or ""
        if _CONFIRM_MARKER_RE.match(descs.get(name, "")):
            findings.append(
                f"Tool '{name}' requires explicit user confirmation before it may be called (its "
                f"description is marked confirmation-required). The user has not confirmed yet. Do NOT "
                f"call it now — instead tell the user exactly what you intend to do (tool and "
                f"parameters) and ask for an explicit yes first."
            )
    return findings


def check_refusal(draft: dict, messages: list[dict], tools) -> list[str]:
    """Anti-over-refusal gate (the mirror of check_completion).

    Fires a CANDIDATE finding when the reply tells the user something cannot be
    done, yet an available *write* tool's name matches the user's request — the
    exact failure where the agent refuses a doable action (fatal on tasks where
    the final state is scored). The finding is phrased for the CoVe teacher to
    VERIFY, because a refusal can still be correct when only a parameter or a
    data field is missing (hallucination tasks). Deterministic, generic — token
    overlap between the request and the tool inventory, no hardcoded names."""
    if _calls(draft):
        return []
    content = (draft.get("content") or "").strip()
    if not content:
        return []
    low = content.lower()
    if not any(p in low for p in _ADMIT_PATTERNS):
        return []
    # Scan the last few user messages, not just the latest: a confirmation like
    # "yes, I still want it" carries no subsystem words — the request they
    # confirm is in an earlier turn (base_0: the gate went silent on the final
    # refusal for exactly this reason).
    user_texts: list[str] = []
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            user_texts.append(str(m["content"]))
            if len(user_texts) >= 3:
                break
    if not user_texts:
        return []
    user_text = " ".join(user_texts)
    req = {_norm_tok(t) for t in tokens(user_text) if len(t) > 3} - _CHANGE_VERBS
    if not req:
        return []
    called = {
        (tc.get("function") or {}).get("name")
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    candidates: list[str] = []
    for name in sorted(tool_index(tools)):
        if _is_read_tool(name) or name in called:
            continue
        subject = {_norm_tok(t) for t in tokens(name) if len(t) > 3} - _CHANGE_VERBS
        if subject and (subject & req):
            candidates.append(name)
    if not candidates:
        return []
    return [
        "The reply tells the user the request cannot be done, but these available tools appear to "
        f"support the operation: {', '.join(candidates[:4])}. VERIFY against the tool schemas and the "
        "data: if the specific operation (tool, every needed parameter, and the required data) IS fully "
        "available, the student must perform it now instead of refusing. IMPORTANT: a policy that "
        "requires CONFIRMATION is not a prohibition — if the user has already explicitly confirmed, the "
        "policy is satisfied and refusing is a defect; the student must perform the action. Only if a "
        "required tool, parameter, or data field is genuinely missing is the refusal correct — then "
        "discard this finding."
    ]


_UNKNOWN_VALUES = ('"unknown"', ": null", "'unknown'", '"n/a"', '"not available"')


def check_unknown_ack(draft: dict, messages: list[dict]) -> list[str]:
    """Unknown-data acknowledgment gate (CANDIDATE, teacher-verified).

    If a tool result during this task carried an unknown/null field and the
    current draft is a user-facing reply that claims success without any
    admit-language, the reply is silently papering over data it never had —
    the exact h_16 failure (acted correctly, then summarized as if everything
    was known). Deterministic detection: scan tool results for unknown markers."""
    if _calls(draft):
        return []
    content = (draft.get("content") or "").strip()
    if not content:
        return []
    low = content.lower()
    if any(p in low for p in _ADMIT_PATTERNS) or "unknown" in low:
        return []  # it already acknowledges something is unreadable
    if not any(p in low for p in _DONE_PATTERNS + _INTENT_PATTERNS):
        return []  # not a completion/summary reply
    unknown_fields: list[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        body = str(m.get("content") or "").lower()
        if any(u in body for u in _UNKNOWN_VALUES):
            unknown_fields.append(str(m.get("name") or "a tool result"))
    if not unknown_fields:
        return []
    return [
        f"A tool result this task ({', '.join(sorted(set(unknown_fields))[:3])}) contained "
        "unknown/unavailable values, but your reply summarizes the outcome without mentioning it. "
        "Keep the reply otherwise the same, but explicitly tell the user which information could "
        "not be read and what was therefore not done or verified because of it (e.g. a check or "
        "dependent action you had to skip). Do not invent or guess the missing values."
    ]


def check_promises(draft: dict, messages: list[dict], tools) -> list[str]:
    """Promise-audit gate (CANDIDATE, teacher-verified).

    A reply that promises future actions ("I'll close all the windows... sound
    good?") must only promise operations the available tools can actually
    perform — promising a removed capability is a hallucination even when
    phrased as a question, which is exactly how it escapes check_completion's
    '?' exemption (h_38/h_48 failures). Emits a candidate directing the teacher
    to cross-check every promised action against the tool inventory."""
    if _calls(draft):
        return []
    content = (draft.get("content") or "").strip()
    if not content:
        return []
    low = content.lower()
    if not any(p in low for p in _INTENT_PATTERNS):
        return []
    if any(p in low for p in _ADMIT_PATTERNS):
        return []  # already acknowledging limits; refusal gate owns that side
    return [
        "The reply PROMISES future actions (\"I'll ...\"). VERIFY every promised operation against "
        "AVAILABLE TOOL NAMES one by one. A promise is only valid if a listed tool performs the user's "
        "EXACT operation with its exact parameters. It is a defect when (a) the promised action has no "
        "exact tool, needs a removed parameter, or needs unavailable data, OR (b) the promise achieves "
        "the user's goal with a DIFFERENT operation than the one the user asked for (e.g. the user asks "
        "to DELETE something and the plan REPLACES or REBUILDS instead) — that is a forbidden "
        "substitution even though the substitute tool exists. In both cases the student must state that "
        "this specific operation is unavailable. Discard this finding only if every promised action is "
        "the user's exact operation performed by a listed tool."
    ]


def gather_prefs_advisory(draft: dict, messages: list[dict], tools) -> list[str]:
    """Preferences gate. HARD again (one-shot per task): a disambiguation run
    proved the stored preferences literally contained the expected values
    ('sunroof default 50%', 'PURPLE for evening') and the LLM teacher discarded
    the advisory version — the agent then guessed 100% / asked the user, both
    scored failures. Reads are free; the cost is one detour turn, once."""
    calls = _calls(draft)
    state_changes = [
        name
        for tc in calls
        if (name := (tc.get("function") or {}).get("name", "")) and not _is_read_tool(name)
    ]
    if not state_changes:
        return []
    history_tools = {
        (tc.get("function") or {}).get("name")
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    prefs_tools = {n for n in _read_tool_names(tools) if "preference" in n}
    if not prefs_tools or (prefs_tools & history_tools):
        return []
    return [
        "Before this first state change ("
        + ", ".join(sorted(set(state_changes)))
        + f"), retrieve the learned user preferences with {sorted(prefs_tools)[0]} for the relevant "
        "category: any argument value the user did not state explicitly (a level, percentage, color, "
        "zone, mode, ...) may be stored there and must come from the preferences, not from a guess. "
        "After reading them, apply: policy > explicit request > stored preference > sensible default."
    ]


# Choice-questions ask the user to pick or specify a value ("which ...?",
# "what color ...?"). Confirmation questions ("should I proceed?") are exempt —
# the confirmation gate requires those. Interrogative structure, not benchmark
# vocabulary, so it generalizes.
_CHOICE_Q_RE = re.compile(r"\b(which|what|how (?:much|many|warm|cold|high|low|far|fast))\b")


def check_ask_guard(draft: dict, messages: list[dict], tools, rules=None) -> list[str]:
    """Ask-guard (hard): do not ask the user to CHOOSE before resolving
    internally. If the draft asks a choice-question while the learned
    preferences and/or the read tools matching the request subject are still
    unread, force the gather first — the ask is only legitimate once the free
    reads are exhausted. Additionally, when the choice-subject matches a WRITE
    tool, the compiled policy's required reads for that tool are demanded too
    (the policy itself links e.g. fog lights to weather — reading it can
    resolve the choice without asking). All rule-driven, nothing hardcoded."""
    if _calls(draft):
        return []
    content = (draft.get("content") or "").strip()
    if not content or "?" not in content:
        return []
    if not _CHOICE_Q_RE.search(content.lower()):
        return []  # a yes/no confirmation, not a choice-question
    history_tools = {
        (tc.get("function") or {}).get("name")
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    read_tools = _read_tool_names(tools)
    unread: set[str] = {n for n in read_tools if "preference" in n} - history_tools
    # read tools whose subject matches the user's request words
    user_text = " ".join(
        str(m["content"]) for m in messages if m.get("role") == "user" and m.get("content")
    )[-1500:]
    req = {_norm_tok(t) for t in tokens(user_text) if len(t) > 3} - _CHANGE_VERBS
    for rt in read_tools - history_tools:
        subject = {_norm_tok(t) for t in tokens(rt) if len(t) > 3} - _CHANGE_VERBS
        if subject & req:
            unread.add(rt)
    # Policy-linked reads: if the choice-subject matches a write tool, the
    # compiled rules may make that tool conditional on information (weather,
    # positions, ...) that can resolve the choice — demand those reads as well.
    policy_linked: set[str] = set()
    if rules:
        all_tools = set(tool_index(tools).keys())
        for wt in all_tools - read_tools:
            subject = {_norm_tok(t) for t in tokens(wt) if len(t) > 3} - _CHANGE_VERBS
            if subject & req:
                policy_linked |= _required_reads(wt, rules, read_tools, all_tools) - history_tools
        unread |= policy_linked
    if not unread:
        return []  # everything gatherable was gathered; asking is now legitimate
    # Policy-linked reads FIRST: they encode the resolution the policy itself
    # prescribes (a plain alphabetical cap once dropped get_weather — the one
    # read that decided the task).
    ordered = sorted(policy_linked) + sorted(unread - policy_linked)
    return [
        "You are asking the user to choose, but you have not yet read the information that could "
        f"resolve the choice yourself: call {', '.join(ordered[:6])} first. Then resolve in "
        "this order: strict policy > the user's explicit words > stored preferences > current "
        "state/context. Ask the user ONLY if two or more valid options genuinely remain after that."
    ]


def check_capability(draft: dict, tools, rules) -> list[str]:
    """Capability/admit gate (hallucination: missing tool).

    For each precondition rule that applies to a proposed state-changing action,
    find the subsystem the precondition demands be *changed* (a subsystem noun
    that appears in some tool name + a change verb in the requirement). If no
    *write* tool for that subsystem is available, the precondition is
    unsatisfiable -> the agent must admit it cannot fulfil the request rather
    than substitute another tool. Fully generic: derived from the tool inventory
    + compiled rules, no hardcoded tool/subsystem names."""
    calls = _calls(draft)
    state_changes = {
        name for tc in calls
        if (name := (tc.get("function") or {}).get("name", "")) and not _is_read_tool(name)
    }
    if not state_changes or not rules:
        return []

    idx = tool_index(tools)
    all_tools = set(idx.keys())
    if not all_tools:
        return []
    write_tools = {n for n in all_tools if not _is_read_tool(n)}
    read_tools = {n for n in all_tools if _is_read_tool(n)}
    # subsystem nouns the vehicle exposes = tokens that appear in any tool name
    # (plural-normalized, so a rule's "windows" matches the tools' "window")
    subsystem_tokens: set[str] = set()
    for n in all_tools:
        subsystem_tokens |= {_norm_tok(t) for t in tokens(n) if len(t) > 3}
    # tokens a WRITE tool can directly control (plural-normalized)
    write_tok: set[str] = set()
    for w in write_tools:
        write_tok |= {_norm_tok(t) for t in tokens(w) if len(t) > 3}

    def _controllable(sub: str) -> bool:
        """Is requirement noun `sub` actually controllable by some available tool?
        Direct: a write tool is named for it. Indirect: a read tool links it to a
        writable subsystem (e.g. 'position' co-occurs with 'window' in
        get_vehicle_window_positions, and 'window' is writable via
        open_close_window). This stops attribute nouns (position/level/settings)
        — which only ever appear in *read* tool names — from being misread as
        uncontrollable subsystems and producing false 'tool unavailable' findings."""
        s = _norm_tok(sub)
        if s in write_tok:
            return True
        for r in read_tools:
            rtok = {_norm_tok(t) for t in tokens(r) if len(t) > 3}
            if s in rtok and ((rtok & write_tok) - {s}):
                return True
        return False

    findings: list[str] = []
    seen: set[str] = set()
    for rule in rules:
        # preconditions ("X only if Y") and auto-actions ("when X also do Y") both
        # require a tool to change another subsystem; if that tool is gone, the
        # requirement is unsatisfiable.
        if rule.get("type") not in ("precondition", "auto_action"):
            continue
        if not (resolved_triggers(rule, all_tools) & state_changes):
            continue
        req = (rule.get("requirement") or "").lower()
        req_tokens = tokens(req)
        if not (req_tokens & _CHANGE_VERBS):
            continue  # the requirement is a read-only condition, not an achievable change
        req_norm = {_norm_tok(t) for t in req_tokens}
        for sub in req_norm & subsystem_tokens:
            # fire only for a genuinely uncontrollable subsystem (no direct or
            # indirect write path), never for attribute nouns or subsystems the
            # student can in fact actuate.
            if not _controllable(sub) and sub not in seen:
                seen.add(sub)
                findings.append(
                    f"Policy {rule.get('id')} requires controlling the '{sub}' for this action, but "
                    f"no available tool can change the '{sub}'. You cannot fully comply — do the part "
                    f"you can and tell the user that this specific capability is unavailable. Do not "
                    f"substitute another tool or claim it was done."
                )
    return findings


# --------------------------------------------------------------------------- #
# Tier B/C — grounded Chain-of-Verification teacher
# --------------------------------------------------------------------------- #
def cove_critic(
    draft: dict,
    messages: list[dict],
    tools: Optional[list],
    det_findings: list[str],
    *,
    teacher_model: str,
    record=None,
) -> list[str]:
    """Grounded Chain-of-Verification: pose verification questions, answer each
    from the provided data, and list defects. Seeded with the deterministic
    pre-check findings (treated as confirmed) so the LLM only adds the semantic
    judgements the deterministic gates cannot make (substitution, partial-progress
    / over-refusal, promise-as-done). Returns the merged findings (det + new)."""
    policy = ""
    if messages and messages[0].get("role") == "system":
        policy = messages[0].get("content") or ""
    # Full signatures, not bare names: the teacher must be able to answer
    # parameter-level questions (Q2/Q4). With names only it once *instructed*
    # the student to pass a removed parameter (h_24).
    sigs = []
    for name, schema in sorted(tool_index(tools).items()):
        params = ", ".join(sorted(((schema or {}).get("properties") or {}).keys()))
        sigs.append(f"{name}({params})")
    tool_names = "\n".join(sigs) or "(none)"
    user = TEACHER_USER_TEMPLATE.format(
        policy=policy,
        tool_names=tool_names,
        transcript=_render_transcript(messages),
        data=_data_obtained(messages),
        action=describe_action(draft),
        det_findings=("\n".join(f"- {f}" for f in det_findings) or "(none)"),
    )
    try:
        msg = call_llm(
            [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": user}],
            None,
            model=teacher_model,
            temperature=0.0,
            json_mode=True,
            record=record,
        )
        data = parse_json_object(msg.get("content"))
        for qa in (data.get("questions") or [])[:8]:
            logger.debug("CoVe  Q: %s | A: %s", str(qa.get("q"))[:90], str(qa.get("a"))[:140])
        if data.get("ok") is True:
            return []
        return [str(f).strip() for f in (data.get("findings") or []) if str(f).strip()]
    except Exception as e:  # noqa: BLE001 - fail-safe
        logger.warning("CoVe critic failed (%s); skipping", e)
        return []


def _data_obtained(messages: list[dict], max_items: int = 24) -> str:
    """All grounded data the assistant has this task: tool results + the arguments
    it passed. Used to ground the claim verifier."""
    results, calls = [], []
    for m in messages:
        if m.get("role") == "tool" and m.get("content"):
            results.append(f"{m.get('name','tool')} -> {m['content']}")
        elif m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                calls.append(f"{fn.get('name')}({fn.get('arguments')})")
    out = "TOOL RESULTS RECEIVED:\n" + ("\n".join(results[-max_items:]) or "(none)")
    out += "\n\nTOOL CALLS MADE:\n" + ("\n".join(calls[-max_items:]) or "(none)")
    return out


def check_claims(draft: dict, messages: list[dict], *, teacher_model: str, record=None) -> list[str]:
    """Claim-grounding gate (CoVe): a user reply may state only facts supported by
    the actual tool results / arguments. Catches missing-parameter and
    missing-response fabrications. General — compares reply vs data only."""
    if _calls(draft):
        return []  # an action, not a spoken claim — nothing to ground here
    content = (draft.get("content") or "").strip()
    if not content:
        return []
    user = CLAIM_GROUNDING_TEMPLATE.format(reply=content, data=_data_obtained(messages))
    try:
        msg = call_llm(
            [{"role": "system", "content": CLAIM_GROUNDING_SYSTEM}, {"role": "user", "content": user}],
            None,
            model=teacher_model,
            temperature=0.0,
            json_mode=True,
            record=record,
        )
        logger.debug("claim-grounding raw verdict: %s", str(msg.get("content"))[:500])
        data = parse_json_object(msg.get("content"))
        if data.get("ok") is True:
            return []
        return [str(f).strip() for f in (data.get("findings") or []) if str(f).strip()]
    except Exception as e:  # noqa: BLE001 - fail-safe
        logger.warning("claim-grounding failed (%s); skipping", e)
        return []


def run_verification(draft, messages, tools, rules, prov, cfg, record=None, stage_sink=None,
                     skip_llm=False, relax_gather=False) -> list[str]:
    """Run all enabled teacher checks; return the findings that should trigger a
    revision. Two tiers:

      HARD (deterministic, certain)     — schema, grounding, capability, loops,
                                          completion, policy-required gathers.
                                          Always returned.
      ADVISORY (plausible, needs judge) — policy reminders, preferences nudge,
                                          over-refusal candidates. Passed to the
                                          CoVe teacher, which confirms or discards
                                          them; only what the teacher confirms is
                                          returned. An advisory alone never forces
                                          a revision — that was pure cost/variance
                                          (e.g. a units reminder on a read call).

    `skip_llm=True` (post-revision safety re-check) returns HARD only.
    `relax_gather=True` drops gather findings — used by the orchestrator's
    oscillation valve once the agent has already taken its gather detour.
    If `stage_sink` (a dict) is given, it records stage -> findings for tracing."""
    hard: list[str] = []
    advisories: list[str] = []

    def _stage(name: str, items: list[str], into: list[str]) -> None:
        logger.info("  [%s] %d finding(s)%s", name, len(items),
                    (": " + " | ".join(items)) if items else "")
        if stage_sink is not None:
            stage_sink[name] = items
        into.extend(items)

    try:
        if cfg.enable_halluc_gate:
            _stage("halluc/schema", check_schema(draft, tools), hard)
        if cfg.enable_provenance:
            _stage("grounding", check_grounding(draft, prov), hard)
        if cfg.enable_capability and rules:
            _stage("capability", check_capability(draft, tools, rules), hard)
        if cfg.enable_loop_check:
            _stage("loop-guard", check_loops(draft, messages), hard)
        if getattr(cfg, "enable_confirmation_gate", True):
            _stage("confirm-gate", check_confirmation(draft, messages, tools), hard)
        if cfg.enable_completion_check:
            _stage("completion", check_completion(draft, messages), hard)
        if cfg.enable_claim_grounding and not skip_llm:
            _stage("claim-grounding", check_claims(
                draft, messages, teacher_model=cfg.teacher_model or cfg.model, record=record), hard)
        if cfg.enable_gather_guard:
            if relax_gather:
                logger.info("  [gather-guard] relaxed (oscillation valve): skipping")
                if stage_sink is not None:
                    stage_sink["gather-guard"] = []
            else:
                _stage("gather-guard", check_gather(draft, messages, tools, rules), hard)
                _stage("gather-prefs", gather_prefs_advisory(draft, messages, tools), hard)
        if cfg.enable_disambig:
            _stage("ask-guard", check_ask_guard(draft, messages, tools, rules), hard)
        if getattr(cfg, "enable_refusal_check", True):
            _stage("refusal", check_refusal(draft, messages, tools), advisories)
            # HARD: the evidence is deterministic ('unknown' literally appeared in
            # a tool result + the reply claims success without mentioning it), and
            # the teacher discarded this candidate wrongly twice. Worst case of a
            # false positive is one extra acknowledgment sentence — cheap.
            _stage("unknown-ack", check_unknown_ack(draft, messages), hard)
            _stage("promise-audit", check_promises(draft, messages, tools), advisories)
        if cfg.enable_policy_enforce and rules:
            called = {(tc.get("function") or {}).get("name") for tc in _calls(draft)}
            called.discard(None)
            reply_only = not _calls(draft)
            available = set(tool_index(tools).keys())
            _stage("policy-advisory", policy_mod.advisories_for(
                called, reply_only, rules, available_tools=available,
                context_text=_recent_context(messages),
            ), advisories)

        # Grounded Chain-of-Verification. Seeded with hard findings (objective)
        # plus the advisory candidates; the teacher verifies each candidate
        # against the tools/data and keeps only real defects.
        if skip_llm:
            findings = list(hard)
        elif cfg.enable_verify:
            seed = hard + advisories
            teacher: list[str] = []
            _stage("cove-critic", cove_critic(
                draft, messages, tools, seed,
                teacher_model=cfg.teacher_model or cfg.model, record=record,
            ), teacher)
            findings = hard + teacher
        else:
            # Ablation fallback (no LLM teacher): keep the old behaviour so the
            # deterministic policy layer still functions standalone.
            findings = hard + advisories
    except Exception as e:  # noqa: BLE001 - fail-safe
        logger.warning("Verification error (%s); proceeding with collected findings", e)
        findings = list(hard)

    # De-duplicate, keep order (hard first — severity order is preserved).
    seen: set[str] = set()
    deduped: list[str] = []
    for f in findings:
        if f and f not in seen:
            seen.add(f)
            deduped.append(f)

    # Cap so a weak model is not overloaded by a long, partly-redundant list.
    cap = getattr(cfg, "max_findings", 6)
    capped = deduped[:cap] if cap and len(deduped) > cap else deduped
    logger.info(
        "verification total: %d revision-finding(s)%s (%d hard, %d advisory candidates) for %s",
        len(deduped), (f" (capped to {len(capped)})" if len(capped) < len(deduped) else ""),
        len(hard), len(advisories), describe_action(draft)[:80],
    )
    return capped
