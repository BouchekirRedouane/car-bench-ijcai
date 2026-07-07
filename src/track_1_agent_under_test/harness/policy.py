"""Runtime policy compiler — turns whatever policy wiki the evaluator provides
into a typed, machine-checkable rule set, then matches it against proposed
actions to produce *advisories* (reminders fed to the student during the CoVe
revise step).

Dynamic by construction: it compiles the *provided* text, so it scales to
policies never seen before. Compiled once per distinct policy (cached by a hash
of the policy with volatile location/time normalized out), so it adds no
per-task variance.
"""
from __future__ import annotations

import difflib
import hashlib
import logging
import re
from typing import Optional

from .llm import call_llm, parse_json_object
from .prompts import POLICY_COMPILER_SYSTEM

logger = logging.getLogger("harness.policy")


def tokens(s: str) -> set[str]:
    """Split a tool/trigger name into lowercase word tokens."""
    return {t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if t}


# Pure connectors only — keep action verbs (get/set/open/close), since they are
# exactly what tells a read tool apart from a write tool with the same subject.
_TRIGGER_STOP = {"the", "a", "an", "to", "of", "and", "or"}


def match_tool(trigger: str, tool_names) -> Optional[str]:
    """Resolve an approximate trigger (e.g. 'open_sunroof') to the closest real
    tool name (e.g. 'open_close_sunroof'). Returns None if nothing is close.
    Deterministic (iterates sorted names) and fully generic — no hardcoded names."""
    if not trigger:
        return None
    t = trigger.strip().lower()
    names = sorted(tool_names)
    if t in set(names):
        return t
    tt = tokens(t) - _TRIGGER_STOP or tokens(t)
    best, best_score = None, 0.0
    for name in names:
        nt = tokens(name)
        cover = len(tt & nt) / len(tt) if tt else 0.0
        ratio = difflib.SequenceMatcher(None, t, name.lower()).ratio()
        score = max(cover, ratio)
        if score > best_score:
            best, best_score = name, score
    return best if best_score >= 0.6 else None


def resolved_triggers(rule: dict, tool_names) -> set[str]:
    """The set of real available tool names a rule's triggers map onto."""
    out: set[str] = set()
    for tr in rule.get("trigger_tools") or []:
        m = match_tool(tr, tool_names)
        if m:
            out.add(m)
    return out

# Process-wide cache: normalized-policy-hash -> list[rule dict].
_CACHE: dict[str, list[dict]] = {}

_VOLATILE_RE = [
    re.compile(r"CURRENT_LOCATION\s*=\s*\{.*?\}", re.DOTALL),
    re.compile(r"DATETIME\s*=\s*\{.*?\}", re.DOTALL),
]

_VALID_TYPES = {
    "precondition",
    "auto_action",
    "confirmation",
    "prohibition",
    "disclosure",
    "constraint",
    "selection",
}


def _normalize(policy_text: str) -> str:
    t = policy_text or ""
    for rx in _VOLATILE_RE:
        t = rx.sub("", t)
    return t.strip()


def compile_policy(policy_text: str, *, model: str, tool_names=None, record=None, tools=None) -> list[dict]:
    """Compile the policy into rules (cached). Never raises — returns [] on failure.
    `tool_names` (the available tools) are given to the compiler so it can ground
    trigger_tools to real names."""
    norm = _normalize(policy_text)
    if not norm:
        return []
    key = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    if key in _CACHE:
        return _CACHE[key]

    user_content = norm
    if tool_names:
        user_content = norm + "\n\nAVAILABLE TOOL NAMES:\n" + ", ".join(sorted(tool_names))

    rules: list[dict] = []
    try:
        data: dict = {}
        for attempt in (1, 2):  # one bounded re-ask on unparseable compiler JSON —
            msg = call_llm(       # losing ALL rules for the whole task is too costly
                [
                    {"role": "system", "content": POLICY_COMPILER_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                None,
                model=model,
                temperature=0.0,
                json_mode=True,
                record=record,
            )
            data = parse_json_object(msg.get("content"))
            if data:
                break
            logger.warning("Policy compile returned unparseable JSON (attempt %d)", attempt)
        for r in data.get("rules", []) or []:
            if not isinstance(r, dict):
                continue
            rtype = str(r.get("type", "")).strip().lower()
            if rtype not in _VALID_TYPES:
                continue
            triggers = r.get("trigger_tools") or []
            if not isinstance(triggers, list):
                triggers = []
            compiled = {
                "id": str(r.get("id", "")).strip() or rtype,
                "type": rtype,
                "trigger_tools": [str(t).strip() for t in triggers if str(t).strip()],
                "requirement": str(r.get("requirement", "")).strip(),
            }
            rules.append(compiled)
        logger.info("Compiled %d policy rules", len(rules))
        for r in rules:
            logger.info(
                "  rule %s/%s triggers=%s :: %s",
                r["id"], r["type"], r["trigger_tools"] or "-", r["requirement"][:120],
            )
    except Exception as e:  # noqa: BLE001 - fail-safe
        logger.warning("Policy compile failed (%s); continuing with no rules", e)
        rules = []

    # v5 second pass: translate state-conditional rules into the executable
    # form (dedicated focused prompt — folding this into the main compile
    # measurably degraded rule extraction). Fail-safe: no execs on any error.
    try:
        _attach_execs(rules, tool_names, tools, model=model, record=record)
    except Exception as e:  # noqa: BLE001
        logger.warning("exec pass failed (%s); rules stay advisory-only", e)

    _CACHE[key] = rules
    return rules


def _attach_execs(rules: list[dict], tool_names, tools=None, *, model: str, record=None) -> None:
    from .prompts import POLICY_EXEC_SYSTEM  # late import (avoids cycle at module load)

    cand = [r for r in rules
            if r.get("type") in ("auto_action", "precondition") and r.get("requirement")]
    if not cand or not tool_names:
        return
    lines = [f'{r["id"]} ({r["type"]}): {r["requirement"]}' for r in cand]
    # REAL signatures (name + parameter names): without the parameters the exec
    # compiler must guess obligation arg names, and wrong guesses are silently
    # dropped at evaluation -> an inert engine that looks like model failure.
    sigs = []
    for tl in tools or []:
        fn = (tl or {}).get("function") or {}
        if fn.get("name"):
            params = ", ".join(sorted(((fn.get("parameters") or {}).get("properties") or {}).keys()))
            sigs.append(f"{fn['name']}({params})")
    if not sigs:
        sigs = sorted(tool_names)
    user = "RULES:\n" + "\n".join(lines) + "\n\nTOOL SIGNATURES:\n" + "\n".join(sorted(sigs))
    data: dict = {}
    for attempt in (1, 2):
        msg = call_llm(
            [{"role": "system", "content": POLICY_EXEC_SYSTEM},
             {"role": "user", "content": user}],
            None, model=model, temperature=0.0, json_mode=True, record=record,
        )
        data = parse_json_object(msg.get("content"))
        if data:
            break
    execs = data.get("execs") or {}
    n = 0
    for r in rules:
        ex = execs.get(r["id"])
        if (
            isinstance(ex, dict)
            and isinstance(ex.get("read"), str)
            and isinstance(ex.get("field_pattern"), str)
            and ex.get("op") in (">", ">=", "<", "<=", "==", "!=", "contains", "not_contains")
            and isinstance(ex.get("obligation"), dict)
        ):
            r["exec"] = ex
            n += 1
            logger.info("  exec[%s]: %s", r["id"], str(ex)[:140])
    logger.info("exec pass: %d/%d state-conditional rules executable", n, len(cand))


_STOP_WORDS = set(
    "the a an and or to of if then you your for is are be must should always only this that with "
    "before after not no any all it its can cannot will would when which what who whom on in at by "
    "as into per each they them their there here so do does done has have had use used using".split()
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 3 and w not in _STOP_WORDS}


def advisories_for(
    called_tools: set[str],
    reply_only: bool,
    rules: list[dict],
    available_tools=None,
    context_text: str = "",
) -> list[str]:
    """Reminders for rules that actually apply this turn.

    * Tool-triggered rules fire when a (fuzzy-resolved) trigger tool is being called.
    * Rules with no specific tool (reply-level) fire only if a meaningful word from
      the rule's requirement appears in the recent context — this keeps irrelevant
      disclosures (e.g. toll roads on a sunroof task) from adding noise.
    """
    available_tools = set(available_tools or [])
    ctx_words = _content_words(context_text)
    out: list[str] = []
    for rule in rules:
        req = rule.get("requirement")
        if not req:
            continue
        triggers = resolved_triggers(rule, available_tools) if available_tools else set(rule.get("trigger_tools") or [])
        if triggers:
            applies = bool(triggers & called_tools)
        elif reply_only:
            # reply-level rule: only when its subject is contextually present
            applies = bool(_content_words(req) & ctx_words)
        else:
            applies = False
        if applies:
            out.append(f"POLICY[{rule['id']}/{rule['type']}]: {req}")
    return out
