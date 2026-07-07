"""v5 Obligation engine — "compute before, execute after".

The policy compiler (an LLM, at runtime) now emits an optional EXECUTABLE form
for state-conditional rules:

    "exec": {
        "read": "get_vehicle_window_positions",      # tool whose result holds the state
        "field_pattern": "window_*_position",        # glob over result fields
        "op": ">",                                   # >, >=, <, <=, ==, !=
        "value": 20,                                 # threshold (number or string)
        "obligation": {                              # what to do per qualifying item
            "tool": "open_close_window",
            "args": {"window": "<item>", "percentage": 0}
        }
    }

This module is a pure INTERPRETER of that structure: it flattens the latest
tool results, matches fields, compares values, and emits concrete obligations
with computed arguments ("close DRIVER and PASSENGER, they are 25% and 100%").
The language→tools binding was done by the LLM compiler from the evaluator's
own policy text, so nothing here is benchmark-specific: a hidden policy like
"close any oven door that is open" compiles and evaluates identically
(covered by the alien-domain tests).

Fail-safe by design: no valid exec block, missing read result, unknown values,
or an argument that cannot be validated against the tool schema -> the rule
silently degrades to v4 behaviour (advisory + teacher). The engine can only
ADD precision, never crash a turn or force an invalid call.
"""
from __future__ import annotations

import json
import logging
from fnmatch import fnmatch
from typing import Optional

from .policy import resolved_triggers
from .tunables import TUN

logger = logging.getLogger("harness.obligations")

_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    # string-membership conditions ("if the direction does not include X"):
    "contains": lambda a, b: str(b).lower() in str(a).lower(),
    "not_contains": lambda a, b: str(b).lower() not in str(a).lower(),
}
_STRING_OPS = {"contains", "not_contains"}

_UNKNOWN = {"unknown", "n/a", "none", "null", ""}


def valid_exec(ex) -> bool:
    """Structural validation of a compiled exec block (schema-level checks of
    the obligation tool/args happen at evaluation time against the live tools)."""
    if not isinstance(ex, dict):
        return False
    ob = ex.get("obligation")
    return (
        isinstance(ex.get("read"), str) and ex["read"]
        and isinstance(ex.get("field_pattern"), str) and ex["field_pattern"]
        and ex.get("op") in _OPS
        and isinstance(ex.get("value"), (int, float, str))
        and isinstance(ob, dict)
        and isinstance(ob.get("tool"), str) and ob["tool"]
        and isinstance(ob.get("args"), dict)
    )


# --------------------------------------------------------------------------- #
# Result parsing
# --------------------------------------------------------------------------- #
def latest_results(messages: list[dict]) -> dict[str, dict]:
    """Latest parsed result per tool name from this task's history."""
    return {k: v for k, (v, _i) in latest_results_indexed(messages).items()}


def latest_results_indexed(messages: list[dict]) -> dict[str, tuple]:
    """Latest parsed result per tool name, with its message index (for the
    staleness check against later writes)."""
    out: dict[str, tuple] = {}
    for i, m in enumerate(messages):
        if m.get("role") != "tool" or not m.get("content"):
            continue
        name = m.get("name") or ""
        try:
            data = json.loads(m["content"])
        except Exception:
            continue
        if isinstance(data, dict):
            body = data.get("result") if isinstance(data.get("result"), dict) else data
            if isinstance(body, dict):
                out[name] = (body, i)
    return out


_READ_PREFIXES = ("get_", "search_", "list_", "find_", "lookup_", "retrieve_",
                  "calculate_", "compute_", "check_", "read_")
_READ_EXACT = {"planning_tool", "think", "note_intermediate_result", "datetime", "math"}
_ATTRS = {"status", "state", "position", "level", "setting", "settings", "current",
          "info", "information", "vehicle"}


def _is_read_name(name: str) -> bool:
    return name in _READ_EXACT or name.startswith(_READ_PREFIXES)


def _subject_tokens(name: str) -> set:
    import re as _re
    toks = {x for x in _re.split(r"[^a-z0-9]+", (name or "").lower()) if len(x) > 3}
    return {x[:-1] if len(x) > 4 and x.endswith("s") else x for x in toks} - _ATTRS


def _stale(read: str, read_idx: int, messages: list[dict]) -> bool:
    """A read result is STALE when a write sharing its subject executed AFTER
    it (observed live: closed ALL windows, then the engine computed
    obligations from the pre-close positions and demanded redundant closes)."""
    subj = _subject_tokens(read)
    if not subj:
        return False
    for m in messages[read_idx + 1:]:
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            n = (tc.get("function") or {}).get("name") or ""
            if n and not _is_read_name(n) and (_subject_tokens(n) & subj):
                return True
    return False


def _wildcard_capture(pattern: str, field: str) -> Optional[str]:
    """Return the substring matched by the single '*' in `pattern`, or None."""
    if "*" not in pattern:
        return "" if pattern == field else None
    prefix, _, suffix = pattern.partition("*")
    if field.startswith(prefix) and field.endswith(suffix) and len(field) >= len(prefix) + len(suffix):
        return field[len(prefix): len(field) - len(suffix) if suffix else len(field)]
    return None


def _numeric(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tool_schema(tools, name: str) -> dict:
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        if fn.get("name") == name:
            return fn.get("parameters") or {}
    return {}


def _bind_item(template_value, item: str, schema_prop: dict):
    """Fill '<item>' templates: the wildcard capture ('driver_rear') becomes the
    argument value, matched case-insensitively against the schema enum when one
    is declared ('DRIVER_REAR'). Returns None when binding is impossible."""
    if template_value != "<item>":
        return template_value
    enum = (schema_prop or {}).get("enum")
    if enum:
        for e in enum:
            if str(e).lower() == item.lower():
                return e
        return None  # cannot validate -> drop this obligation (fallback to v4)
    return item.upper()


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(rules, tools, messages, triggered_by: set) -> dict:
    """Evaluate every valid exec rule triggered by the proposed writes.

    Returns {obligations, missing_reads, unknown_fields, total_matched}:
      obligations   : [{tool, args, reason, rule}] with fully computed args
      missing_reads : reads the conditions need but were never called
      unknown_fields: state fields whose value is unknown/unavailable
      total_matched : pattern-matched field count per rule id (for ALL-scope checks)
    """
    results_idx = latest_results_indexed(messages)
    all_tools = { (t.get("function") or {}).get("name") for t in tools or [] } - {None}
    obligations: list[dict] = []
    missing_reads: set[str] = set()
    unknown_fields: list[str] = []
    total_matched: dict[str, int] = {}
    # per obligation-tool: which arg names the item, and the aligned item
    # values that qualified / matched / were unknown — used to flag EXTRA
    # writes on items the policy says to leave alone.
    items: dict[str, dict] = {}
    seen: set[str] = set()

    for rule in rules or []:
        ex = rule.get("exec")
        if not valid_exec(ex):
            continue
        # normalize names: the exec compiler sees `name(params)` signatures and
        # sometimes copies the parentheses into the names (observed live:
        # 'get_vehicle_window_positions()') — strip anything from '(' on.
        ex = dict(ex, read=ex["read"].split("(")[0].strip(),
                  obligation=dict(ex["obligation"],
                                  tool=ex["obligation"]["tool"].split("(")[0].strip()))
        if not (resolved_triggers(rule, all_tools) & triggered_by):
            continue
        ob_tool = ex["obligation"]["tool"]
        if ob_tool not in all_tools:
            continue  # obligation tool unavailable -> capability/teacher territory
        read = ex["read"]
        entry = results_idx.get(read)
        if entry is None:
            if read in all_tools:
                missing_reads.add(read)
            continue
        body, read_idx = entry
        if _stale(read, read_idx, messages):
            missing_reads.add(read)  # state changed since the read: demand a fresh one
            continue
        schema_props = (_tool_schema(tools, ob_tool).get("properties") or {})
        matched = 0
        for field, value in body.items():
            item = _wildcard_capture(ex["field_pattern"], field)
            if item is None:
                continue
            matched += 1
            item_key0 = next((k for k, v in ex["obligation"]["args"].items()
                              if isinstance(v, str) and "<item>" in v), None)
            rec0 = items.setdefault(ob_tool, {"item_key": item_key0, "qualifying": set(),
                                              "matched": set(), "unknown": set()})
            if item_key0:
                aligned0 = _bind_item(ex["obligation"]["args"][item_key0], item,
                                      ((_tool_schema(tools, ob_tool).get("properties") or {}).get(item_key0) or {}))
                if aligned0 is not None:
                    rec0["matched"].add(str(aligned0))
                    if isinstance(value, str) and value.strip().lower() in _UNKNOWN:
                        rec0["unknown"].add(str(aligned0))
            if isinstance(value, str) and value.strip().lower() in _UNKNOWN:
                unknown_fields.append(field)
                continue
            a, b = value, ex["value"]
            if ex["op"] in _STRING_OPS:
                try:
                    hit = _OPS[ex["op"]](a, b)
                except Exception:
                    continue
            else:
                na, nb = _numeric(a), _numeric(b)
                try:
                    hit = _OPS[ex["op"]](na, nb) if na is not None and nb is not None \
                        else _OPS[ex["op"]](str(a).lower(), str(b).lower())
                except Exception:
                    continue
            if not hit:
                continue
            item_key = next((k for k, v in ex["obligation"]["args"].items()
                             if isinstance(v, str) and "<item>" in v), None)
            if item_key:
                rec = items.setdefault(ob_tool, {"item_key": item_key, "qualifying": set(),
                                                 "matched": set(), "unknown": set()})
                aligned = _bind_item(ex["obligation"]["args"][item_key], item,
                                     (schema_props.get(item_key) or {}))
                if aligned is not None:
                    rec["qualifying"].add(str(aligned))
            args = {}
            ok = True
            for k, v in ex["obligation"]["args"].items():
                if k not in schema_props and schema_props:
                    ok = False
                    break
                bound = _bind_item(v, item, schema_props.get(k) or {})
                if bound is None:
                    ok = False
                    break
                args[k] = bound
            if not ok:
                continue
            key = ob_tool + json.dumps(args, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            obligations.append({
                "tool": ob_tool, "args": args,
                "reason": f"{field}={value} {ex['op']} {ex['value']}",
                "rule": rule.get("id", "?"),
            })
        total_matched[rule.get("id", "?")] = matched
    return {"obligations": obligations, "missing_reads": missing_reads,
            "unknown_fields": unknown_fields, "total_matched": total_matched,
            "items": items}


def _render(ob: dict) -> str:
    return f"{ob['tool']}({json.dumps(ob['args'], ensure_ascii=False)})"


def check_obligations(draft: dict, messages: list[dict], tools, rules) -> list[str]:
    """Hard gate: compare the draft's write calls against the COMPUTED
    obligations. Emits exact, ready-to-execute corrections."""
    calls = draft.get("tool_calls") or []
    draft_calls = []
    triggered: set = set()
    for tc in calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments")
        try:
            args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
        except Exception:
            args = {}
        draft_calls.append((name, args))
        triggered.add(name)
    if not triggered:
        return []

    try:
        ev = evaluate(rules, tools, messages, triggered)
    except Exception as e:  # noqa: BLE001 - the engine must never crash a turn
        logger.warning("obligation engine failed (%s); skipping", e)
        return []

    # obligations already EXECUTED earlier in the task are covered — otherwise
    # the missing-finding pushes the student to re-send old writes (observed
    # live: AC + temperature executed twice)
    executed: set = set()
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc0 in (m.get("tool_calls") or []):
            fn0 = tc0.get("function") or {}
            raw0 = fn0.get("arguments")
            try:
                a0 = raw0 if isinstance(raw0, dict) else json.loads(raw0 or "{}")
            except Exception:
                a0 = {}
            executed.add((fn0.get("name"), json.dumps(a0, sort_keys=True)))

    findings: list[str] = []
    if ev["missing_reads"]:
        findings.append(TUN["finding.obligation_read"].format(
            reads=", ".join(sorted(ev["missing_reads"]))))

    # Which computed obligations are not covered by the draft?
    def covered(ob) -> bool:
        if (ob["tool"], json.dumps(ob["args"], sort_keys=True)) in executed:
            return True
        for name, args in draft_calls:
            if name != ob["tool"]:
                continue
            if all(args.get(k) == v for k, v in ob["args"].items()):
                return True
            # aggregate form: an 'ALL'-style argument covers every item only if
            # every matched item qualified — checked below via total_matched
            if any(isinstance(v, str) and v.upper() == "ALL" for v in args.values()):
                total = ev["total_matched"].get(ob["rule"], 0)
                per_rule = [o for o in ev["obligations"] if o["rule"] == ob["rule"]]
                if total and len(per_rule) == total:
                    return True
        return False

    missing = [ob for ob in ev["obligations"] if not covered(ob)]
    if missing:
        findings.append(TUN["finding.obligation_missing"].format(
            calls="; ".join(_render(ob) for ob in missing[:6]),
            reasons="; ".join(f"{ob['tool']}: {ob['reason']} (rule {ob['rule']})"
                              for ob in missing[:6])))

    # ALL used although only a subset qualifies -> exact per-item replacement
    for name, args in draft_calls:
        if not any(isinstance(v, str) and v.upper() == "ALL" for v in args.values()):
            continue
        per_tool = [ob for ob in ev["obligations"] if ob["tool"] == name]
        if not per_tool:
            continue
        rule_id = per_tool[0]["rule"]
        total = ev["total_matched"].get(rule_id, 0)
        if total and len(per_tool) < total:
            findings.append(TUN["finding.obligation_scope"].format(
                tool=name,
                calls="; ".join(_render(ob) for ob in per_tool[:6])))

    # INVERSE scope: writes on items the policy says to LEAVE ALONE (live
    # base_94 failure: closed the 10%-open window under a >20% rule). Skipped
    # when the user explicitly mentioned the tool's subject — an explicit
    # request overrides the rule-derived scope.
    user_text = " ".join(str(m.get("content") or "")
                         for m in messages if m.get("role") == "user").lower()
    ob_tools = {ob["tool"] for ob in ev["obligations"]}
    for name, args in draft_calls:
        rec = ev["items"].get(name)
        if rec is None:
            continue
        if any(w in user_text for w in _subject_tokens(name)):
            continue  # user explicitly talked about this subject
        if rec.get("item_key"):
            val = args.get(rec["item_key"])
            sval = str(val)
            if val is None or (isinstance(val, str) and val.strip().upper() == "ALL"):
                continue  # aggregate handled by the scope check above
            if sval in rec["matched"] and sval not in rec["qualifying"]                     and sval not in rec["unknown"]:
                findings.append(TUN["finding.obligation_extra"].format(
                    tool=name, item=sval))
        else:
            # constant-args obligation (e.g. airflow): the rule matched state
            # fields but the condition is NOT met -> the call is unnecessary
            if rec["matched"] == set() and name not in ob_tools                     and ev["total_matched"].get(_rule_of(ev, name), 1) is not None:
                pass  # nothing recorded; fall through
            if name not in ob_tools and _tool_had_matches(ev, name):
                findings.append(TUN["finding.obligation_unneeded"].format(tool=name))
    return findings


def _rule_of(ev, tool_name):
    for ob in ev["obligations"]:
        if ob["tool"] == tool_name:
            return ob["rule"]
    return None


def _tool_had_matches(ev, tool_name) -> bool:
    """True when an exec rule targeting this tool evaluated at least one field
    with a KNOWN value (so 'no obligation' really means 'condition not met')."""
    rec = ev["items"].get(tool_name)
    if rec is None:
        return False
    if rec.get("item_key"):
        return bool(rec["matched"] - rec["unknown"])
    # constant-args exec: rely on total_matched of any rule that produced the
    # record — recorded matches imply the read succeeded with known values
    return True
