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
}

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
    out: dict[str, dict] = {}
    for m in messages:
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
                out[name] = body
    return out


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
    results = latest_results(messages)
    all_tools = { (t.get("function") or {}).get("name") for t in tools or [] } - {None}
    obligations: list[dict] = []
    missing_reads: set[str] = set()
    unknown_fields: list[str] = []
    total_matched: dict[str, int] = {}
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
        body = results.get(read)
        if body is None:
            if read in all_tools:
                missing_reads.add(read)
            continue
        schema_props = (_tool_schema(tools, ob_tool).get("properties") or {})
        matched = 0
        for field, value in body.items():
            item = _wildcard_capture(ex["field_pattern"], field)
            if item is None:
                continue
            matched += 1
            if isinstance(value, str) and value.strip().lower() in _UNKNOWN:
                unknown_fields.append(field)
                continue
            a, b = value, ex["value"]
            na, nb = _numeric(a), _numeric(b)
            try:
                hit = _OPS[ex["op"]](na, nb) if na is not None and nb is not None \
                    else _OPS[ex["op"]](str(a).lower(), str(b).lower())
            except Exception:
                continue
            if not hit:
                continue
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
            "unknown_fields": unknown_fields, "total_matched": total_matched}


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

    findings: list[str] = []
    if ev["missing_reads"]:
        findings.append(TUN["finding.obligation_read"].format(
            reads=", ".join(sorted(ev["missing_reads"]))))

    # Which computed obligations are not covered by the draft?
    def covered(ob) -> bool:
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
    return findings
