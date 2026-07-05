#!/usr/bin/env python3
"""Candidate acceptance audit — the anti-overfitting firewall.

A tunables candidate may only be evaluated if it passes ALL of:
  1. schema: only known tunable keys, string values, <= MAX_CHANGES changes;
  2. literal audit: no benchmark task ids and no benchmark tool names inside
     any override string (tool names are harvested dynamically from the
     vendored benchmark, so the audit itself contains no hardcoded names);
  3. the full deterministic gate test suite (incl. alien-domain scalability
     tests) passes with the candidate loaded.

Usage:
    python autoresearch/audit.py autoresearch/candidates/<name>.json
Exit code 0 = accepted for evaluation, 1 = rejected (reasons on stdout).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "track_1_agent_under_test"))
from harness.tunables import DEFAULTS  # noqa: E402

MAX_CHANGES = 3
_TASK_ID_RE = re.compile(r"\b(base|hallucination|disambiguation)_\d+\b", re.I)


def benchmark_tool_names() -> set[str]:
    """Harvest the vendored benchmark's tool names from its tool modules —
    nothing hardcoded here, so the audit works for any refreshed benchmark."""
    base = REPO / "third_party" / "car-bench" / "car_bench" / "envs" / "car_voice_assistant" / "tools"
    if not base.exists():
        print("WARN: vendored benchmark not found; tool-name audit skipped")
        return set()
    names = {p.stem for p in base.rglob("*.py")} - {"__init__", "base", "registry"}
    # generic utilities may legitimately be referenced in prompts
    return names - {"think", "planning_tool", "note_intermediate_result"}


def audit(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        cand = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return [f"unreadable JSON: {e}"]
    if not isinstance(cand, dict):
        return ["top level must be a JSON object"]

    overrides = {k: v for k, v in cand.items() if not k.startswith("_")}
    unknown = [k for k in overrides if k not in DEFAULTS]
    if unknown:
        problems.append(f"unknown tunable keys: {unknown}")
    nonstr = [k for k, v in overrides.items() if not isinstance(v, str)]
    if nonstr:
        problems.append(f"non-string values: {nonstr}")
    changed = [k for k, v in overrides.items()
               if k in DEFAULTS and isinstance(v, str) and v != DEFAULTS[k]]
    if len(changed) > MAX_CHANGES:
        problems.append(f"{len(changed)} keys changed; max {MAX_CHANGES} per candidate "
                        f"(keeps improvements attributable)")

    # placeholder preservation: a template must keep the placeholders its
    # default uses, or .format() will crash at runtime
    for k in changed:
        needed = set(re.findall(r"{(\w+)}", DEFAULTS[k]))
        have = set(re.findall(r"{(\w+)}", overrides[k]))
        if not needed <= have:
            problems.append(f"'{k}' lost required placeholders: {sorted(needed - have)}")

    # literal audit
    tool_names = benchmark_tool_names()
    for k in changed:
        text = overrides[k]
        if _TASK_ID_RE.search(text):
            problems.append(f"'{k}' contains a benchmark task id — hardcoding is forbidden")
        low = text.lower()
        hits = sorted(n for n in tool_names if n in low)
        # get_user_preferences appears in the default directives already; allow
        # names that the DEFAULT text for this key also contains
        hits = [h for h in hits if h not in DEFAULTS[k].lower()]
        if hits:
            problems.append(f"'{k}' contains benchmark tool names {hits} — hardcoding is forbidden")
    return problems


def run_tests_with_candidate(path: Path) -> bool:
    env = {"HARNESS_TUNABLES": str(path)}
    import os
    full_env = {**os.environ, **env}
    r = subprocess.run(
        [sys.executable, str(REPO / "tests" / "test_harness_gates.py")],
        env=full_env, capture_output=True, text=True, timeout=300,
    )
    ok = "ALL TESTS PASS" in r.stdout
    if not ok:
        print(r.stdout[-1500:])
    return ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    problems = audit(path)
    for pr in problems:
        print("REJECT:", pr)
    if problems:
        raise SystemExit(1)
    print("schema + literal audit: OK")
    if not run_tests_with_candidate(path):
        print("REJECT: gate test suite failed with candidate loaded")
        raise SystemExit(1)
    print("test suite with candidate: OK")
    print("ACCEPTED for evaluation")
