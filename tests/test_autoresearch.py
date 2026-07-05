"""Tests for the autoresearch tunables layer + acceptance audit.

Run: .venv/bin/python tests/test_autoresearch.py   (or pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "track_1_agent_under_test"))
sys.path.insert(0, str(REPO / "autoresearch"))

from harness import tunables
import audit as audit_mod


def _tmp_candidate(payload: dict) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(payload, f)
    f.close()
    return Path(f.name)


def test_defaults_load_without_env():
    tun = tunables.load(path=None)
    assert tun == tunables.DEFAULTS
    assert "directive.7" in tun and "finding.refusal" in tun


def test_override_applies_and_metadata_ignored():
    p = _tmp_candidate({"_name": "x", "directive.5": "SPEAK FOR VOICE. Short sentences only."})
    tun = tunables.load(path=str(p))
    assert tun["directive.5"] == "SPEAK FOR VOICE. Short sentences only."
    assert tun["directive.4"] == tunables.DEFAULTS["directive.4"]  # untouched


def test_unknown_key_fails_loudly():
    p = _tmp_candidate({"directive.99": "nope"})
    try:
        tunables.load(path=str(p))
    except ValueError as e:
        assert "unknown tunable key" in str(e)
    else:
        raise AssertionError("unknown key must raise")


def test_audit_rejects_task_ids_and_tool_names():
    p = _tmp_candidate({"directive.2": "NEVER FABRICATE. Remember hallucination_48 needs care."})
    problems = audit_mod.audit(p)
    assert any("task id" in x for x in problems), problems
    # tool-name rejection (only if the vendored benchmark is present)
    if audit_mod.benchmark_tool_names():
        p2 = _tmp_candidate({"directive.2": "NEVER FABRICATE. Prefer open_close_sunroof always."})
        problems2 = audit_mod.audit(p2)
        assert any("tool names" in x for x in problems2), problems2


def test_audit_rejects_too_many_changes_and_lost_placeholders():
    many = {f"directive.{i}": f"changed {i}." for i in range(1, 6)}
    assert any("max" in x for x in audit_mod.audit(_tmp_candidate(many)))
    lost = {"finding.confirmation": "Ask for confirmation first."}  # lost {name}
    assert any("placeholders" in x for x in audit_mod.audit(_tmp_candidate(lost)))


def test_audit_accepts_clean_generic_rewording():
    p = _tmp_candidate({
        "_hypothesis": "shorter directive improves compliance",
        "directive.5": "SPEAK FOR VOICE. Plain short spoken sentences; never lists, markdown or emoji.",
    })
    assert audit_mod.audit(p) == []


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
