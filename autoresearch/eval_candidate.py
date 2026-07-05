#!/usr/bin/env python3
"""Evaluate one tunables candidate on the tune subset.

Runs autoresearch/audit.py first (hard requirement), then car-bench-run on
autoresearch/tune_set.toml with HARNESS_TUNABLES pointing at the candidate,
parses the newest output file, and writes a scored result JSON.

Score = mean pass rate over trials  −  0.5 * flip rate
(a task that passes one trial and fails the other counts against the
candidate: Pass^3 punishes variance as hard as failure).

Usage:
    python autoresearch/eval_candidate.py autoresearch/candidates/<name>.json
    python autoresearch/eval_candidate.py --baseline        # empty override
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "output" / "track_1_agent_under_test"
RES_DIR = Path(__file__).resolve().parent / "results"

FLIP_PENALTY = 0.5


def newest_output(after: float) -> Path | None:
    cands = [p for p in OUT_DIR.glob("*.json") if p.stat().st_mtime >= after]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def parse_run(path: Path) -> dict:
    """-> {task_id: [reward per trial, ...]}"""
    data = json.loads(path.read_text())
    per_task: dict[str, list[float]] = defaultdict(list)
    for trial in data.get("results") or []:
        for split_tasks in (trial.get("detailed_results_by_split") or {}).values():
            for e in split_tasks:
                tid = e.get("task_id")
                if tid:
                    per_task[tid].append(float(e.get("reward") or 0.0))
    return dict(per_task)


def score(per_task: dict) -> dict:
    n = len(per_task)
    mean_pass = sum(sum(v) / len(v) for v in per_task.values()) / n if n else 0.0
    flips = sum(1 for v in per_task.values() if len(set(v)) > 1)
    flip_rate = flips / n if n else 0.0
    return {
        "tasks": n,
        "mean_pass": round(mean_pass, 4),
        "flips": flips,
        "flip_rate": round(flip_rate, 4),
        "score": round(mean_pass - FLIP_PENALTY * flip_rate, 4),
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--baseline":
        cand_path = Path(__file__).resolve().parent / "candidates" / "baseline.json"
        cand_path.write_text("{}\n")
    else:
        cand_path = Path(sys.argv[1])
    name = cand_path.stem

    # 1) audit (includes the test suite with the candidate loaded)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "audit.py"), str(cand_path)])
    if r.returncode != 0:
        raise SystemExit(f"candidate {name}: REJECTED by audit")

    # 2) run the tune subset with the candidate active in the agent process
    env = {**os.environ, "HARNESS_TUNABLES": str(cand_path.resolve())}
    t0 = time.time()
    print(f"evaluating candidate '{name}' on the tune subset (2 trials)…")
    r = subprocess.run(
        ["uv", "run", "car-bench-run", "autoresearch/tune_set.toml"],
        cwd=REPO, env=env,
    )
    if r.returncode != 0:
        raise SystemExit(f"car-bench-run failed (exit {r.returncode})")

    out = newest_output(after=t0)
    if not out:
        raise SystemExit("no output file produced")
    per_task = parse_run(out)
    result = {
        "candidate": name,
        "candidate_file": str(cand_path),
        "run_output": out.name,
        **score(per_task),
        "per_task": {k: v for k, v in sorted(per_task.items())},
    }
    RES_DIR.mkdir(exist_ok=True)
    res_path = RES_DIR / f"{name}.json"
    res_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2))
    print(f"wrote {res_path}")


if __name__ == "__main__":
    main()
