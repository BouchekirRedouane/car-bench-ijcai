#!/usr/bin/env python3
"""Plot CAR-bench Track 1 evaluation results (GCoVe harness runs).

Reads every run JSON in results/raw/ (the files produced by car-bench-run,
copied from output/track_1_agent_under_test/), then writes:

  results/summary.csv          one row per (run, task): reward, end keyword, cost
  results/plots/pass_rate_by_run.png    pass-rate per run, grouped by category
  results/plots/task_matrix.png         task x run pass/fail matrix (green/red)
  results/plots/cost_per_task.png       agent cost per task, colored by model

Usage:
    python results/plot_results.py            # from the repo root
    python plot_results.py                    # or from inside results/

Only needs matplotlib (pip install matplotlib).
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PLOTS = HERE / "plots"

# Filename shape:
# 20260703-113347__track_1_agent_under_test-local_hallucination__train-trials1-hall5ids__openrouter-google-gemini-2.5-flash.json
_NAME_RE = re.compile(
    r"^(?P<ts>\d{8}-\d{6})__.*?-local_(?P<category>[a-z_]+?)__"
    r".*?__(?P<model>.+?)\.json$"
)


def load_runs():
    """-> list of dicts: {ts, category, model, tasks: {task_id: (reward, kw, cost)}}

    Parses the car-bench-run output format:
      {results: [{detailed_results_by_split: {<category>: [task entries]}}]}
    where each task entry has task_id, reward, total_agent_cost, reward_info,
    and error/traceback when the run crashed before the agent ran."""
    runs = []
    for f in sorted(RAW.glob("*.json")):
        m = _NAME_RE.match(f.name)
        if not m:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        tasks = {}
        for trial in (data.get("results") or []) if isinstance(data, dict) else []:
            for split_tasks in (trial.get("detailed_results_by_split") or {}).values():
                for e in split_tasks:
                    tid = e.get("task_id")
                    if not tid:
                        continue
                    ri = (e.get("reward_info") or {})
                    kw = (ri.get("info") or {}).get("end_conversation_keyword")
                    if e.get("error") and not ri:
                        kw = "INFRA_ERROR"  # crashed before the agent ever ran
                    tasks[tid] = (
                        float(e.get("reward", ri.get("reward", 0.0)) or 0.0),
                        kw,
                        float(e.get("total_agent_cost") or 0.0),
                    )
        if tasks:
            runs.append({
                "ts": m.group("ts"),
                "category": m.group("category"),
                "model": m.group("model").replace("openrouter-", ""),
                "file": f.name,
                "tasks": tasks,
            })
    return sorted(runs, key=lambda r: r["ts"])


def write_summary(runs):
    out = HERE / "summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_ts", "category", "model", "task_id", "reward", "end_keyword", "agent_cost_usd"])
        for run in runs:
            for tid, (rew, kw, cost) in sorted(run["tasks"].items()):
                w.writerow([run["ts"], run["category"], run["model"], tid, rew, kw or "", f"{cost:.4f}"])
    print(f"wrote {out}")


_CAT_COLORS = {"base": "#4C72B0", "disambiguation": "#DD8452", "hallucination": "#55A868"}


def plot_pass_rate(runs):
    labels, rates, colors = [], [], []
    for run in runs:
        vals = list(run["tasks"].values())
        rate = sum(1 for v in vals if v[0] == 1.0) / len(vals)
        labels.append(f"{run['ts'][4:8]}-{run['ts'][9:13]}\n{run['category'][:7]}·{run['model'][:14]}")
        rates.append(rate)
        colors.append(_CAT_COLORS.get(run["category"], "#777777"))
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(runs)), 4.5))
    bars = ax.bar(range(len(runs)), rates, color=colors)
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.0%}", ha="center", fontsize=8)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Pass rate (Pass^1)")
    ax.set_title("GCoVe harness — pass rate per run (chronological)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _CAT_COLORS.values()]
    ax.legend(handles, _CAT_COLORS.keys(), loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "pass_rate_by_run.png", dpi=160)
    print(f"wrote {PLOTS/'pass_rate_by_run.png'}")


def plot_task_matrix(runs):
    """task x run matrix: green pass / red fail / grey infra-error."""
    for category in sorted({r["category"] for r in runs}):
        cat_runs = [r for r in runs if r["category"] == category]
        task_ids = sorted({t for r in cat_runs for t in r["tasks"]},
                          key=lambda s: (len(s), s))
        fig, ax = plt.subplots(
            figsize=(max(6, 1.1 * len(cat_runs)), max(3, 0.5 * len(task_ids) + 1.5)))
        for y, tid in enumerate(task_ids):
            for x, run in enumerate(cat_runs):
                if tid not in run["tasks"]:
                    continue
                rew, kw, _ = run["tasks"][tid]
                color = ("#BBBBBB" if kw == "INFRA_ERROR"
                         else "#55A868" if rew == 1.0 else "#C44E52")
                ax.add_patch(plt.Rectangle((x, y), 0.92, 0.92, color=color))
                if kw and kw != "INFRA_ERROR" and rew != 1.0:
                    ax.text(x + 0.46, y + 0.46, kw[:12], ha="center", va="center",
                            fontsize=5.5, color="white")
        ax.set_xlim(0, len(cat_runs)); ax.set_ylim(0, len(task_ids))
        ax.set_xticks([i + 0.46 for i in range(len(cat_runs))])
        ax.set_xticklabels([f"{r['ts'][4:8]}-{r['ts'][9:13]}\n{r['model'][:16]}"
                            for r in cat_runs], fontsize=6.5)
        ax.set_yticks([i + 0.46 for i in range(len(task_ids))])
        ax.set_yticklabels(task_ids, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{category}: task outcomes across runs "
                     f"(green=pass, red=fail w/ end keyword, grey=infra error)")
        fig.tight_layout()
        out = PLOTS / f"task_matrix_{category}.png"
        fig.savefig(out, dpi=160)
        print(f"wrote {out}")


def plot_cost(runs):
    by_model = defaultdict(list)
    for run in runs:
        for tid, (_, kw, cost) in run["tasks"].items():
            if cost > 0:
                by_model[run["model"]].append(cost)
    fig, ax = plt.subplots(figsize=(7, 4))
    models = sorted(by_model)
    ax.boxplot([by_model[m] for m in models], tick_labels=[m[:24] for m in models])
    ax.set_ylabel("agent cost per task (USD)")
    ax.set_title("Per-task agent cost by model")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "cost_per_task.png", dpi=160)
    print(f"wrote {PLOTS/'cost_per_task.png'}")


if __name__ == "__main__":
    PLOTS.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no parseable run files in {RAW}")
    print(f"loaded {len(runs)} runs")
    write_summary(runs)
    plot_pass_rate(runs)
    plot_task_matrix(runs)
    plot_cost(runs)
    print("done — see results/plots/")
