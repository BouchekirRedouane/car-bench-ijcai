# Benchmark run logs — full per-task conversation chains

Each `*.json` file is one complete benchmark run (all LLM roles on
`gemini-2.5-flash` unless the filename says otherwise). File naming:
`<start-timestamp>__<scenario>__<split+trials>__<model>.json`.

## Included runs

| File (timestamp) | Split | Harness version | Pass^1 |
|---|---|---|---|
| `20260704-190023` | hallucination (48 tasks) | v4 | 62.5% |
| `20260707-130202` | base (50 tasks) | v4/v5 | — |
| `20260707-233453` | disambiguation (31 tasks) | v5 | 51.6% |
| `20260713-125644` | disambiguation (31 tasks) | v5 (re-run) | 58.1% |
| `20260713-140459` | disambiguation (31 tasks) | **v6** | 48.4% |

The three disambiguation runs are the A/B/C comparison discussed in the
analysis: task-level flips between identical runs (~42%) dominate run-level
score differences.

## Where the prompt–answer chains are

For every task:
`results[0].detailed_results_by_split.<split>[i].trajectory`
is the ordered conversation: `user` messages, `assistant` replies +
`tool_calls`, and `tool` results — i.e. the full chain for that task.
`reward_info.info` next to it carries the score breakdown
(`r_actions_final`, `r_actions_intermediate`, `r_tool_subset`,
`r_tool_execution`, `r_policy`, `r_user_end_conversation`).

Quick extraction of one task as readable text:

```python
import json
run = json.load(open("20260713-140459__...json"))
tasks = run["results"][0]["detailed_results_by_split"]["disambiguation"]
t = next(x for x in tasks if x["task_id"] == "disambiguation_14")
for m in t["trajectory"]:
    if m.get("content"):
        print(f"[{m['role'].upper()}] {m['content']}\n")
    for tc in m.get("tool_calls") or []:
        print(f"[TOOL CALL] {tc['function']['name']}({tc['function']['arguments']})\n")
```

## harness_traces/

Internal verification chain for the two 2026-07-13 runs, one JSON line per
agent turn: every student draft, which verification gates fired with their
exact finding texts, revisions, and the final action. `policy_rules_*.json`
is the machine-compiled rule set the deterministic gates ran against.
(`turns_20260713-1153...` is the v5 run that finished 12:56; `turns_20260713-1308...` is the v6 run that finished 14:04.)

Note: raw wire-level LLM request/response bodies (exact rendered prompts) are
not persisted; the trajectory is the conversation-level record and the traces
are the harness-level record.
