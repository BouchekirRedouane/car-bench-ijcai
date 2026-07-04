# Evaluation results — GCoVe harness (CAR-bench Track 1)

- `raw/` — every `car-bench-run` output JSON (full per-task trajectories, rewards,
  reward components, agent cost). Filenames encode `timestamp__scenario__split__model`.
- `summary.csv` — one row per (run, task): reward, end-conversation keyword, agent cost.
- `plots/` — generated figures:
  - `pass_rate_by_run.png` — pass-rate per run, chronological, colored by category
  - `task_matrix_<category>.png` — task x run pass/fail matrix (green=pass, red=fail
    with the judge's end keyword, grey=local infra error)
  - `cost_per_task.png` — per-task agent cost by model

Regenerate everything after new runs:

    cp output/track_1_agent_under_test/*.json results/raw/
    python results/plot_results.py            # needs matplotlib only

Reading guide: the leftmost dense column in the hallucination matrix is the full
48-task run (gemini-2.5-flash, 2026-06-26). The 5-task columns to its right are the
diagnostic set (16/24/38/40/48 — the hardest cases) used to iterate on the harness;
progress shows as columns turning green left to right.
