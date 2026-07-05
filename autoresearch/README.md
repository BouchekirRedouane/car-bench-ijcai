# Autoresearch loop — automated tuning of the GCoVe harness

Optimizes the harness's *tunable text* (reliability directives, CoVe teacher
prompt, gate finding templates — see `harness/tunables.py`) against a
stratified tune subset of the released train split, with a hard
anti-overfitting firewall so the result stays valid on the HIDDEN benchmark.

## Pieces
- `tune_set.toml`     — 23-task stratified subset (half known failures, half known passes), 2 trials, official-matching judge (gemini-2.5-flash)
- `audit.py`          — acceptance firewall: known keys only, <=3 changes, placeholders preserved, NO task ids / benchmark tool names, full gate test suite (incl. alien-domain) must pass with the candidate loaded
- `eval_candidate.py` — audit -> run tune subset -> score = mean pass − 0.5·flip_rate -> results/<name>.json
- `loop.md`           — the research-agent protocol (one hypothesis -> one candidate -> evaluate -> journal)
- `candidates/`       — candidate JSONs (`{"tunable.key": "new text", "_hypothesis": "..."}`)
- `results/`, `journal.jsonl` — scores and the experiment log (feeds the technical report)

## Quick start
```bash
python autoresearch/eval_candidate.py --baseline                          # score the current defaults (~$2.5)
python autoresearch/eval_candidate.py autoresearch/candidates/example_q4_checklist.json
```
Acceptance bar: beat the champion score by >= 0.08 (~2 tasks) — smaller deltas
are inside the measured run-to-run noise band. Champions are promoted into
`tunables.py` DEFAULTS via a reviewed git commit, verified on the full split,
and the final submission config gets a Pass^3 run.

## Hidden-benchmark scalability guarantees
1. Only text under known keys can change — no new machinery, no token lists.
2. The literal audit rejects benchmark tool names (harvested dynamically from
   the vendored benchmark) and task ids inside any override.
3. The alien-domain test suite runs with every candidate — logic that only
   works on CAR-bench vocabulary fails the gate.
4. Tune/holdout split: the optimizer only ever sees the 23 tune tasks.
