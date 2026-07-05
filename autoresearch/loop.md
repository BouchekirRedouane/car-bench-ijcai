# Autoresearch loop protocol (GCoVe tunables optimization)

Role: you are a research agent iterating on the GCoVe harness's tunable text
(see `src/track_1_agent_under_test/harness/tunables.py` for all keys). One
iteration = one hypothesis = one candidate. Never edit code — only propose
tunables candidates. The deterministic token lists / verb clusters are code
and are out of bounds by design.

## One iteration

1. READ the evidence, newest first:
   - `autoresearch/journal.jsonl` (what has been tried; do not repeat)
   - `autoresearch/results/*.json` (per-task scores of previous candidates)
   - `results/raw/` transcripts of the tune-set failures
2. FORM one falsifiable hypothesis about ONE mechanism, e.g.
   "the teacher discards promise-audit candidates because Q4 asks for a
   holistic judgment; an explicit per-promise checklist will make it verify
   each promise" — not "make the prompt better".
3. WRITE one candidate `autoresearch/candidates/<slug>.json`:
   - keys: subset of tunables keys, at most 3 changed;
   - `_name` and `_hypothesis` metadata keys are allowed and encouraged;
   - keep every `{placeholder}` that the default text for that key uses;
   - NEVER include benchmark task ids, tool names, or values from train
     tasks — the audit rejects the candidate and the iteration is wasted.
4. EVALUATE: `python autoresearch/eval_candidate.py autoresearch/candidates/<slug>.json`
   (this runs the audit + test suite first, then the 23-task tune subset x2
   trials, ~$2.5). The baseline to beat is `autoresearch/results/baseline.json`.
5. JUDGE: accept only if `score` beats the current champion by >= 0.08
   (~2 tasks) — anything smaller is inside the measured noise band.
6. LOG one line to `autoresearch/journal.jsonl`:
   `{"candidate": ..., "hypothesis": ..., "score": ..., "champion_score": ...,
     "verdict": "accept|reject", "notes": ...}`
7. If accepted: the candidate becomes the champion; propose it as a normal
   git commit (tunables override promoted into DEFAULTS) for human review.

## Standing rules

- Judge/user-sim model in `tune_set.toml` stays gemini-2.5-flash (matches the
  official evaluator). Never "save money" by switching the judge — wording
  tuned against a different judge does not transfer.
- The holdout (all train tasks not in tune_set.toml) is off-limits: never read
  its transcripts, never add its ids to the tune set.
- A champion must pass a full-split verification run before being promoted to
  DEFAULTS, and the final submission config additionally needs a Pass^3 run.
- Every accepted change lands as a reviewed git commit — the journal entry is
  its documentation for the technical report.

## Suggested first hypotheses (from measured failure evidence)

1. teacher.system Q4 -> explicit per-promise checklist output format
   (flash discarded valid promise-audit candidates 5x).
2. finding.conditional_scope -> require the teacher to output the item/value
   enumeration in its answer, not just be told to enumerate (base_40/94).
3. directive.3 + finding.ask_guard -> stronger resolve-ladder wording
   (d_4-class ask-instead-of-resolve on the cheap agent).
4. Flag ablation: ENABLE_CLAIM_GROUNDING=1 with claim.* prompts (currently
   folded into Q1 — measure whether the dedicated pass pays for itself).
