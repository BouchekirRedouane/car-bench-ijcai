# GCoVe: A Grounded Chain-of-Verification Harness for Reliable In-Car LLM Agents

**CAR-bench IJCAI-ECAI 2026 — Track 1 (Open Track) submission.**
The agent under test lives in `src/track_1_agent_under_test/`; everything described here is the
reliability harness wrapped around a single tool-calling LLM.

---

## 1. The idea

LLM agents fail CAR-bench-style tasks in a small number of *recurring, mechanical* ways:

| Failure mode | Example |
|---|---|
| **Fabrication** | calling a tool/parameter that does not exist, inventing an id or a value that no tool returned |
| **Silent gaps** | a tool result says `"unknown"` and the final answer pretends everything was verified |
| **Substitution** | the exact tool is missing, so the agent "helpfully" rebuilds the outcome with different tools |
| **Over-refusal** | refusing a doable action, or treating a *confirmation* policy as a *prohibition* |
| **Guess instead of resolve** | asking the user (or guessing) when preferences / vehicle state / policy already determine the answer |
| **Policy skips** | missing required confirmations, auto-actions, disclosures |

The key observation is a **verification asymmetry**: *checking* a proposed action against grounded
evidence (the tool schemas, the policy, the tool results, the transcript) is much easier than
*generating* the action. So a verifier does not need to be smarter than the actor — it needs to be
**grounded**. GCoVe therefore splits verification into:

1. **Deterministic gates** (plain code, no LLM): checks whose evidence is mechanically decidable —
   does the tool exist? is the argument in the schema enum? was this id ever returned by a tool?
   did a result contain `"unknown"`? does the tool description require confirmation?
   These produce **hard findings** — always acted upon.
2. **An LLM teacher running Chain-of-Verification**: poses fixed verification questions
   (grounding, capability & substitution, partial progress, promises/completion, policy, ambiguity)
   and must answer them **only from the provided evidence**. Heuristic gate outputs are passed to it
   as **candidate findings** which it must *verify or discard* — the teacher arbitrates what code
   cannot decide, and code catches what the teacher overlooks.

A second design principle: the competition metric is **Pass^3** (a task counts only if all three
independent trials pass), so *variance is as costly as error*. The harness therefore revises a
draft **only on verified findings** — advisories alone never trigger a revision — and every layer
is deterministic, cached, or temperature-0.

## 2. Architecture

```
            evaluator (A2A)                        agent under test
  ┌────────────────────────────┐      ┌──────────────────────────────────────────┐
  │ system+user, 57 tools,     │─────▶│ policy compiler (LLM, cached by wiki hash)│
  │ tool_results per turn      │      │   policy wiki ──▶ typed rule IR           │
  └────────────────────────────┘      ├──────────────────────────────────────────┤
                                      │ S1  student draft (tool calls / reply)    │
                                      │ S2  deterministic gates      ── hard ──┐  │
                                      │     schema/enum · grounding ·          │  │
                                      │     capability · confirmation ·        │  │
                                      │     gather · ask-guard · loops ·       │  │
                                      │     completion · unknown-ack           │  │
                                      │     refusal · promises · policy ─ cand.│  │
                                      │ S3  CoVe teacher (LLM): answer Q1–Q6   │  │
                                      │     from evidence; verify candidates   │  │
                                      │ S4  revise draft on verified findings ◀┘  │
                                      │     + deterministic post-revise re-check  │
                                      │     + oscillation valve (no gather loops) │
                                      │ TTS sanitize ▶ final A2A response         │
                                      └──────────────────────────────────────────┘
```

## 3. The algorithm

```
Algorithm GCoVe-Turn(messages, tools, state)                      # one assistant turn
────────────────────────────────────────────────────────────────
1  if first turn: state.rules ← CompilePolicy(system_prompt, tools)   # LLM → typed rules, cached
2  state.provenance.ingest(messages)                                  # ids/values returned by tools
3  draft ← Student(messages, tools)                                   # S1, temperature 0
4  relax ← WriteSet(draft) ∈ state.blocked_writes                     # oscillation valve
5  for round = 1 .. cove_rounds:
6      hard, candidates ← DeterministicGates(draft, messages, tools, state.rules, relax)
7      confirmed ← CoVeTeacher(draft, evidence, hard ∪ candidates)    # S2/S3: answer Q1–Q6 from
8      findings ← hard ∪ confirmed                                    #        evidence; verify or
9      if findings = ∅: break                                         #        discard candidates
10     draft ← Revise(draft, findings)                                # S4
11 if revised:                                                        # safety net: a revision must
12     det ← DeterministicGates(draft, ...)  [hard only]              # not introduce a NEW violation
13     if det ≠ ∅: draft ← Revise(draft, det)
14 if WriteSet(S1 draft) ⊄ WriteSet(draft):                           # writes were blocked this turn:
15     state.blocked_writes += WriteSet(S1 draft)                     # next re-proposal relaxes the
16 return Sanitize(draft)                                             # gather guard (no infinite loop)
```

**Deterministic gates** (tier: H = hard finding, C = candidate for the teacher):

| Gate | Tier | Catches |
|---|---|---|
| `schema/enum` | H | nonexistent tool, invented parameter (incl. empty-schema tools), value outside a declared enum |
| `grounding` | H | id-like argument never returned by any tool |
| `capability` | H | policy demands controlling a subsystem no available tool can write |
| `confirmation` | H | calling a tool whose description carries an ALL-CAPS `*CONFIRM*` marker without the user's explicit yes (an affirmation only counts as an answer to a question) |
| `gather` | H | state change before the policy-required read (e.g. weather before sunroof) |
| `prefs` | H, one-shot | first state change before reading the learned-preferences tool (found by `*preference*` name pattern) |
| `ask-guard` | H | asking the user a *choice*-question (which/what/how-much) while preferences, request-matched reads, or policy-linked reads are still unread; yes/no confirmation questions are exempt |
| `loop` | H | identical call retried ≥2 times |
| `completion` | H | reply claims an action was done but no state-changing tool ever ran |
| `unknown-ack` | H | a tool result contained unknown/unavailable values and the success summary hides it |
| `output-integrity` | H | empty final response, or a reply cut off mid-sentence (provider truncation) — pure string-structure check |
| `refusal` | C | reply says "can't" while an available write tool matches the request (scans the last 3 user turns) |
| `promise-audit` | C | reply promises future actions — each must map to the user's *exact* operation with a listed tool; a substitute or a removed parameter is a hallucination even inside a question |
| `call-substitution` | C | the user asked to REMOVE/DELETE something and the draft CALLS a non-delete write tool on that same subject — the teacher must verify operation identity before execution |
| `policy advisories` | C | compiled-rule reminders matched to the proposed calls |

**Bounded self-correction** (failure-triggered only — clean turns are unchanged, preserving the
Pass^3 variance discipline): transient provider errors are retried (litellm `num_retries=2` on every
harness call); an unparseable teacher/compiler JSON verdict is re-asked once; `COVE_ROUNDS=2` gives
a second verify→revise cycle with the teacher in the loop when round 1 found defects; a revision
that introduces a new hard violation gets one deterministic final fix; an admission of a gap must
always offer the concrete doable alternative (directive 6 + the unknown-ack finding).

**Scalability invariant** (the framework is evaluated on a *hidden* benchmark): no gate contains a
task id, tool name, or answer. All behavior derives at runtime from the tool schemas
(names/parameters/enums/descriptions), the compiled policy rules, the tool results, and the
transcript. Markers are structural (regex over description prefixes, interrogative sentence
structure, name-token overlap). The test suite includes **alien-domain tests** that run every gate
against a fictional smart-home toolset — any benchmark-specific logic would fail them.

## 4. How to use

### Setup

```bash
git clone <this repo> && cd car-bench-ijcai
uv sync                                     # or: pip install -e .
bash setup.sh                               # pulls third_party/car-bench + mock data
cp .env.example .env                        # if present; otherwise create .env
```

`.env` (never committed):

```bash
OPENROUTER_API_KEY=sk-or-...       # any litellm-routable provider works
TEACHER_LLM=openrouter/anthropic/claude-sonnet-4.6   # verifier model (defaults to the agent model)
```

### Run an evaluation slice

```bash
# 5-task slices per category (models are set inside the toml / via --agent-llm)
uv run car-bench-run scenarios/track_1_agent_under_test/local_base.toml            --show-logs
uv run car-bench-run scenarios/track_1_agent_under_test/local_disambiguation.toml  --show-logs
uv run car-bench-run scenarios/track_1_agent_under_test/local_hallucination.toml   --show-logs
# Pass^3: set num_trials = 3 in the toml. Full split: tasks_<type>_num_tasks = -1
```

Results land in `output/track_1_agent_under_test/…json`; per-turn harness traces (drafts, every
gate's findings, teacher verdicts, revisions) in `output/harness_traces/`.

### Configuration (all env vars, all optional)

Every layer is independently switchable so organizers/ablations can toggle without code changes:

```
ENABLE_HARNESS=0            # pure single-pass baseline (everything off)
ENABLE_SYSTEM_PROMPT / ENABLE_PROVENANCE / ENABLE_POLICY_COMPILE / ENABLE_POLICY_ENFORCE
ENABLE_GATHER_GUARD / ENABLE_HALLUC_GATE / ENABLE_CAPABILITY / ENABLE_REFUSAL_CHECK
ENABLE_CONFIRMATION_GATE / ENABLE_COMPLETION_CHECK / ENABLE_LOOP_CHECK
ENABLE_VERIFY               # the LLM teacher (CoVe)
ENABLE_DISAMBIG             # ask-guard + ambiguity question
ENABLE_SANITIZE             # TTS output cleanup
COVE_ROUNDS=2  MAX_FINDINGS=6  TEACHER_LLM=<model>
```

### No API budget? Manual (human-as-LLM) mode

```bash
# .env: MANUAL_LLM=1   (optional MANUAL_LLM_PORT=8765)
uv run car-bench-run scenarios/track_1_agent_under_test/local_hallucination.toml --show-logs
# open http://127.0.0.1:8765 — each model call appears as a browser form:
# copy the prompt into any chat GUI, paste the reply back, submit. $0 spent.
```

### Tests (no LLM calls, <5 s)

```bash
.venv/bin/python tests/test_harness_gates.py     # or pytest tests/
```

Regression tests replay the exact recorded failures that motivated each gate, plus the
alien-domain scalability suite.

## 5. Results on train slices (5 tasks/category, Pass^1, gemini-2.5-flash unless noted)

| Category | Before the harness fixes | Best measured with GCoVe |
|---|---|---|
| Base | 2/5 (over-refusal, skipped confirmation, invalid enum) | **5/5** |
| Hallucination (5 hardest tasks) | 1/5 | **4/5** (3/5 typical; h_48 substitution now passes consistently) |
| Disambiguation | 2/5 (asked/guessed instead of resolving) | **3/5** flash · **4/5** with a Sonnet-4.6 agent |

Remaining failures cluster on (a) verifier-model judgment — the cheap teacher sometimes discards a
correct candidate finding (one-line fix: a stronger `TEACHER_LLM`) — and (b) user-simulator
judgment variance on borderline acknowledgment phrasings, which is exactly what the Pass^3 metric
punishes and why per-trial consistency drives the whole design.

## 6. Repository layout (submission-relevant)

```
src/track_1_agent_under_test/
  car_bench_agent.py          # A2A executor (unchanged protocol boundary)
  harness/
    orchestrator.py           # the GCoVe turn loop (algorithm above)
    verify.py                 # all deterministic gates + the CoVe teacher call
    policy.py                 # runtime policy wiki -> typed rule IR (cached)
    provenance.py             # grounding ledger for ids/values
    prompts.py                # directives, compiler/teacher/revise prompts
    config.py                 # env-flag configuration
    sanitize.py / trace.py    # TTS cleanup, on-disk turn traces
src/manual_llm.py             # browser-form human-as-LLM mode (no API cost)
scenarios/track_1_agent_under_test/local_{base,disambiguation,hallucination}.toml
tests/test_harness_gates.py   # regression + alien-domain scalability suite
report/technical_report.{md,tex}
```
