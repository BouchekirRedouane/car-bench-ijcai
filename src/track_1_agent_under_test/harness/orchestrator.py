"""CoVe orchestrator — runs one assistant turn through the reliability shell and
returns a single assistant message dict (same shape the executor renders to A2A).

  S1  student draft           -> proposed next action
  S2  teacher plans questions  } run_verification()
  S3  teacher answers (grounded)
  S4  student revises with the findings  (<= cove_rounds)
      deterministic TTS sanitize

Fail-safe: any internal error after a successful draft returns the best draft so
far, so the harness can never crash a turn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .config import HarnessConfig
from .llm import call_llm, parse_json_object
from .policy import compile_policy
from .prompts import REVISE_USER_TEMPLATE
from .provenance import ProvenanceLedger
from .sanitize import sanitize
from . import trace
from .verify import describe_action, run_verification, _is_read_tool
from .tunables import TUN
import json as _json

logger = logging.getLogger("harness.orchestrator")


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
        if m.get("role") == "tool":
            return "[tool results]"
    return ""


def _write_set(draft: dict) -> frozenset:
    """The state-changing tool names a draft proposes (empty for replies/reads)."""
    return frozenset(
        name
        for tc in (draft.get("tool_calls") or [])
        if (name := (tc.get("function") or {}).get("name")) and not _is_read_tool(name)
    )


@dataclass
class ContextState:
    """Per-task (per context_id) harness state."""

    provenance: ProvenanceLedger = field(default_factory=ProvenanceLedger)
    rules: Optional[list] = None  # compiled lazily on first turn
    turn: int = 0                 # assistant-turn counter for this task
    # Oscillation valve: write-sets that were proposed but revised away. If the
    # student re-proposes the same writes on a later turn (after its gather
    # detour), the gather guard is relaxed so the turn can execute instead of
    # looping propose -> gather -> re-propose forever.
    blocked_writes: list = field(default_factory=list)
    # v5 request ledger: the user's asks (extracted once per user message);
    # the final reply must cover each one (done or acknowledged).
    ledger: list = field(default_factory=list)
    seen_user_msgs: int = 0


class CoVeOrchestrator:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg

    # -- LLM passes -------------------------------------------------------- #
    def _student(self, messages, tools, record) -> dict:
        return call_llm(
            messages,
            tools,
            model=self.cfg.model,
            temperature=self.cfg.temperature,
            thinking=self.cfg.thinking,
            reasoning_effort=self.cfg.reasoning_effort,
            interleaved_thinking=self.cfg.interleaved_thinking,
            record=record,
        )

    def _revise(self, messages, tools, draft, findings, record, model=None) -> dict:
        feedback = REVISE_USER_TEMPLATE.format(
            findings="\n".join(f"- {f}" for f in findings),
            action=describe_action(draft),
        )
        working = list(messages) + [{"role": "user", "content": feedback}]
        if model:
            logger.info("escalation: revising with %s", model)
        return call_llm(
            working,
            tools,
            model=model or self.cfg.model,
            temperature=self.cfg.temperature,
            thinking=self.cfg.thinking,
            reasoning_effort=self.cfg.reasoning_effort,
            interleaved_thinking=self.cfg.interleaved_thinking,
            record=record,
        )

    def _student_vote(self, messages, tools, record) -> tuple[dict, int]:
        """S1 with self-consistency: when the draft changes state and vote_n > 1,
        sample additional drafts and execute the MODAL action set. Read-only and
        reply turns are never re-sampled (no benefit, pure cost)."""
        draft = self._student(messages, tools, record)
        n = max(1, int(self.cfg.vote_n or 1))
        if n <= 1 or not _write_set(draft):
            return draft, 1
        def key(d):
            calls = []
            for tc in (d.get("tool_calls") or []):
                fn = tc.get("function") or {}
                a = fn.get("arguments")
                if isinstance(a, (dict, list)):
                    a = _json.dumps(a, sort_keys=True)
                calls.append(f"{fn.get('name')}|{a}")
            return "||".join(sorted(calls))
        votes = {key(draft): [draft]}
        for _ in range(n - 1):
            try:
                d = self._student(messages, tools, record)
            except Exception:  # noqa: BLE001 — a failed sample never blocks the turn
                continue
            votes.setdefault(key(d), []).append(d)
        best_key, best = max(votes.items(), key=lambda kv: len(kv[1]))
        if len(best) > 1 and best_key != key(draft):
            logger.info("vote: modal action set won %d/%d (replacing first draft)",
                        len(best), sum(len(v) for v in votes.values()))
        return best[0], sum(len(v) for v in votes.values())

    def _update_ledger(self, messages, state: ContextState, record) -> None:
        """Extract the user's asks from any NEW user message (v5 request ledger)."""
        user_msgs = [m for m in messages if m.get("role") == "user" and m.get("content")]
        if len(user_msgs) <= state.seen_user_msgs:
            return
        for m in user_msgs[state.seen_user_msgs:]:
            try:
                msg = call_llm(
                    [{"role": "system", "content": TUN["ledger.system"]},
                     {"role": "user", "content": str(m["content"])[:2000]}],
                    None, model=self.cfg.teacher_model or self.cfg.model,
                    temperature=0.0, json_mode=True, record=record,
                )
                asks = parse_json_object(msg.get("content")).get("asks") or []
                state.ledger.extend(str(a).strip() for a in asks[:5] if str(a).strip())
                state.ledger = state.ledger[-12:]
            except Exception as e:  # noqa: BLE001 — fail-safe
                logger.warning("ledger extraction failed (%s); skipping", e)
        state.seen_user_msgs = len(user_msgs)
        if state.ledger:
            logger.info("ledger: %s", state.ledger)

    # -- main entry -------------------------------------------------------- #
    def run_turn(self, messages, tools, state: ContextState, record=None) -> dict:
        cfg = self.cfg

        # Update grounding ledger and (once) compile the policy.
        if cfg.enable_provenance:
            try:
                state.provenance.ingest_messages(messages)
            except Exception as e:  # noqa: BLE001
                logger.warning("Provenance ingest failed (%s)", e)

        if cfg.enable_policy_compile and state.rules is None:
            sys_text = ""
            if messages and messages[0].get("role") == "system":
                sys_text = messages[0].get("content") or ""
            # Compile only the evaluator's policy, not our own injected reliability
            # directives — otherwise the compiler turns them into circular "rules".
            marker = "## Reliability directives"
            if marker in sys_text:
                sys_text = sys_text[: sys_text.index(marker)].rstrip()
            tool_names = [
                (t or {}).get("function", {}).get("name")
                for t in (tools or [])
                if (t or {}).get("function", {}).get("name")
            ]
            state.rules = compile_policy(
                sys_text, model=cfg.teacher_model or cfg.model,
                tool_names=tool_names, record=record,
            )
            trace.record_rules(state.rules or [])
        rules = state.rules or []

        state.turn += 1
        rounds: list[dict] = []  # for the on-disk trace

        if cfg.enable_harness and cfg.enable_ledger:
            self._update_ledger(messages, state, record)

        # S1 — student draft (this call may raise; the executor handles LLM errors).
        draft, passes = self._student_vote(messages, tools, record)
        logger.info("S1 student draft: %s", describe_action(draft)[:300])
        if draft.get("reasoning_content"):
            logger.debug("S1 student reasoning: %s", str(draft.get("reasoning_content"))[:500])
        rounds.append({"draft": describe_action(draft)})

        if not cfg.enable_harness:
            draft["num_passes"] = passes
            self._record_turn(state, messages, rules, rounds, draft, passes)
            return draft

        # Oscillation valve: if this exact write-set was already proposed and
        # revised away on an earlier turn, the agent has had its gather detour —
        # relax the gather guard so the action can finally execute (or the
        # teacher can direct an admit), instead of looping forever.
        s1_writes = _write_set(draft)
        relax = bool(s1_writes) and s1_writes in state.blocked_writes
        if relax:
            logger.info("oscillation valve: write-set re-proposed after a block -> relaxing gather guard")

        # S2–S4 — verify then revise, bounded by cove_rounds. Fail-safe.
        try:
            for rnd in range(max(0, cfg.cove_rounds)):
                sink: dict = {}
                findings = run_verification(
                    draft, messages, tools, rules, state.provenance, cfg,
                    record=record, stage_sink=sink, relax_gather=relax,
                    ledger=state.ledger,
                )
                rounds[-1]["stages"] = sink
                rounds[-1]["findings_total"] = len(findings)
                if not findings:
                    logger.info("CoVe round %d: verified, no revision needed", rnd + 1)
                    break
                logger.info("CoVe round %d: %d finding(s) -> revising", rnd + 1, len(findings))
                revised = self._revise(messages, tools, draft, findings, record)
                if revised:
                    draft = revised
                    passes += 1
                    logger.info("S4 revised draft: %s", describe_action(draft)[:300])
                    rounds.append({"draft": describe_action(draft)})
        except Exception as e:  # noqa: BLE001 - never crash a turn
            logger.warning("CoVe loop error (%s); using current draft", e)

        # Safety net: a revision must not silently introduce a NEW deterministic
        # violation (e.g. the critic pushing a removed parameter, an invented id,
        # or a fake completion). Re-check the revised draft with the cheap
        # deterministic gates only (no extra LLM verification) and apply one
        # corrective fix. General + hidden-safe: schema/grounding/capability, no
        # task-specific knowledge.
        if passes > 1:
            try:
                det_sink: dict = {}
                det = run_verification(
                    draft, messages, tools, rules, state.provenance, cfg,
                    record=record, stage_sink=det_sink, skip_llm=True,
                )
                if det:
                    logger.info("post-revise re-check: %d deterministic violation(s) -> final fix", len(det))
                    fixed = self._revise(messages, tools, draft, det, record,
                                         model=cfg.escalation_model)
                    if fixed:
                        draft = fixed
                        passes += 1
                        rounds.append({"draft": describe_action(draft),
                                       "stages": {"post-revise-recheck": det_sink}})
                else:
                    logger.info("post-revise re-check: clean")
            except Exception as e:  # noqa: BLE001 - never crash a turn
                logger.warning("post-revise re-check failed (%s)", e)

        # Remember writes that were proposed but revised away, so a later
        # re-proposal opens the oscillation valve above.
        try:
            if s1_writes and not (s1_writes <= _write_set(draft)) and s1_writes not in state.blocked_writes:
                state.blocked_writes.append(s1_writes)
                logger.info("oscillation valve: recorded blocked write-set %s", sorted(s1_writes))
        except Exception:  # noqa: BLE001
            pass

        logger.info("turn complete: policy_rules=%d cove_passes=%d", len(rules), passes)

        # Deterministic TTS sanitize on the final user-facing text.
        if cfg.enable_sanitize:
            try:
                if draft.get("content"):
                    draft["content"] = sanitize(draft["content"])
            except Exception as e:  # noqa: BLE001
                logger.warning("Sanitize failed (%s)", e)

        draft["num_passes"] = passes
        self._record_turn(state, messages, rules, rounds, draft, passes)
        return draft

    def _record_turn(self, state, messages, rules, rounds, draft, passes) -> None:
        """Persist a structured trace of this turn (fail-safe)."""
        try:
            final_tools = [
                (tc.get("function") or {}).get("name")
                for tc in (draft.get("tool_calls") or [])
            ]
            trace.record_turn({
                "context_id": getattr(state, "_context_id", None),
                "turn": state.turn,
                "user_msg": _last_user_text(messages)[:500],
                "policy_rules": len(rules),
                "rounds": rounds,
                "num_passes": passes,
                "final": {
                    "content": draft.get("content"),
                    "tool_calls": final_tools,
                },
            })
        except Exception:
            pass
