"""
CAR-bench Track 1 reliability harness (v4 — Grounded Chain-of-Verification).

A deterministic "reliability shell" wrapped around a stochastic LLM core. It
inverts the benchmark's observed *Completion > Compliance* bias and reduces
Pass^3 variance by:

  * STUDENT drafts the next action (the LLM core).
  * TEACHER verifies it with decomposed, grounded questions (CoVe):
      - provenance/grounding check  (hallucination, deterministic)
      - tool/param schema check     (hallucination, deterministic)
      - compiled-policy advisories  (policy, deterministic, dynamic ruleset)
      - gather guard                (disambiguation/tool-subset, deterministic)
      - LLM critic                  (soft policy + ambiguity, grounded)
  * STUDENT revises against the teacher's facts (<= COVE_ROUNDS).
  * Deterministic TTS sanitize.

Every layer is env-flag gated and fail-safe: any internal error falls back to
the plain student draft, so the harness can never score below the baseline by
crashing. See `config.HarnessConfig` for the switches.
"""
from .config import HarnessConfig
from .orchestrator import CoVeOrchestrator, ContextState

__all__ = ["HarnessConfig", "CoVeOrchestrator", "ContextState"]
