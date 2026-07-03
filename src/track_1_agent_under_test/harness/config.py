"""Harness configuration — all behaviour is controlled by environment variables
so the organizers (and our own ablation runs) can toggle every layer without
code changes. Defaults are chosen to be the full v4 harness."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


@dataclass
class HarnessConfig:
    """Resolved configuration for one agent process."""

    # Model / inference (from CLI/env, passed in by the executor)
    model: str = "gemini/gemini-2.5-flash"
    temperature: float | None = 0.0
    thinking: bool = False
    reasoning_effort: str = "medium"
    interleaved_thinking: bool = False

    # Separate (optionally stronger) verifier model. Defaults to the student.
    teacher_model: str | None = None

    # Master switch — when off, the agent is a pure single-pass baseline.
    enable_harness: bool = True

    # Individual layers (all on by default).
    enable_system_prompt: bool = True   # compliance-first guidance suffix
    enable_provenance: bool = True       # grounding ledger + hallucination on ids
    enable_policy_compile: bool = True   # runtime policy -> rule IR
    enable_policy_enforce: bool = True   # rule advisories on proposed actions
    enable_gather_guard: bool = True     # require gather before state change
    enable_halluc_gate: bool = True      # tool/param existence check
    enable_capability: bool = True       # admit when a required tool is unavailable
    enable_refusal_check: bool = True    # flag "claims can't-do but a matching tool exists"
    enable_confirmation_gate: bool = True  # block confirmation-marked tools until user affirms
    enable_completion_check: bool = True  # flag "claims done but performed no action"
    enable_output_guard: bool = True     # reject empty / mid-sentence-truncated final replies
    enable_claim_grounding: bool = False  # folded into the CoVe critic (Q1); kept for ablation
    enable_loop_check: bool = True        # flag repeated identical tool calls (stuck)
    enable_verify: bool = True           # LLM teacher critic (semantic)
    enable_disambig: bool = True         # ask-vs-resolve guidance in critic
    enable_sanitize: bool = True         # TTS output cleanup

    # CoVe revise rounds and (future) self-consistency vote width.
    cove_rounds: int = 1
    vote_n: int = 1
    # Cap on findings passed to a revision so a weak model is not overloaded.
    max_findings: int = 6

    @classmethod
    def from_env(
        cls,
        model: str,
        temperature: float | None,
        thinking: bool,
        reasoning_effort: str,
        interleaved_thinking: bool,
    ) -> "HarnessConfig":
        return cls(
            model=model,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            interleaved_thinking=interleaved_thinking,
            teacher_model=(os.getenv("TEACHER_LLM") or model),
            enable_harness=_flag("ENABLE_HARNESS", True),
            enable_system_prompt=_flag("ENABLE_SYSTEM_PROMPT", True),
            enable_provenance=_flag("ENABLE_PROVENANCE", True),
            enable_policy_compile=_flag("ENABLE_POLICY_COMPILE", True),
            enable_policy_enforce=_flag("ENABLE_POLICY_ENFORCE", True),
            enable_gather_guard=_flag("ENABLE_GATHER_GUARD", True),
            enable_halluc_gate=_flag("ENABLE_HALLUC_GATE", True),
            enable_capability=_flag("ENABLE_CAPABILITY", True),
            enable_refusal_check=_flag("ENABLE_REFUSAL_CHECK", True),
            enable_confirmation_gate=_flag("ENABLE_CONFIRMATION_GATE", True),
            enable_completion_check=_flag("ENABLE_COMPLETION_CHECK", True),
            enable_output_guard=_flag("ENABLE_OUTPUT_GUARD", True),
            enable_claim_grounding=_flag("ENABLE_CLAIM_GROUNDING", False),
            enable_loop_check=_flag("ENABLE_LOOP_CHECK", True),
            enable_verify=_flag("ENABLE_VERIFY", True),
            enable_disambig=_flag("ENABLE_DISAMBIG", True),
            enable_sanitize=_flag("ENABLE_SANITIZE", True),
            cove_rounds=_int("COVE_ROUNDS", 1),
            vote_n=_int("VOTE_N", 1),
            max_findings=_int("MAX_FINDINGS", 6),
        )

    def summary(self) -> dict:
        """Compact dict for startup logging / the technical report."""
        return {
            "model": self.model,
            "teacher_model": self.teacher_model,
            "enable_harness": self.enable_harness,
            "system_prompt": self.enable_system_prompt,
            "provenance": self.enable_provenance,
            "policy_compile": self.enable_policy_compile,
            "policy_enforce": self.enable_policy_enforce,
            "gather_guard": self.enable_gather_guard,
            "halluc_gate": self.enable_halluc_gate,
            "capability": self.enable_capability,
            "refusal_check": self.enable_refusal_check,
            "confirmation_gate": self.enable_confirmation_gate,
            "completion_check": self.enable_completion_check,
            "output_guard": self.enable_output_guard,
            "claim_grounding": self.enable_claim_grounding,
            "loop_check": self.enable_loop_check,
            "verify": self.enable_verify,
            "disambig": self.enable_disambig,
            "sanitize": self.enable_sanitize,
            "cove_rounds": self.cove_rounds,
            "vote_n": self.vote_n,
        }
