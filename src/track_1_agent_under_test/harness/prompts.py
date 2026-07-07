"""Prompt fragments for the harness.

All optimizable text now lives in tunables.py (see the autoresearch loop);
this module composes the runtime prompts from it. Set HARNESS_TUNABLES to a
candidate JSON to override any piece without code changes.
"""
from .tunables import TUN

# Appended once to the evaluator-provided policy (the system message). It does
# not contradict the policy — it makes the reliability priorities explicit, which
# is exactly what the baseline models under-weight (Completion > Compliance).
HARNESS_SYSTEM_SUFFIX = (
    TUN["suffix.header"]
    + "\n"
    + "\n".join(f"{i}. {TUN[f'directive.{i}']}" for i in range(1, 11))
    + "\n"
)

# Compiles the provided policy wiki into a typed, machine-checkable rule set.
# Shape-based (not policy-specific) so it generalizes to unseen policies.
POLICY_COMPILER_SYSTEM = TUN["compiler.system"]
POLICY_EXEC_SYSTEM = TUN["compiler.exec_system"]

# The TEACHER — a grounded Chain-of-Verification pass. It POSES verification
# questions, ANSWERS each from the provided evidence only, and lists defects.
# Deterministic pre-check findings are passed in as candidates it must verify.
TEACHER_SYSTEM = TUN["teacher.system"]

TEACHER_USER_TEMPLATE = """\
=== POLICY ===
{policy}

=== AVAILABLE TOOL NAMES ===
{tool_names}

=== TRANSCRIPT (most recent) ===
{transcript}

=== DATA THE STUDENT HAS (tool results received + arguments passed) ===
{data}

=== STUDENT PROPOSED NEXT ACTION ===
{action}

=== PRE-CHECK FINDINGS (candidate findings to verify — discard any that are wrong) ===
{det_findings}

Run the chain of verification and return the JSON verdict now.
"""

# Dedicated claim-grounding verifier — kept for ablation (folded into CoVe Q1).
CLAIM_GROUNDING_SYSTEM = """\
You verify that an assistant's spoken REPLY states ONLY facts it actually has. You are given the
REPLY and ALL data the assistant obtained this task: the TOOL RESULTS it received and the TOOL CALLS
(with arguments) it made.

Flag any specific factual claim in the reply that is NOT supported by that data:
- a value, number, measurement, level, percentage, name, or status the reply states that does not
  appear in any tool result;
- a claim that it set / changed / turned / adjusted something to a specific value when no tool call
  passed that value as an argument;
- a status it asserts about something for which the tool result has no field (the field may have been
  unavailable — it must NOT invent it).

Paraphrasing, tone, and general helpfulness are fine — only flag concrete UNSUPPORTED facts. If a
claim is unsupported, the assistant is fabricating: it must state only what the data shows, or tell
the user that this specific information or capability is unavailable.

Output ONLY JSON: {"ok": <true if every claim is supported>, "findings": ["<one imperative fix per
unsupported claim>"]}. No prose, no markdown fences.
"""

CLAIM_GROUNDING_TEMPLATE = """\
=== ASSISTANT REPLY ===
{reply}

=== DATA THE ASSISTANT ACTUALLY HAS (tool results it received + arguments it passed) ===
{data}

Return the JSON verdict now.
"""

# Feedback handed back to the student to drive the revision (CoVe S4).
REVISE_USER_TEMPLATE = TUN["revise.template"]
