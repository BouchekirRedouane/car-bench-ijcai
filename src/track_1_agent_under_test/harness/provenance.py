"""Provenance ledger — the deterministic oracle behind the hallucination gate.

It accumulates every value the agent is *allowed* to use: facts in the system
policy (e.g. CURRENT_LOCATION), values the user stated, and every field returned
by a real tool result. Any id-like token in an outgoing tool argument that is not
in the ledger is a fabrication (the agent invented an id instead of looking it up).

The check is intentionally conservative: it only flags opaque id-like tokens
(loc_..., poi_..., route_..., contact id patterns), never plain numbers/enums the
user supplied, to keep false positives ~zero.
"""
from __future__ import annotations

import json
import re

# Opaque id patterns used across CAR-bench mock data (locations, POIs, routes,
# charging stations, contacts). Matched case-insensitively.
_ID_RE = re.compile(
    r"\b(?:loc|poi|route|rt|chg|charge|charging|station|cont|contact|cal|event|user)_[A-Za-z0-9][A-Za-z0-9_\-]*\b",
    re.IGNORECASE,
)


def extract_ids(text: str) -> set[str]:
    if not text:
        return set()
    return {m.group(0) for m in _ID_RE.finditer(text)}


class ProvenanceLedger:
    """Set of grounded id tokens + the raw grounded texts (for substring checks)."""

    def __init__(self) -> None:
        self.ids: set[str] = set()
        self._texts: list[str] = []

    def reset(self) -> None:
        self.ids.clear()
        self._texts.clear()

    def ingest_messages(self, messages: list[dict]) -> None:
        """Rebuild the ledger from the full conversation (idempotent)."""
        self.reset()
        for m in messages:
            role = m.get("role")
            if role in ("system", "user", "tool"):
                self._ingest_text(m.get("content"))

    def _ingest_text(self, content) -> None:
        if not content:
            return
        if not isinstance(content, str):
            try:
                content = json.dumps(content)
            except Exception:
                content = str(content)
        self._texts.append(content)
        self.ids |= extract_ids(content)

    def is_grounded(self, token: str) -> bool:
        """True if the token was provided somewhere (id set or any grounded text)."""
        if not token:
            return True
        if token in self.ids:
            return True
        return any(token in t for t in self._texts)
