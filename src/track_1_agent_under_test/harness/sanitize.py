"""Deterministic TTS output sanitizer. The evaluator forwards the agent's reply
to a text-to-speech model and penalizes non-speakable content (markdown, lists,
bullets, emoji). This strips those so a single missed instruction never costs a
whole task."""
from __future__ import annotations

import re

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "\U0000FE00-\U0000FE0F"
    "\U00002022"  # bullet •
    "]",
    flags=re.UNICODE,
)


def sanitize(text: str) -> str:
    if not text:
        return text
    t = text
    # Code fences / inline code ticks
    t = re.sub(r"```[a-zA-Z]*", "", t)
    t = t.replace("`", "")
    # Bold / italic markers
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"\*(.+?)\*", r"\1", t, flags=re.DOTALL)
    t = re.sub(r"__(.+?)__", r"\1", t, flags=re.DOTALL)
    # Markdown headers
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
    # Bullet list markers at line start
    t = re.sub(r"(?m)^\s*[-*+•]\s+", "", t)
    # Numbered list markers at line start ("1. ", "2) ")
    t = re.sub(r"(?m)^\s*\d+[.)]\s+", "", t)
    # Markdown links [text](url) -> text
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    # Emoji / symbols
    t = _EMOJI_RE.sub("", t)
    # Collapse whitespace
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
