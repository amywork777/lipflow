"""Your own words: names and terms lip reading can't guess (a name is just lip shapes).

One per line in ~/Library/Application Support/Lipflow/words.txt. They're used to pick
between the model's top guesses and to fix their capitalisation, and given to the LLM.
"""
from __future__ import annotations

import os
import re

from .paths import HOME

PATH = os.path.join(HOME, "words.txt")
_TEMPLATE = """# Lipflow custom words: one name or term per line, written how you want it typed.
# When one of the model's top guesses contains a word from this list, that guess wins.
Lipflow
"""


def load() -> list[str]:
    if not os.path.exists(PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        with open(PATH, "w", encoding="utf-8") as f:
            f.write(_TEMPLATE)
    words = []
    for line in open(PATH, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            words.append(line)
    return words


def _hits(text: str, words: list[str]) -> int:
    t = f" {text.upper()} "
    from .text import contains
    return sum(contains(text, w) for w in words)


def rerank(candidates: list[str], words: list[str]) -> list[str]:
    """Stable sort: guesses containing more of your words first."""
    if not words:
        return candidates
    return sorted(candidates, key=lambda c: -_hits(c, words))


def apply_case(text: str, words: list[str]) -> str:
    for w in words:
        text = re.sub(rf"(?<![a-z0-9_]){re.escape(w)}(?![a-z0-9_])", w, text, flags=re.I)
    return text
