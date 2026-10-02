"""Learn from your corrections: when you fix a word Lipflow typed, that clip + your fixed sentence
becomes training data (like Wispr Flow does), so accuracy improves without practice sessions.

After a paste, the same text field is re-read a few times over 30 s. The dictated span is found by
diffing the field against its contents before the paste, then aligned word-by-word with what was
pasted, so text you keep typing afterwards isn't mistaken for a correction. Small edits count;
rewrites don't. Only that field is read, only in memory; saved: the mouth clip + corrected text.
"""
from __future__ import annotations

import difflib
import os
import re
import time

import numpy as np

from .paths import HOME

DIR = os.path.join(HOME, "clips", "corrections")
CHECKS = (4.0, 12.0, 30.0)
KEEP = 500


def _words(t: str) -> list[str]:
    from .text import tokens
    return tokens(t)


def inserted_span(before: str, after: str) -> str:
    """What now stands where the paste went: after minus before's common prefix and suffix."""
    p = 0
    while p < min(len(before), len(after)) and before[p] == after[p]:
        p += 1
    s = 0
    while s < min(len(before), len(after)) - p and before[-1 - s] == after[-1 - s]:
        s += 1
    return after[p:len(after) - s]


def find_correction(pasted: str, span: str) -> "str | None":
    """The corrected version of `pasted` inside `span`, or None if it wasn't corrected (or was
    rewritten beyond recognition). Words typed before/after the dictation are trimmed off."""
    from .text import has_han, tokens
    if has_han(pasted):
        from .guard import edits
        pw, sw = tokens(pasted), tokens(span)
        if not pw or len(sw) < max(1, len(pw)-2):
            return None
        possibilities = []
        max_change = max(2, int(len(pw)*0.4))
        for start in range(min(len(sw), 30)):
            for length in range(max(1, len(pw)-2), min(len(sw)-start, len(pw)+2)+1):
                candidate = sw[start:start+length]
                distance = edits(pw, candidate)
                if 0 <= distance <= max_change:
                    possibilities.append((distance, abs(length-len(pw)), start, candidate))
        if not possibilities:
            return None
        best = min(possibilities)
        return None if best[0] == 0 else "".join(best[3])
    pw, sw_raw = _words(pasted), re.findall(r"\S+", span)
    sw = [" ".join(_words(w)) for w in sw_raw]
    if not pw or not sw:
        return None
    ops = difflib.SequenceMatcher(a=pw, b=sw, autojunk=False).get_opcodes()
    # trim pure insertions at either end: that's new typing, not the dictation
    while ops and ops[0][0] == "insert":
        ops.pop(0)
    while ops and ops[-1][0] == "insert":
        ops.pop()
    if not ops:
        return None
    # a replace at either end can swallow new typing next to it ("school" → "tool. Also can you…"):
    # keep only as many span words as dictated words it replaces, plus one
    tag, i1, i2, j1, j2 = ops[-1]
    if tag == "replace" and j2 - j1 > i2 - i1 + 1:
        ops[-1] = (tag, i1, i2, j1, j1 + (i2 - i1))
    tag, i1, i2, j1, j2 = ops[0]
    if tag == "replace" and j2 - j1 > i2 - i1 + 1:
        ops[0] = (tag, i1, i2, j2 - (i2 - i1), j2)
    lo, hi = ops[0][3], ops[-1][4]
    fixed_raw = sw_raw[lo:hi]
    fixed = [w for w in sw[lo:hi] if w]
    if fixed == pw:
        return None  # untouched
    sm = difflib.SequenceMatcher(a=pw, b=fixed, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    if sm.ratio() < 0.5 or changed > max(2, 0.4 * len(pw)):
        return None  # a rewrite, not a correction
    return " ".join(fixed_raw).strip()


def save(rois, pasted: str, corrected: str, raw) -> str:
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, f"{int(time.time() * 1000)}.npz")
    np.savez_compressed(path, rois=rois, text=corrected, pasted=pasted, raw=np.array(raw))
    for f in sorted(os.listdir(DIR))[:-KEEP]:
        os.remove(os.path.join(DIR, f))
    return path


def count() -> int:
    return len(os.listdir(DIR)) if os.path.isdir(DIR) else 0


def load_all(language="en") -> list[dict]:
    if not os.path.isdir(DIR):
        return []
    out = []
    for f in sorted(os.listdir(DIR)):
        d = np.load(os.path.join(DIR, f), allow_pickle=True)
        from .text import has_han
        text = str(d["text"])
        if has_han(text) == (language == "zh"):
            out.append({"rois": d["rois"], "text": text})
    return out


class Watcher:
    """Re-reads one AX text element after a paste. Call on the main thread."""

    def __init__(self, element, before: str, pasted: str, rois, raw, on_saved=None):
        self.element, self.before, self.pasted = element, before, pasted.strip()
        self.rois, self.raw, self.on_saved = rois, raw, on_saved
        self.best = None
        self.done = False

    def start(self):
        from PyObjCTools import AppHelper
        for i, t in enumerate(CHECKS):
            AppHelper.callLater(t, self.check, i == len(CHECKS) - 1)

    def check(self, last: bool):
        if self.done:
            return
        try:
            from .context import _ax
            val = _ax(self.element, "AXValue")
        except Exception:
            val = None
        if isinstance(val, str):
            fix = find_correction(self.pasted, inserted_span(self.before, val))
            if fix:
                self.best = fix
        if last or val is None:
            self.done = True
            if self.best:
                save(self.rois, self.pasted, self.best, self.raw)
                print(f"[lipflow] learned from your correction ({len(_words(self.best))} words)")
                if self.on_saved:
                    self.on_saved()
