"""Snap mis-read names to the names you're likely saying, by how they *look* on the lips.

Lip reading can't hear, so "Miguel" comes out as MCCALL / MC HALE / NICKEL. Those differ in
letters but not in lip shapes. Each word is mapped to a coarse viseme sequence (lip/tongue shape
classes, from spelling — no pronunciation dictionary needed), and a span of 1–2 read words is
replaced by a candidate name when their viseme sequences are close. Only uncommon words are
eligible, so ordinary words ("my", "tool") are never swapped.
"""
from __future__ import annotations

import os
import re

# digraphs first, then single letters → viseme class
_DIGRAPHS = [("ch", "J"), ("sh", "J"), ("th", "D"), ("ph", "F"), ("ck", "K"), ("qu", "KW"), ("gu", "K"),
             ("mc", "MK"), ("wh", "W"), ("ng", "K"), ("ee", "I"), ("oo", "U"), ("ou", "U"), ("ea", "I")]
_SINGLE = {**dict.fromkeys("pbm", "M"), **dict.fromkeys("fv", "F"), **dict.fromkeys("tdnszlx", "T"),
           **dict.fromkeys("kgcq", "K"), **dict.fromkeys("jy", "J"), "r": "W", "w": "W",
           **dict.fromkeys("ae", "A"), "i": "I", **dict.fromkeys("ou", "U"), "h": ""}


def visemes(word: str) -> str:
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) > 3 and w.endswith("e") and w[-2] not in "aeiou":
        w = w[:-1]  # silent e: HALE is said "hail"
    out, i = [], 0
    while i < len(w):
        for dg, v in _DIGRAPHS:
            if w.startswith(dg, i):
                out.append(v)
                i += len(dg)
                break
        else:
            out.append(_SINGLE.get(w[i], ""))
            i += 1
    s = "".join(out)
    return re.sub(r"(.)\1+", r"\1", s)  # doubled letters look like one


_DICT: "set[str] | None" = None


def is_word(w: str) -> bool:
    """A real lowercase English word (macOS's /usr/share/dict/words; proper nouns there are
    capitalised, so names like Mccall don't count). Misread names are almost always non-words."""
    global _DICT
    if _DICT is None:
        try:
            import sys
            if sys.platform == "darwin" and os.path.exists("/usr/share/dict/words"):
                _DICT = {l.strip() for l in open("/usr/share/dict/words") if l[:1].islower()}
            else:  # Linux / Windows: use bundled web2 (1934, public domain)
                raise OSError
        except OSError:  # Windows and Linux: the same list (web2, 1934, public domain) ships with Lipflow
            import gzip
            path = os.path.join(os.path.dirname(__file__), "data", "web2-lower.txt.gz")
            try:
                _DICT = set(gzip.open(path, "rt", encoding="utf-8").read().split())
            except OSError:
                _DICT = set()
    w = w.lower().replace("'", "")
    if w in _DICT:
        return True
    # the dictionary has no inflections: try stems (planks→plank, served→serve, stopped→stop)
    for suf, add in (("ies", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"),
                     ("er", ""), ("est", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            stem = w[: -len(suf)] + add
            if stem in _DICT or (len(stem) > 3 and stem[-1] == stem[-2] and stem[:-1] in _DICT):
                return True
    return False


def _dist(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def snap_names(text: str, names: list[str], is_common, max_ratio: float = 0.25) -> str:
    """Replace 1–2 word spans that look like a name on the lips. `is_common(word)` protects
    everyday words. Works on the raw uppercase guess, before cleanup."""
    if not names:
        return text
    words = text.split()
    # a "name" that is an ordinary word (Balance, Flow) would turn real words into itself
    targets = [(n, visemes(n)) for n in names if len(visemes(n)) >= 3 and not is_word(n)]
    out, i = [], 0
    while i < len(words):
        best = None
        for span in (2, 1):
            chunk = words[i:i + span]
            if len(chunk) < span or any(is_common(w) for w in chunk):
                continue
            if all(is_word(w) for w in chunk):  # "four", "planks": real words are left alone
                continue
            v = visemes("".join(chunk))
            for name, nv in targets:
                if chunk and " ".join(chunk).lower() == name.lower():
                    continue
                if not v or v[0] != nv[0]:  # the first lip shape is the most visible
                    continue
                r = _dist(v, nv) / max(len(nv), len(v))
                if r <= max_ratio and (best is None or r < best[0]):
                    best = (r, span, name)
            if best:
                break
        if best:
            out.append(best[2].upper())
            i += best[1]
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)
