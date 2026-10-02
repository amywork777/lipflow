"""Learn from what you already say: import your Wispr Flow history (or any text).

`lipflow import-wispr` reads Wispr Flow's local database read-only, keeps just the text of
your dictations, and writes it to ~/Library/Application Support/Lipflow/phrases.txt. Nothing
leaves your Mac. From those phrases Lipflow builds:

- a small word-pair model of how you talk, used to pick between the lip-reader's guesses;
- a lookup of your past sentences closest to a new guess, shown to the cleanup model;
- suggested names/terms, added to words.txt for you to review.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sqlite3
import sys
from collections import Counter

from . import vocab

from .paths import HOME as DIR
PHRASES = os.path.join(DIR, "phrases.txt")
if sys.platform == "win32":  # Electron keeps its data in %APPDATA%\<app name>
    WISPR_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Wispr Flow")
else:
    WISPR_DIR = os.path.expanduser("~/Library/Application Support/Wispr Flow")
_WORD = re.compile(r"[a-z0-9']+")


def words_of(text: str) -> list[str]:
    from .text import tokens
    return tokens(text)


# -- import ---------------------------------------------------------------------------
def _looks_like_dictation(values: list[str]) -> float:
    """Score a column: long-ish natural-language strings, not ids/JSON/paths."""
    good = 0
    for v in values:
        if not isinstance(v, str):
            continue
        n = len(v.split())
        letters = sum(c.isalpha() or c.isspace() for c in v) / max(len(v), 1)
        if 2 <= n <= 400 and letters > 0.85 and not v.lstrip().startswith(("{", "[", "/", "http")):
            good += 1
    return good / max(len(values), 1)


def find_text_columns(db_path: str) -> list[tuple[str, str, float, int]]:
    """[(table, column, score, rows)] for text columns that look like dictated sentences."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = []
    try:
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
        for t in tables:
            cols = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
            for c in cols:
                try:
                    vals = [r[0] for r in con.execute(f'select "{c}" from "{t}" where "{c}" is not null limit 300')]
                except sqlite3.Error:
                    continue
                if len(vals) < 5:
                    continue
                score = _looks_like_dictation(vals)
                if score > 0.6:
                    rows = con.execute(f'select count(*) from "{t}" where "{c}" is not null').fetchone()[0]
                    out.append((t, c, score, rows))
    finally:
        con.close()
    # Prefer the final, formatted text over raw ASR when both exist
    prio = lambda name: (0 if re.search(r"format|final|edit|clean", name, re.I) else
                         2 if re.search(r"asr|raw|transcri", name, re.I) else 1)
    return sorted(out, key=lambda x: (prio(x[1]), -x[2], -x[3]))


def read_column(db_path: str, table: str, column: str) -> list[str]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(f'select "{column}" from "{table}" where "{column}" is not null')
                if isinstance(r[0], str)]
    finally:
        con.close()


def import_wispr(folder: str = WISPR_DIR) -> dict:
    dbs = [p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
           if p.endswith((".sqlite", ".db", ".sqlite3")) and os.path.isfile(p)]
    if not dbs:
        raise FileNotFoundError(f"No Wispr Flow database found in {folder}")
    best = None
    for db in dbs:
        try:
            cols = find_text_columns(db)
        except sqlite3.Error:
            continue
        if cols and (best is None or cols[0][3] > best[1][3]):
            best = (db, cols[0])
    if best is None:
        raise RuntimeError("Found Wispr Flow's database but no column that looks like dictated text")
    db, (table, column, _, _) = best
    texts = read_column(db, table, column)
    return save_phrases(texts) | {"source": f"{os.path.basename(db)} → {table}.{column}"}


def save_phrases(texts: list[str]) -> dict:
    seen, phrases = set(), []
    for t in texts:
        for line in re.split(r"[\r\n]+", t):
            line = line.strip()
            if len(words_of(line)) >= 2 and line.lower() not in seen:
                seen.add(line.lower())
                phrases.append(line)
    os.makedirs(DIR, exist_ok=True)
    with open(PHRASES, "w", encoding="utf-8") as f:
        f.write("\n".join(phrases) + "\n")
    added = suggest_words(phrases)
    return {"phrases": len(phrases), "words": sum(len(words_of(p)) for p in phrases), "new_names": added}


def suggest_words(phrases: list[str], min_count: int = 3) -> list[str]:
    """Words you capitalise mid-sentence again and again (names, products) → words.txt."""
    caps, lower = Counter(), Counter()
    for p in phrases:
        toks = re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", p)
        for i, w in enumerate(toks):
            if w[0].isupper() and i > 0 and not w.isupper() and w.lower() not in ("i", "i'm", "i'll", "i've", "i'd"):
                caps[w] += 1
            elif w.islower():
                lower[w] += 1
    have = {w.lower() for w in vocab.load()}
    new = [w for w, n in caps.most_common(200)
           if n >= min_count and lower[w.lower()] < n and w.lower() not in have]
    if new:
        with open(vocab.PATH, "a", encoding="utf-8") as f:
            f.write("\n# from your Wispr Flow history (delete any that are wrong)\n" + "\n".join(new) + "\n")
    return new


# -- use ------------------------------------------------------------------------------
class Personal:
    """Bigram model + nearest-phrase lookup over your own phrases."""

    def __init__(self, path: str = PHRASES):
        self.phrases: list[str] = []
        if os.path.exists(path):
            self.phrases = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
        self.uni, self.bi = Counter(), Counter()
        self.index = []
        for p in self.phrases:
            w = ["<s>"] + words_of(p)
            self.uni.update(w)
            self.bi.update(zip(w, w[1:]))
            self.index.append(set(w[1:]))
        self.total = sum(self.uni.values())
        self.df = Counter(x for s in self.index for x in s)

    def __bool__(self):
        return bool(self.phrases)

    _STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been am i
        you he she it we they me my your our their this that these those do does did have has had
        not no so just can will would could should there here what when where who how why all
        about up out then than too very really also like get got go going im it's i'm don't""".split())

    def common_words(self, n: int = 120) -> list[str]:
        """Your most-used distinctive words (function words dropped), for the model's prompt."""
        return [w for w, _ in self.uni.most_common(n + 200) if w != "<s>" and w not in self._STOP
                and len(w) > 2][:n]

    def knows(self, word: str, min_count: int = 2) -> bool:
        return self.uni[word] >= min_count

    def logprob(self, text: str) -> float:
        """Average per-word log P under an interpolated bigram model (higher = more like you)."""
        w = ["<s>"] + words_of(text)
        if len(w) < 2 or not self.total:
            return 0.0
        v = len(self.uni) + 1
        lp = 0.0
        for a, b in zip(w, w[1:]):
            p_uni = (self.uni[b] + 0.1) / (self.total + 0.1 * v)
            p_bi = self.bi[(a, b)] / self.uni[a] if self.uni[a] else 0.0
            lp += math.log(0.6 * p_bi + 0.4 * p_uni)
        return lp / (len(w) - 1)

    def rerank(self, candidates: list[str], prior: float = 0.35) -> list[str]:
        """Beam order is a prior (rank r costs r*prior); your phrasing decides close calls."""
        if not self or len(candidates) < 2:
            return candidates
        scored = [(self.logprob(c) - i * prior, i, c) for i, c in enumerate(candidates)]
        return [c for _, _, c in sorted(scored, key=lambda x: (-x[0], x[1]))]

    def similar(self, text: str, k: int = 3) -> list[str]:
        """Your past sentences that share the most (rarity-weighted) words with `text`."""
        q = set(words_of(text))
        if not self or not q:
            return []
        n = len(self.index)
        scores = []
        for i, s in enumerate(self.index):
            common = q & s
            if common:
                scores.append((sum(math.log(n / self.df[x]) for x in common), i))
        scores.sort(reverse=True)
        return [self.phrases[i] for sc, i in scores[:k] if sc > 2.0]
