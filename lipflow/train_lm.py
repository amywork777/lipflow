"""`lipflow train-lm`: fine-tune the decoder's language model on how you talk.

The beam search already uses a 16-layer subword Transformer LM trained on LRS3/TED text. This
makes a personal copy fine-tuned on your imported phrases (phrases.txt). The copy runs next to the
original as an extra scorer, so the decoder still knows general English but leans towards your
phrasing word by word, not just when picking among finished guesses.

Guard against forgetting: perplexity is measured on your held-out phrases *and* on general text
before and after; the model is only saved if yours improves and general text doesn't get much worse.
"""
from __future__ import annotations

import math
import os
import random
import re
import time

import torch

from .personal import PHRASES
from .vsr import MODELS, apple_gpu

from .paths import PERSONAL_LM
SPM = os.path.join(MODELS, "lm", "unigram5000.model")
GENERAL_SAMPLE = [  # TED/news-style text, for the forgetting check (from the public-domain test clips)
    "SINCE THE DAYS OF GEORGE WASHINGTON PRESIDENTS HAVE DELIVERED SOME FORM OF FINAL MESSAGE WHILE IN OFFICE",
    "A FAREWELL ADDRESS TO THE AMERICAN PEOPLE",
    "ON TUESDAY NIGHT IN CHICAGO I'LL DELIVER MINE",
    "I CHOSE CHICAGO NOT ONLY BECAUSE IT'S MY HOMETOWN WHERE I MET MY WIFE AND WE STARTED A FAMILY",
    "THIS PAST WEEK WE LOST AN AMERICAN ICON AND ONE OF THE MOST INFLUENTIAL FIGURES OF HER TIME",
    "BORN IN NEW YORK CITY AND RAISED MOSTLY IN CHICAGO NANCY DAVIS GRADUATED FROM SMITH COLLEGE",
    "AS AN ACTRESS SHE APPEARED IN ELEVEN FILMS",
    "AND OFF SCREEN SHE STARRED IN A REAL LIFE HOLLYWOOD ROMANCE WITH THE LOVE OF HER LIFE",
    "I THINK THE MOST IMPORTANT THING IS TO KEEP LEARNING EVERY SINGLE DAY",
    "WHEN WE STARTED THIS PROJECT NOBODY BELIEVED IT WOULD WORK",
    "THE QUESTION IS NOT WHETHER WE CAN DO IT BUT WHETHER WE SHOULD",
    "AND THAT IS WHY I WANT TO TALK TO YOU ABOUT WATER",
]


def normalise(line: str) -> list[str]:
    """Your text → LRS3-style sentences: upper case, letters and apostrophes, no digits."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", line):
        if re.search(r"\d", sent):
            continue  # LRS3 spells numbers out; skip rather than guess
        s = re.sub(r"[^A-Za-z' ]+", " ", sent).upper()
        s = " ".join(w.strip("'") for w in s.split() if w.strip("'"))
        if len(s.split()) >= 2:
            out.append(s)
    return out


class Tok:
    def __init__(self, token_list: list[str]):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=SPM)
        self.id = {t: i for i, t in enumerate(token_list)}
        self.unk = self.id["<unk>"]

    def __call__(self, s: str) -> list[int]:
        return [self.id.get(p, self.unk) for p in self.sp.encode(s, out_type=str)]


def _batches(seqs, eos, bs, shuffle=True):
    seqs = list(seqs)
    if shuffle:
        random.shuffle(seqs)
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        n = max(len(s) for s in chunk) + 1
        x = torch.zeros(len(chunk), n, dtype=torch.long)
        t = torch.zeros(len(chunk), n, dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, :len(s) + 1] = torch.tensor([eos] + s)
            t[j, :len(s) + 1] = torch.tensor(s + [eos])
        yield x, t


@torch.no_grad()
def perplexity(lm, seqs, eos, device) -> float:
    lm.eval()
    nll, n = 0.0, 0
    for x, t in _batches(seqs, eos, 64, shuffle=False):
        _, s_nll, count = lm(x.to(device), t.to(device))
        nll += float(s_nll)
        n += int(count)
    return math.exp(nll / max(n, 1))


def train(epochs: int = 3, lr: float = 3e-5, bs: int = 32, max_general_loss: float = 1.15, seed: int = 0) -> dict:
    from espnet.asr.asr_utils import get_model_conf, torch_load
    from espnet.nets.lm_interface import dynamic_import_lm
    from .vsr import LipReader

    random.seed(seed)
    torch.manual_seed(seed)
    if not os.path.exists(PHRASES):
        raise FileNotFoundError("No phrases yet. Run `lipflow import-wispr` first.")
    units = os.path.join(os.path.dirname(__file__), "unigram5000_units.txt")
    token_list = ["<blank>"] + [l.split()[0] for l in open(units, encoding="utf-8").read().splitlines()] + ["<eos>"]
    eos = len(token_list) - 1
    tok = Tok(token_list)

    sents = sorted({s for line in open(PHRASES, encoding="utf-8") for s in normalise(line)})
    random.shuffle(sents)
    seqs = [tok(s)[:58] for s in sents]
    n_val = max(20, len(seqs) // 10)
    val, tr = seqs[:n_val], seqs[n_val:]
    general = [tok(s) for s in GENERAL_SAMPLE]

    lm_path = os.path.join(MODELS, "lm", "model.pth")
    args = get_model_conf(lm_path, os.path.join(MODELS, "lm", "model.json"))
    lm = dynamic_import_lm(getattr(args, "model_module", "default"), args.backend)(len(token_list), args)
    torch_load(lm_path, lm)
    device = torch.device("mps" if apple_gpu() else "cpu")
    lm.to(device)

    before = {"yours": perplexity(lm, val, eos, device), "general": perplexity(lm, general, eos, device)}
    opt = torch.optim.AdamW(lm.parameters(), lr=lr, weight_decay=0.01)
    t0 = time.time()
    best, best_state = before["yours"], None
    for ep in range(epochs):
        lm.train()
        for x, t in _batches(tr, eos, bs):
            loss, _, _ = lm(x.to(device), t.to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
        p_y, p_g = perplexity(lm, val, eos, device), perplexity(lm, general, eos, device)
        print(f"  epoch {ep + 1}: your held-out phrases {p_y:.1f}, general text {p_g:.1f}  ({time.time() - t0:.0f}s)")
        if p_y < best and p_g <= before["general"] * max_general_loss:
            best, best_state = p_y, {k: v.detach().cpu().clone() for k, v in lm.state_dict().items()}
    after = {"yours": best, "general": None}
    saved = best_state is not None
    if saved:
        os.makedirs(os.path.dirname(PERSONAL_LM), exist_ok=True)
        torch.save(best_state, PERSONAL_LM)
        lm.load_state_dict(best_state)
        after["general"] = perplexity(lm, general, eos, device)
    return {"sentences": len(sents), "train": len(tr), "val": len(val), "before": before, "after": after,
            "saved": saved, "seconds": time.time() - t0}
