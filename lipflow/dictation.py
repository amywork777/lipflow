"""The parts of a dictation that don't depend on the OS: settings, clip checks, history, kept clips,
and training on your face. app.py (macOS) and win/app.py (Windows) wrap these in their own UI."""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .paths import HOME

HISTORY = os.path.join(HOME, "history.jsonl")
SETTINGS = os.path.join(HOME, "settings.json")
MIN_SECONDS = 0.6
MAX_SECONDS = 60.0
PREVIEW_EVERY = 0.45
KEEP_CLIPS = 100  # recent dictation clips kept (96x96 grayscale mouth crops, no audio)
TAIL_SECONDS = 0.4  # keep filming after release: the last word needs the frames after it
JOIN_WINDOW = 45.0  # dictations this close together get a separating space


def load_settings() -> dict:
    try:
        return json.load(open(SETTINGS))
    except (OSError, ValueError):
        return {}


def save_settings(d: dict):
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    json.dump(d, open(SETTINGS, "w"), indent=2)


def clip_problem(rec) -> "tuple[str, str] | None":
    """Why a recording can't be read, as (title, advice), or None if it's fine."""
    if rec.duration < MIN_SECONDS or len(rec.ts) < 12:
        return ("Too short", "Hold the key while you mouth the words")
    if rec.face_ratio < 0.4:
        return ("Can't see your face", "Face the camera with your mouth in view")
    if np.std([m for m in rec.mouth_open if m > 0] or [0]) < 0.012:
        return ("No lip movement", "Mouth the words clearly — no sound needed")
    return None


def rois_for(rec):
    from .face import mouth_rois
    from .vsr import LipReader
    ts, grays, anchors = rec.snapshot()
    idx = LipReader.resample(ts, len(ts))
    return mouth_rois([grays[i] for i in idx], [anchors[i] for i in idx])


def log_history(rec, candidates, text, secs, cleanup: str, evidence=None):
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    with open(HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(rec.duration, 2),
                            "raw": candidates, "text": text, "latency": round(secs, 2),
                            "cleanup": cleanup, **({"evidence": evidence} if evidence else {})}) + "\n")


def keep_clip(rois, candidates, text, settings: dict):
    """Keep the mouth crops of recent dictations (local only, last KEEP_CLIPS) so accuracy changes
    can be measured on your real clips. Off switch in Settings."""
    if not settings.get("save_clips", True):
        return
    d = os.path.join(os.path.dirname(HISTORY), "clips", "dictations")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{int(time.time() * 1000)}.npz"), rois=rois,
                        raw=np.array(candidates), text=text)
    # Only the recent ones are useful (to measure accuracy on your real dictations): keep 100.
    for f in sorted(os.listdir(d))[:-KEEP_CLIPS]:
        os.remove(os.path.join(d, f))


def train_on_face(beam: int, report, language="en") -> dict:
    """Personal LM (if you imported phrases), then face adaptation with a held-out check.

    report(pct, text) gets progress. Returns {"before", "after", "kept", "clips", "note"};
    "after" is None when there weren't enough clips to train on.
    """
    import random as _r
    from . import corrections
    from .bench import wer
    from .personal import PHRASES
    from .practice import N_HELD_OUT, saved_clips
    from .train_vsr import finetune, save
    from .vsr import LipReader
    note = ""
    if language == "en" and os.path.exists(PHRASES):
        report(3, "Learning how you talk from your phrases…")
        from .train_lm import train as train_lm
        try:
            r = train_lm(epochs=3)
            note = (f"Your phrasing: {r['before']['yours']:.0f} → {r['after']['yours']:.0f} perplexity. "
                    if r["saved"] else "")
        except Exception as e:
            print(f"[lipflow] train-lm failed: {e}")
    clips = saved_clips(language)
    learned = corrections.load_all(language)
    if len(clips) < N_HELD_OUT + 6:
        return {"before": 0, "after": None, "kept": False, "clips": len(clips),
                "note": "Not enough practice clips to train on. Run setup again from the menu."}
    _r.Random(1).shuffle(clips)
    # held out: practice clips only (their text is certain); corrections only ever train
    test, train = clips[:N_HELD_OUT], clips[N_HELD_OUT:] + learned
    report(15, f"Measuring the standard model on {len(test)} of your sentences…")
    base = LipReader(beam_size=beam, personal=False, language=language)

    def score(reader):
        e = n = 0
        for c in test:
            a, b = wer(reader.beam_search(reader.encode(c["rois"])), c["text"])
            e, n = e + a, n + b
        return e / max(n, 1)

    before = score(base)
    report(25, f"Training on {len(train)} of your clips…")
    finetune(train, reader=base, log=lambda *a: None,
             on_epoch=lambda k, n: report(25 + 65 * k / n, f"Training on your face: pass {k} of {n}"))
    report(92, "Checking it on the sentences it didn't see…")
    after = score(base)
    kept = after < before
    if kept:
        save(base)
    print(f"[lipflow] onboarding: held-out WER {before:.1%} → {after:.1%} ({'kept' if kept else 'discarded'})")
    return {"before": before, "after": after, "kept": kept, "clips": len(clips), "note": note}
