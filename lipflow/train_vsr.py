"""Adapt the lip-reading model to one person's face from a few dozen short clips.

Only the visual front end (3D-conv + ResNet: pixels of *your* mouth → features) is trained, plus
its layer norms if asked; the Conformer encoder, decoder and CTC head keep what they learned from
thousands of hours of speech. Loss is the model's own training objective: 0.1·CTC + 0.9·attention
with label smoothing. Augmentation: random 88px crop of the 96px mouth patch, horizontal flip,
time masking, small brightness jitter.

Saved as <LIPFLOW_HOME>/models/vsr_face.pth containing only the changed tensors (~45 MB); LipReader loads it
on top of the base model. Clips come from onboarding (you mouth sentences with known text).
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .vsr import MODELS, LipReader, _MEAN, _STD

from .paths import personal_vsr


def _augment(rois: np.ndarray) -> torch.Tensor:
    """(T,96,96) uint8 → (1,T,88,88) normalised, randomly cropped/flipped/masked."""
    x = torch.from_numpy(rois).float() / 255.0
    i, j = random.randint(0, 8), random.randint(0, 8)
    x = x[:, i:i + 88, j:j + 88]
    if random.random() < 0.5:
        x = x.flip(-1)
    x = x * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06)
    T = x.shape[0]
    for _ in range(2):  # time masks of up to 0.4 s
        w = random.randint(0, min(10, T // 5))
        s = random.randint(0, max(T - w, 0))
        x[s:s + w] = x.mean()
    return ((x - _MEAN) / _STD).unsqueeze(0)


def _targets(reader: LipReader, text: str) -> list[int]:
    if getattr(reader, "language", "en") == "zh":
        from .text import tokens
        ids = {t: i for i, t in enumerate(reader.token_list)}
        chars = tokens(text)
        if not chars or any(c not in ids for c in chars):
            raise ValueError("Chinese training text must use the model's Han vocabulary; no digits or Latin letters")
        return [ids[c] for c in chars]
    from .train_lm import Tok
    if not hasattr(reader, "_tok"):
        reader._tok = Tok(reader.token_list)
    # Letters and apostrophes only, like the model's training text: punctuation would become <unk>
    import re
    clean = " ".join(re.sub(r"[^A-Za-z' ]+", " ", text).upper().split())
    return reader._tok(clean)


def clip_loss(reader: LipReader, x: torch.Tensor, ys: list[int]) -> torch.Tensor:
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    m = reader.model
    dev = reader.enc_device
    hs, _ = m.encoder(x.unsqueeze(0).to(dev), None)                    # (1, T, 768)
    # CTC (on CPU: MPS has no ctc_loss)
    logp = m.ctc.ctc_lo(hs).log_softmax(-1).transpose(0, 1).cpu()     # (T, 1, V)
    y = torch.tensor(ys, dtype=torch.long)
    ctc = F.ctc_loss(logp, y.unsqueeze(0), torch.tensor([logp.shape[0]]), torch.tensor([len(ys)]),
                     blank=0, reduction="sum", zero_infinity=True)
    # attention decoder, teacher forced
    sos = eos = m.eos
    ys_in = torch.tensor([[sos] + ys], dtype=torch.long, device=dev)
    ys_out = torch.tensor([ys + [eos]], dtype=torch.long, device=dev)
    mask = subsequent_mask(ys_in.shape[1], device=dev).unsqueeze(0)
    pred, _ = m.decoder(ys_in, mask, hs, None)
    att = m.criterion(pred, ys_out)  # label-smoothed, summed over tokens
    return (0.1 * ctc.to(dev) + 0.9 * att) / len(ys)


def trainable(reader: LipReader, scope: str = "frontend"):
    m = reader.model
    for p in m.parameters():
        p.requires_grad_(False)
    params = list(m.encoder.frontend.parameters())
    if scope in ("frontend+embed", "frontend+encoder1"):
        params += list(m.encoder.embed.parameters())
    if scope == "frontend+encoder1":
        params += list(m.encoder.encoders[0].parameters())
    for p in params:
        p.requires_grad_(True)
    return params


# Chosen on held-out speaker-adaptation runs (train on one video, test on another; WER 36.3% →
# frontend 6ep 32.2%, 12ep 30.4%, lr 3e-4 33.7%, frontend+first encoder layer 6ep 29.6%).
DEFAULT_SCOPE = "frontend+encoder1"


def finetune(clips: list[dict], epochs: int = 6, lr: float = 1e-4, scope: str = DEFAULT_SCOPE,
             reader: LipReader | None = None, log=print, seed: int = 0, on_epoch=None) -> LipReader:
    """clips: [{'rois': (T,96,96) uint8, 'text': str}]. Returns the adapted reader (in memory)."""
    random.seed(seed)
    torch.manual_seed(seed)
    reader = reader or LipReader(beam_size=10, personal=False)
    # Training on the GPU: the decoder follows the encoder there for the teacher-forced pass
    reader.model.decoder.to(reader.enc_device)
    reader.model.ctc.to(reader.enc_device)
    reader.model.criterion.to(reader.enc_device) if hasattr(reader.model.criterion, "to") else None
    params = trainable(reader, scope)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    data = [(c["rois"], _targets(reader, c["text"])) for c in clips]
    t0 = time.time()
    for ep in range(epochs):
        reader.model.train()
        # BatchNorm statistics stay frozen: a few clips would wreck them
        for mod in reader.model.modules():
            if isinstance(mod, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                mod.eval()
        random.shuffle(data)
        total = 0.0
        for rois, ys in data:
            loss = clip_loss(reader, _augment(rois), ys)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            total += float(loss.detach())
        log(f"  epoch {ep + 1}/{epochs}: loss {total / len(data):.3f}  ({time.time() - t0:.0f}s)")
        if on_epoch:
            on_epoch(ep + 1, epochs)
    reader.model.eval()
    reader.model.decoder.to(reader.device)
    reader.model.ctc.to(reader.device)
    for p in reader.model.parameters():
        p.requires_grad_(False)
    reader._trained_scope = scope
    return reader


def save(reader: LipReader, scope: str = DEFAULT_SCOPE):
    prefixes = ["encoder.frontend."]
    if scope in ("frontend+embed", "frontend+encoder1"):
        prefixes.append("encoder.embed.")
    if scope == "frontend+encoder1":
        prefixes.append("encoder.encoders.0.")
    state = {k: v.detach().cpu() for k, v in reader.model.state_dict().items() if k.startswith(tuple(prefixes))}
    path = personal_vsr(getattr(reader, "language", "en"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    return path
