"""Visual speech recognition: mouth crops -> text.

Wraps the Auto-AVSR LRS3 visual-only model (Ma et al., 2023; Apache 2.0) and its
subword RNN language model, decoded with ESPnet's batch beam search.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from espnet.asr.asr_utils import add_results_to_json, get_model_conf, torch_load  # noqa: E402
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: E402
from espnet.nets.lm_interface import dynamic_import_lm  # noqa: E402
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: E402
from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: E402

MODEL_FPS = 25
PERSONAL_LM_WEIGHT = 0.2
MODELS = os.path.join(ROOT, "models")
_MEAN, _STD = 0.421, 0.165


def pick_encoder_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LipReader:
    def __init__(self, device: str = "auto", beam_size: int = 20, lm_weight: float = 0.3,
                 ctc_weight: float = 0.1, use_lm: bool = True, personal_weight: "float | None" = None,
                 personal: bool = True, language: str = "en", model_dir: str | None = None):
        if language not in ("en", "zh"):
            raise ValueError("language must be en or zh")
        self.language = language
        model_root = model_dir or (os.path.join(MODELS, "zh") if language == "zh" else MODELS)
        # Chinese checkpoints use a different vocabulary. Never load English personal weights.

        # The 3D-conv + conformer encoder is 17x faster on the Apple GPU; beam search is
        # thousands of tiny ops and runs faster on CPU, so the two are split.
        self.enc_device = pick_encoder_device(device)
        self.device = torch.device("cuda") if self.enc_device.type == "cuda" else torch.device("cpu")
        config_path = os.path.join(model_root, "vsr", "model.json")
        if language == "zh" and not os.path.exists(config_path):
            raise FileNotFoundError("Chinese research weights missing. Read docs/CHINESE.md and run "
                                    "lipflow install-chinese --accept-research-license, or use Chinese whisper mode.")
        with open(config_path, "rb") as f:
            confs = json.load(f)
        args = argparse.Namespace(**(confs if isinstance(confs, dict) else confs[2]))
        units = os.path.join(os.path.dirname(__file__), "unigram5000_units.txt")
        self.token_list = (args.char_list if language == "zh" else
                           ["<blank>"] + [l.split()[0] for l in open(units, encoding="utf-8").read().splitlines()] + ["<eos>"])
        if language == "zh":
            args.ctc_type = "builtin"  # inference does not need the old warp-ctc extension
            if not any("\u4e00" <= t <= "\u9fff" for t in self.token_list):
                raise ValueError("Chinese checkpoint must contain a Han character vocabulary")
        odim = len(self.token_list)

        self.model = E2E(odim, args)
        state = torch.load(os.path.join(model_root, "vsr", "model.pth"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        # Your face: a fine-tuned copy of the visual model from onboarding (only the changed tensors)
        self.personal_vsr = False
        from .paths import PERSONAL_LM, personal_vsr
        pv = personal_vsr(language)
        if personal and os.path.exists(pv):
            self.model.load_state_dict(torch.load(pv, map_location="cpu", weights_only=True), strict=False)
            self.personal_vsr = True
        self.model.to(self.device).eval()
        self.model.encoder.to(self.enc_device)

        scorers = self.model.scorers()
        lm = None
        if use_lm and lm_weight > 0:
            lm_path = os.path.join(model_root, "lm", "model.pth")
            lm_args = get_model_conf(lm_path, os.path.join(model_root, "lm", "model.json"))
            lm_class = dynamic_import_lm(getattr(lm_args, "model_module", "default"), lm_args.backend)
            lm = lm_class(odim, lm_args)
            torch_load(lm_path, lm)
            lm.eval()
        scorers["lm"] = lm
        # How you talk: a copy of the LM fine-tuned on your phrases (lipflow train-lm), scored
        # alongside the general one so the search still knows ordinary English.
        plm = None
        pl = PERSONAL_LM
        if lm is not None and personal and language == "en" and os.path.exists(pl):
            plm = lm_class(odim, lm_args)
            plm.load_state_dict(torch.load(pl, map_location="cpu", weights_only=True))
            plm.eval()
        self.personal_lm = plm is not None
        pw = PERSONAL_LM_WEIGHT if personal_weight is None else personal_weight
        if plm is not None and pw > 0:
            scorers["plm"] = plm
        scorers["length_bonus"] = LengthBonus(odim)
        # CMLR's published decoding recipe includes a 0.3 character-length bonus.
        # Keep the English recipe unchanged: the two checkpoints were evaluated
        # with different decoding configurations.
        length_bonus = 0.3 if language == "zh" else 0.0
        self.beam = BatchBeamSearch(
            beam_size=beam_size,
            vocab_size=odim,
            weights=dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=lm_weight if lm else 0.0, length_bonus=length_bonus,
                         **({"plm": pw} if "plm" in scorers else {})),
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=self.token_list,
            pre_beam_score_key="decoder",
        ).to(self.device).eval()

    @staticmethod
    def resample(timestamps: list[float], n_frames: int) -> list[int]:
        """Indices that turn a variable-rate capture into a steady 25 fps sequence."""
        if n_frames == 0:
            return []
        t = np.asarray(timestamps, dtype=np.float64)
        if t.ndim != 1 or len(t) != n_frames or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
            raise ValueError("timestamps must match the frames and be finite and sorted")
        grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * MODEL_FPS + 1e-6)) + 1) / MODEL_FPS
        right = np.clip(np.searchsorted(t, grid), 0, n_frames - 1)
        left = np.maximum(right - 1, 0)
        # Choose the nearest capture. Always choosing the frame to the right skips
        # frames even at exactly 25 fps when floating point rounding shifts a grid
        # timestamp a fraction above its corresponding capture timestamp.
        indices = np.where(grid - t[left] <= t[right] - grid, left, right)
        return indices.tolist()

    @staticmethod
    def to_tensor(rois: np.ndarray) -> torch.Tensor:
        """(T, 96, 96) uint8 -> (C=1, T, 88, 88) normalised; encode() adds the batch dim."""
        x = torch.from_numpy(rois).float() / 255.0
        o = (x.shape[-1] - 88) // 2
        x = x[:, o:o + 88, o:o + 88]
        return ((x - _MEAN) / _STD).unsqueeze(0)

    @torch.inference_mode()
    def encode(self, rois: np.ndarray) -> torch.Tensor:
        # Tried and measured on the benchmark, no gain: averaging with the mirrored clip (36.3% ->
        # 37.4% WER), beam 20, CTC weight 0.2/0.3, LM weight 0.2-0.6. Stock decoding stays.
        x = self.to_tensor(rois).unsqueeze(0).to(self.enc_device)
        enc, _ = self.model.encoder(x, None)
        return enc.squeeze(0).to(self.device)

    @torch.inference_mode()
    def greedy(self, enc: torch.Tensor) -> str:
        """CTC best path: ~1 ms, used for the live preview while you are still talking."""
        ids = self.model.ctc.argmax(enc.unsqueeze(0))[0].tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i != 0:
                out.append(self.token_list[i])
            prev = i
        return self._clean("".join(out))

    @torch.inference_mode()
    def hypotheses(self, enc: torch.Tensor, nbest: int = 5):
        """Keep decoder evidence for routing. These scores are not correctness probabilities."""
        from .confidence import Hypothesis
        # The attention optimization caches encoder-memory projections by pointer.
        # Allocators can reuse that pointer for the next utterance of equal length.
        # Its lifetime must be one beam search, not the reader's whole session.
        # AVReader replaces the beam's decoder with its audiovisual decoder.
        # Clear the active scorer, which can differ from self.model.decoder.
        decoder = self.beam.full_scorers.get("decoder", self.model.decoder)
        for module in decoder.modules():
            if hasattr(module, "_mem_kv"):
                del module._mem_kv
        texts, out = set(), []
        for h in self.beam(enc):
            text = self._clean(add_results_to_json([h.asdict()], self.token_list))
            if text in texts or not text:
                continue
            texts.add(text)
            out.append(Hypothesis(text, float(h.score), max(len(h.yseq) - 2, 1)))
            if len(out) >= nbest:
                break
        return out

    def beam_search(self, enc: torch.Tensor, nbest: int = 1) -> "str | list[str]":
        texts = [h.text for h in self.hypotheses(enc, nbest)]
        return texts if nbest > 1 else (texts[0] if texts else "")

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.replace("▁", " ").replace("<space>", " ").replace("<eos>", "").split())

    def read(self, rois: np.ndarray, fast: bool = False) -> tuple[str, float]:
        """Return (UPPERCASE transcript, seconds spent)."""
        t0 = time.time()
        enc = self.encode(rois)
        text = self.greedy(enc) if fast else self.beam_search(enc)
        return text, time.time() - t0

    def warmup(self):
        self.read(np.zeros((25, 96, 96), dtype=np.uint8))
