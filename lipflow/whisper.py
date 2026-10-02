"""Optional Chinese quiet-speech ASR. Mouth quality gates it; this is NOT neural AV fusion."""
import math
import numpy as np
from .confidence import Hypothesis


class ChineseWhisper:
    def __init__(self, model='large-v3-turbo', device='cpu'):
        self.name, self.device, self.model = model, device, None

    def load(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError('For Chinese whisper input run: uv sync --extra chinese-whisper') from e
            self.model = WhisperModel(self.name, device=self.device, compute_type='int8')
        return self.model

    def hypotheses(self, wave):
        if wave is None or len(wave) < 9600 or not np.all(np.isfinite(wave)):
            return []
        if np.sqrt(np.mean(np.square(wave))) < 1e-5:
            return []  # silence must not produce invented dictation
        self.load()
        # Quiet speech can be rejected by voiced-speech VAD. Amplify a bounded amount,
        # then let the decoder's no-speech check reject empty output (always reviewed).
        wave = np.asarray(wave, dtype=np.float32)
        wave = wave * min(32.0, 0.3 / (float(np.max(np.abs(wave))) + 1e-8))
        segments, _ = self.model.transcribe(wave, language='zh', task='transcribe',
            beam_size=5, vad_filter=False, condition_on_previous_text=False)
        parts, score, n = [], 0.0, 0
        for seg in segments:
            if seg.no_speech_prob > 0.6 or not math.isfinite(seg.avg_logprob):
                continue
            parts.append(seg.text.strip())
            count = max(len(seg.tokens), 1)
            score += seg.avg_logprob * count
            n += count
        from opencc import OpenCC
        text = OpenCC('t2s').convert(''.join(parts))
        # Single-best ASR has no reliable n-best margin: always review in the confidence router.
        return [Hypothesis(text, score, n)] if text else []
