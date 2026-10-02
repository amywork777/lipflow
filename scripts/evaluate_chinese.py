"""Local Mandarin evaluation; no included datasets and no cloud cleanup.

Single clip (backward compatible):
    uv run python scripts/evaluate_chinese.py clip.mp4 --reference '今天下雨'
Independent batch, exit 2 if the declared readiness criteria are not met:
    uv run python scripts/evaluate_chinese.py --manifest test.json --output report.json
The batch protocol measures pure visual input, with a preloaded/warmed model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from lipflow.evaluation import Prediction, Thresholds, character_error, evaluate, load_manifest


class ClipRejected(Exception):
    """A live pre-inference rejection, rather than a decoder failure."""

    def __init__(self, reason, duration):
        super().__init__(reason)
        self.duration = duration


def _visual_input(sample, reader):
    """Use the live application's tracker, alignment, sampling and quality rules."""
    import cv2
    import numpy as np
    from lipflow.confidence import Quality
    from lipflow.face import FaceTracker, mouth_rois

    cap = cv2.VideoCapture(sample.video)
    tracker = None
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot establish video frame rate/duration')
        if sample.start:
            cap.set(cv2.CAP_PROP_POS_MSEC, sample.start * 1000)
        if not sample.mouth_roi:
            tracker = FaceTracker()
        timestamps, grays, anchors, mouth_pixels, motion = [], [], [], [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = sample.start + len(timestamps) / fps
            if sample.end is not None and timestamp >= sample.end:
                break
            if sample.mouth_roi and frame.shape[:2] != (96, 96):
                raise ValueError('mouth_roi domain expects already aligned 96x96 crops')
            timestamps.append(timestamp)
            grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if tracker:
                observation = tracker.detect(frame, int(timestamp * 1000))
                anchors.append(observation.anchors if observation else None)
                mouth_pixels.append(float(np.ptp(observation.outer_lips[:, 0])) if observation else 0.0)
                motion.append(observation.mouth_open if observation else 0.0)
        if not timestamps:
            raise ValueError('No frames in selected range')
        duration = len(timestamps) / fps
        if tracker:
            from lipflow.dictation import clip_problem
            recording = SimpleNamespace(ts=timestamps, duration=timestamps[-1] - timestamps[0],
                                        face_ratio=sum(a is not None for a in anchors) / len(anchors),
                                        mouth_open=motion)
            problem = clip_problem(recording)
            if problem:
                raise ClipRejected(': '.join(problem), duration)
        indices = reader.resample(timestamps, len(timestamps))
        if sample.mouth_roi:
            rois = np.stack([grays[i] for i in indices])
            # These values describe an already cropped asset, not the original
            # camera. Such clips can only supply a smoke test to the default gate.
            quality = Quality(brightness=float(np.mean(rois)), contrast=float(np.std(rois)), mouth_pixels=96)
        else:
            rois = mouth_rois([grays[i] for i in indices], [anchors[i] for i in indices])
            if rois is None:
                raise ValueError('No face found in selected range')
            quality = Quality(sum(a is not None for a in anchors) / len(anchors),
                              float(np.mean(rois)), float(np.std(rois)), float(np.median(mouth_pixels)))
        return rois, duration, quality
    finally:
        cap.release()
        if tracker is not None:
            tracker.close()


def _batch(args):
    from lipflow.confidence import assess
    from lipflow.vsr import LipReader

    dataset = load_manifest(args.manifest)
    thresholds = Thresholds(
        min_samples=args.min_samples, min_speakers=args.min_speakers,
        min_sessions_per_speaker=args.min_sessions_per_speaker,
        max_cer=args.max_cer, min_exact_sentence_rate=args.min_exact_sentence_rate,
        min_non_rejected_coverage=args.min_non_rejected_coverage,
        max_p95_rtf=args.max_p95_rtf, max_p95_processing_seconds=args.max_p95_processing_seconds,
        require_webcam_domain=not args.allow_non_webcam_domain,
        require_silent_articulation=not args.allow_voiced_articulation)
    predictions, startup = [], time.monotonic()
    if dataset.samples:
        # References are kept in the scoring layer only. No phrase retrieval,
        # personal model, cleanup, prompt or LM adaptation sees these labels.
        reader = LipReader(language='zh', beam_size=args.beam_size, personal=False, device=args.device,
                           model_dir=args.model_dir)
        reader.warmup()
    startup_seconds = time.monotonic() - startup
    for sample in dataset.samples:
        started, duration = time.monotonic(), None
        try:
            rois, duration, quality = _visual_input(sample, reader)
            enc = reader.encode(rois)
            hypotheses = reader.hypotheses(enc, nbest=5)
            greedy = reader.greedy(enc)
            decision = assess(hypotheses, greedy, quality, policy='auto', language='zh')
            raw = hypotheses[0].text if hypotheses else ''
            prediction = Prediction(sample.id, raw, duration, time.monotonic() - started,
                                    decision.action, decision.reason, margin=decision.margin)
        except ClipRejected as exc:
            prediction = Prediction(sample.id, '', exc.duration, time.monotonic() - started,
                                    'retry', reason=str(exc))
        except Exception as exc:
            # A failed video stays in the CER and coverage denominators.
            prediction = Prediction(sample.id, '', duration, time.monotonic() - started, 'retry',
                                    error=f'{type(exc).__name__}: {exc}')
        predictions.append(prediction)
        print(f'[{len(predictions)}/{len(dataset.samples)}] {sample.id}: {prediction.action}', file=sys.stderr)
    report = evaluate(dataset, predictions, thresholds)
    model_root = Path(args.model_dir) if args.model_dir else Path(__file__).resolve().parents[1] / 'models' / 'zh'
    fingerprints = {}
    for name in ('vsr/model.json', 'vsr/model.pth', 'lm/model.json', 'lm/model.pth'):
        path = model_root / name
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            fingerprints[name] = digest.hexdigest()
    report['model'] = {'language': 'zh', 'beam_size': args.beam_size, 'personal': False,
                       'startup_seconds_including_warmup': startup_seconds,
                       'file_sha256': fingerprints,
                       'vocabulary_size': len(reader.token_list) if dataset.samples else None,
                       'encoder_device': str(reader.enc_device) if dataset.samples else None,
                       'decoder_device': str(reader.device) if dataset.samples else None,
                       'decoder_weights': dict(reader.beam.weights) if dataset.samples else None}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('video', nargs='?')
    p.add_argument('--reference')
    p.add_argument('--mode', choices=['silent', 'whisper'], default='silent')
    p.add_argument('--mouth-roi', action='store_true')
    p.add_argument('--whisper-model', default='large-v3-turbo')
    p.add_argument('--manifest', help='Version 1 manifest of independently labelled local videos')
    p.add_argument('--output', help='Save the same JSON report printed to stdout')
    p.add_argument('--device', default='auto')
    p.add_argument('--model-dir', help='CMLR-compatible local directory with vsr/ and lm/ checkpoints')
    p.add_argument('--beam-size', type=int, default=10)
    p.add_argument('--min-samples', type=int, default=50)
    p.add_argument('--min-speakers', type=int, default=5)
    p.add_argument('--min-sessions-per-speaker', type=int, default=2)
    p.add_argument('--max-cer', type=float, default=.10)
    p.add_argument('--min-exact-sentence-rate', type=float, default=.70)
    p.add_argument('--min-non-rejected-coverage', type=float, default=.80)
    p.add_argument('--max-p95-rtf', type=float, default=1.0)
    p.add_argument('--max-p95-processing-seconds', type=float, default=2.0)
    p.add_argument('--allow-non-webcam-domain', action='store_true',
                   help='Override the webcam criterion for a stated research domain; not webcam readiness')
    p.add_argument('--allow-voiced-articulation', action='store_true',
                   help='Research-only visual benchmark of voiced/whispered mouth movements; not silent-speech readiness')
    args = p.parse_args()
    if args.beam_size < 1:
        p.error('--beam-size must be positive')
    if args.manifest:
        if args.video or args.reference or args.mouth_roi or args.mode != 'silent':
            p.error('--manifest is pure visual; each sample declares its own path/reference/domain')
        try:
            report = _batch(args)
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            p.error(str(exc))
    else:
        if not args.video or args.reference is None:
            p.error('single-clip evaluation requires video and --reference')
        try:
            character_error('', args.reference)
        except ValueError as exc:
            p.error(str(exc))
        started = time.monotonic()
        if args.mode == 'whisper':
            from lipflow.av import load_audio
            from lipflow.whisper import ChineseWhisper
            hypotheses = ChineseWhisper(args.whisper_model).hypotheses(load_audio(args.video, 0, 60))
            raw = hypotheses[0].text if hypotheses else ''
        else:
            from lipflow.offline import transcribe_file
            from lipflow.vsr import LipReader
            raw = transcribe_file(args.video, LipReader(language='zh', beam_size=args.beam_size,
                                                        personal=False, device=args.device, model_dir=args.model_dir),
                                  mouth_roi=args.mouth_roi)
        errors, chars = character_error(raw, args.reference)
        report = {'raw': raw, 'reference': args.reference, 'character_errors': errors,
                  'reference_characters': chars, 'cer': errors / chars,
                  'seconds_including_load': time.monotonic() - started, 'mode': args.mode}
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 2 if args.manifest and not report['readiness']['ready'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
