"""Known-sample CNVSRC crop diagnostics, never an acceptance evaluation.

Every predeclared transform is applied to every labelled mouth ROI. References
are used only after inference to score the entire transform, never to choose a
different crop per sentence. No audio is read and no LLM or personal model runs.
Supply the same research-only author sources/weights as evaluate_cnvsrc.py.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from lipflow.evaluation import character_error, characters, load_manifest


@dataclass(frozen=True)
class CropTransform:
    name: str
    scale: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    mirror: bool = False
    grayscale: str = 'opencv_uint8'


def diagnostic_grid() -> tuple[CropTransform, ...]:
    """Fixed before examining recognition output; no adaptive/oracle selection."""
    return (
        CropTransform('baseline'),
        CropTransform('scale_0.8', scale=.8),
        CropTransform('scale_1.2', scale=1.2),
        CropTransform('scale_1.4', scale=1.4),
        CropTransform('translate_x_minus6', dx=-6),
        CropTransform('translate_x_plus6', dx=6),
        CropTransform('translate_y_minus6', dy=-6),
        CropTransform('translate_y_plus6', dy=6),
        CropTransform('translate_y_minus12', dy=-12),
        CropTransform('translate_y_plus12', dy=12),
        CropTransform('mirror', mirror=True),
        CropTransform('official_float_grayscale', grayscale='torchvision_float'),
    )


def transform_frames(bgr: np.ndarray, transform: CropTransform) -> np.ndarray:
    """T,96,96,3 BGR -> T,96,96 grayscale in the original 0..255 scale.

    Affines move image content, not the crop window; positive dy moves lips down.
    Reflect padding is synthetic context and cannot recover missing source pixels.
    The reader still performs its normal center-88 crop and normalization.
    """
    if bgr.ndim != 4 or bgr.shape[1:] != (96, 96, 3) or bgr.dtype != np.uint8 or not len(bgr):
        raise ValueError('Expected nonempty T,96,96,3 uint8 mouth images')
    if not all(math.isfinite(value) for value in (transform.scale, transform.dx, transform.dy)) or transform.scale <= 0:
        raise ValueError('Transform scale must be positive; affine parameters must be finite')
    if transform.grayscale == 'opencv_uint8':
        gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in bgr])
    elif transform.grayscale == 'torchvision_float':
        # Published test pipeline divides RGB by255 before torchvision.Grayscale.
        # Retain its float precision instead of rounding back into uint8 pixels.
        rgb = bgr[..., ::-1].astype(np.float32) / 255.0
        gray = (rgb[..., 0] * .2989 + rgb[..., 1] * .5870 + rgb[..., 2] * .1140) * 255.0
    else:
        raise ValueError('Unknown grayscale conversion')
    if transform.mirror:
        gray = np.ascontiguousarray(gray[:, :, ::-1])
    if (transform.scale, transform.dx, transform.dy) != (1.0, 0.0, 0.0):
        center = 47.5
        matrix = np.array([[transform.scale, 0, (1 - transform.scale) * center + transform.dx],
                           [0, transform.scale, (1 - transform.scale) * center + transform.dy]], dtype=np.float32)
        gray = np.stack([cv2.warpAffine(frame, matrix, (96, 96), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101) for frame in gray])
    return gray


def load_images(video: str, start: float, end: float | None, reader):
    """Image-only loader; its API has no reference text argument."""
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid video frame rate')
        if start:
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        frames, stamps = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = start + len(stamps) / fps
            if end is not None and timestamp >= end:
                break
            if frame.shape != (96, 96, 3):
                raise ValueError('Crop diagnostics require already cropped96x96 video')
            frames.append(frame)
            stamps.append(timestamp)
        if not frames:
            raise ValueError('No selected images')
        indices = reader.resample(stamps, len(stamps))
        return np.stack([frames[i] for i in indices]), len(frames) / fps, fps
    finally:
        cap.release()


def summarize(rows: list[dict]) -> dict:
    """Failure/blank rows remain in the denominator; never select crops by label."""
    errors = sum(row['character_errors'] for row in rows)
    count = sum(row['reference_characters'] for row in rows)
    return {'samples': len(rows), 'character_errors': errors, 'reference_characters': count,
            'cer': errors / count if count else None,
            'exact_sentences': sum(row['exact'] for row in rows),
            'inference_failures': sum(bool(row.get('error')) for row in rows)}


def run(args):
    import torch
    from evaluate_cnvsrc import CHECKPOINT_SHA256, SOURCE_REVISION, _reader

    dataset = load_manifest(args.manifest)
    if not dataset.samples or any(not sample.mouth_roi for sample in dataset.samples):
        raise ValueError('This diagnostic requires a nonempty manifest consisting only of mouth_roi samples')
    transforms = diagnostic_grid()
    torch.set_num_threads(4)
    reader = _reader(Path(args.checkpoint), Path(args.source_dir), args.beam_size, args.ctc_weight, args.device)
    reader.warmup()
    # Read once; no transformed media are persisted or redistributed.
    images = {}
    for sample in dataset.samples:
        try:
            images[sample.id] = load_images(sample.video, sample.start, sample.end, reader)
        except Exception as exc:
            images[sample.id] = exc
    report = {'schema_version': 1, 'purpose': 'known_sample_crop_sensitivity',
              'readiness': {'ready': False, 'status': 'development_diagnostic_not_acceptance'},
              'model': {'source_revision': SOURCE_REVISION, 'checkpoint_sha256': CHECKPOINT_SHA256,
                        'beam_size': args.beam_size, 'ctc_weight': args.ctc_weight,
                        'external_lm': False, 'audio_used': False},
              'predeclared_grid': [asdict(transform) for transform in transforms], 'transforms': [],
              'limitations': ['Every transform reuses the same known labelled samples; independent acceptance requires new data.',
                              'No per-sample reference oracle, transcription correction, training, or best-crop input mode is implemented.',
                              'Existing96x96 assets do not retain full-face alignment landmarks or missing surrounding pixels.',
                              'Reflect padding in affine diagnostics is synthetic context; it is not a recovered source crop.']}
    for transform in transforms:
        rows = []
        for sample in dataset.samples:
            started = time.monotonic()
            error, raw, hypotheses, duration, fps = None, '', [], None, None
            try:
                loaded = images[sample.id]
                if isinstance(loaded, Exception):
                    raise loaded
                bgr, duration, fps = loaded
                rois = transform_frames(bgr, transform)
                hypotheses = reader.hypotheses(reader.encode(rois), nbest=5)
                raw = hypotheses[0].text if hypotheses else ''
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            elapsed = time.monotonic() - started
            # Labels first enter after image inference is complete.
            errors, count = character_error(raw, sample.reference)
            rows.append({'sample_id': sample.id, 'speaker': sample.speaker, 'reference': sample.reference,
                         'raw': raw, 'error': error, 'source_fps': fps, 'duration_seconds': duration,
                         'processing_seconds_excluding_file_load': elapsed,
                         'character_errors': errors, 'reference_characters': count,
                         'exact': characters(raw) == characters(sample.reference),
                         'hypotheses': [asdict(hypothesis) for hypothesis in hypotheses]})
        result = {'transform': asdict(transform), 'metrics': summarize(rows), 'samples': rows}
        report['transforms'].append(result)
        print(f"[{len(report['transforms'])}/{len(transforms)}] {transform.name}: {result['metrics']}", file=sys.stderr)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-research-license', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--beam-size', type=int, default=40)
    parser.add_argument('--ctc-weight', type=float, default=.5)
    args = parser.parse_args()
    if not args.accept_research_license:
        parser.error('Read VSR/LICENSE and accept the author research-only terms with --accept-research-license')
    if args.beam_size < 1 or not math.isfinite(args.ctc_weight) or not 0 <= args.ctc_weight <= 1:
        parser.error('Invalid beam size or CTC weight')
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
