"""Auditable Mandarin visual evaluation, independent of models and LLM cleanup.

Readiness is relative to the declared protocol and configurable thresholds. It is
not a universal accuracy guarantee or a calibrated decoder probability.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata


def characters(text: str) -> list[str]:
    """CER units: Unicode letters/numbers after NFKC, without punctuation/space.

    English is measured character by character, rather than as one word. Digits
    remain evidence; no number expansion, translation, or reference-driven fixes.
    Simplified and traditional Han are intentionally different units.
    """
    if not isinstance(text, str):
        raise ValueError("transcript must be a string")
    return [c for c in unicodedata.normalize('NFKC', text).casefold()
            if unicodedata.category(c)[0] in ('L', 'N')]


def character_error(hypothesis: str, reference: str) -> tuple[int, int]:
    h, r = characters(hypothesis), characters(reference)
    if not r:
        raise ValueError("reference must contain at least one letter or number")
    d = list(range(len(r) + 1))
    for i, a in enumerate(h, 1):
        previous, d[0] = d[0], i
        for j, b in enumerate(r, 1):
            previous, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, previous + (a != b))
    return d[-1], len(r)


def percentile(values: list[float], quantile: float) -> float | None:
    """Linear interpolation on sorted values (including p50/p95 for small sets)."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


@dataclass(frozen=True)
class Sample:
    id: str
    video: str
    reference: str
    speaker: str
    session: str
    label_source: str
    label_verified: bool
    domain: str
    mouth_roi: bool = False
    start: float = 0.0
    end: float | None = None
    sha256: str = ''
    split: str = 'test'
    articulation: str = 'unknown'  # silent / voiced / whispered / unknown
    articulation_verified: bool = False


@dataclass(frozen=True)
class Exclusion:
    """Known training or calibration identity; used to detect test leakage."""
    id: str
    speaker: str
    session: str
    sha256: str = ''


@dataclass(frozen=True)
class Dataset:
    samples: tuple[Sample, ...]
    split: str = 'test'
    training_overlap_checked: bool = False
    development_samples: tuple[Exclusion, ...] = ()
    description: str = ''


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    raw: str
    duration_seconds: float | None
    processing_seconds: float | None
    action: str = 'review'
    reason: str = ''
    error: str | None = None
    margin: float | None = None


@dataclass(frozen=True)
class Thresholds:
    min_samples: int = 50
    min_speakers: int = 5
    min_sessions_per_speaker: int = 2
    max_cer: float = 0.10
    min_exact_sentence_rate: float = 0.70
    min_non_rejected_coverage: float = 0.80
    max_p95_rtf: float = 1.0
    max_p95_processing_seconds: float = 2.0
    require_webcam_domain: bool = True
    require_silent_articulation: bool = True

    def __post_init__(self):
        for key in ('min_samples', 'min_speakers', 'min_sessions_per_speaker'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('max_cer', 'min_exact_sentence_rate', 'min_non_rejected_coverage'):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{key} must be finite and between 0 and 1')
        if not math.isfinite(self.max_p95_rtf) or self.max_p95_rtf <= 0:
            raise ValueError('max_p95_rtf must be finite and positive')
        if not math.isfinite(self.max_p95_processing_seconds) or self.max_p95_processing_seconds <= 0:
            raise ValueError('max_p95_processing_seconds must be finite and positive')


def _required_text(item: dict, key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{key} must be a nonempty string')
    return value


def _number(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key} must be a finite number')
    return float(value)


def _video(item, base: Path, hashes: dict) -> tuple[str, str]:
    path = (base / _required_text(item, 'video')).resolve()
    if not path.is_file():
        raise ValueError(f'video does not exist: {path}')
    if path not in hashes:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[path] = digest.hexdigest()
    return str(path), hashes[path]


def load_manifest(path: str | Path) -> Dataset:
    """Load local labelled files; labels are never sent to a model or cleaner.

    A missing/blank label or video is an invalid evaluation input, not a sample
    that may be silently removed from the denominator. Dataset independence is
    a documented human audit, supplemented with content/identity checks here.
    """
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('manifest must be an object with schema_version=1')
    if not isinstance(data.get('samples'), list):
        raise ValueError('manifest samples must be a list')
    checked = data.get('training_overlap_checked', False)
    if not isinstance(checked, bool):
        raise ValueError('training_overlap_checked must be boolean')
    hashes, samples, exclusions = {}, [], []
    for item in data['samples']:
        if not isinstance(item, dict):
            raise ValueError('each sample must be an object')
        fields = {key: _required_text(item, key) for key in
                  ('id', 'reference', 'speaker', 'session', 'label_source', 'domain')}
        character_error('', fields['reference'])
        if fields['domain'] not in ('mouth_roi', 'full_face', 'webcam'):
            raise ValueError('domain must be mouth_roi, full_face, or webcam')
        verified = item.get('label_verified', False)
        roi = item.get('mouth_roi', fields['domain'] == 'mouth_roi')
        articulation = item.get('articulation', 'unknown')
        articulation_verified = item.get('articulation_verified', False)
        if articulation not in ('silent', 'voiced', 'whispered', 'unknown') or not isinstance(articulation_verified, bool):
            raise ValueError('articulation must be silent/voiced/whispered/unknown with boolean articulation_verified')
        if not isinstance(verified, bool) or not isinstance(roi, bool):
            raise ValueError('label_verified and mouth_roi must be boolean')
        if roi != (fields['domain'] == 'mouth_roi'):
            raise ValueError('mouth_roi must match the declared domain')
        start = _number(item.get('start', 0), 'start')
        end = None if item.get('end') is None else _number(item['end'], 'end')
        if start < 0 or (end is not None and end <= start):
            raise ValueError('sample range must have start >= 0 and end > start')
        video, digest = _video(item, path.parent, hashes)
        split = item.get('split', data.get('split', 'test'))
        if not isinstance(split, str) or not split:
            raise ValueError('sample split must be a nonempty string')
        samples.append(Sample(**fields, video=video, sha256=digest, split=split,
                              label_verified=verified, mouth_roi=roi, start=start, end=end,
                              articulation=articulation, articulation_verified=articulation_verified))
    development = data.get('development_samples', [])
    if not isinstance(development, list):
        raise ValueError('development_samples must be a list')
    for item in development:
        fields = {key: _required_text(item, key) for key in ('id', 'speaker', 'session')}
        digest = _video(item, path.parent, hashes)[1] if 'video' in item else ''
        exclusions.append(Exclusion(**fields, sha256=digest))
    return Dataset(tuple(samples), data.get('split', 'test'), checked, tuple(exclusions),
                   data.get('description', ''))


def independence_issues(dataset: Dataset) -> list[str]:
    issues = []
    if dataset.split != 'test' or any(sample.split != 'test' for sample in dataset.samples):
        issues.append('Evaluation split must be test, independent of tuning/calibration.')
    if not dataset.training_overlap_checked:
        issues.append('Training/calibration overlap audit is not declared complete.')
    if any(not sample.label_verified or not sample.label_source for sample in dataset.samples):
        issues.append('Every reference needs independent verified label provenance.')
    counts = Counter(sample.id for sample in dataset.samples)
    if any(count > 1 for count in counts.values()):
        issues.append('Duplicate evaluation sample IDs.')
    for i, a in enumerate(dataset.samples):
        for b in dataset.samples[i + 1:]:
            same_content = (a.sha256 and a.sha256 == b.sha256) or a.video == b.video
            if same_content and max(a.start, b.start) < min(a.end or math.inf, b.end or math.inf):
                issues.append(f'Overlapping/duplicate video ranges: {a.id}, {b.id}.')
    development_speakers = {sample.speaker for sample in dataset.development_samples}
    development_ids = {sample.id for sample in dataset.development_samples}
    development_hashes = {sample.sha256 for sample in dataset.development_samples if sample.sha256}
    for sample in dataset.samples:
        if (sample.speaker in development_speakers or sample.id in development_ids
                or sample.sha256 and sample.sha256 in development_hashes):
            issues.append(f'Training/calibration identity or content overlaps evaluation: {sample.id}.')
    return issues


def evaluate(dataset: Dataset, predictions: list[Prediction], thresholds: Thresholds | None = None,
             *, mode: str = 'silent') -> dict:
    thresholds = thresholds or Thresholds()
    issues = independence_issues(dataset)
    by_id = {}
    for prediction in predictions:
        if prediction.sample_id in by_id:
            issues.append(f'Duplicate prediction: {prediction.sample_id}.')
        by_id[prediction.sample_id] = prediction
    unknown = set(by_id) - {sample.id for sample in dataset.samples}
    if unknown:
        issues.append('Predictions not present in manifest: ' + ', '.join(sorted(unknown)))
    rows, seconds, rtfs = [], [], []
    sessions = defaultdict(set)
    totals = Counter()
    for sample in dataset.samples:
        sessions[sample.speaker].add(sample.session)
        prediction = by_id.get(sample.id)
        if prediction is None:
            prediction = Prediction(sample.id, '', None, None, 'retry', error='Missing prediction')
        if prediction.action not in ('retry', 'review', 'auto'):
            raise ValueError('prediction action must be retry, review, or auto')
        errors, count = character_error(prediction.raw, sample.reference)
        exact = characters(prediction.raw) == characters(sample.reference)
        non_rejected = bool(characters(prediction.raw)) and prediction.action != 'retry' and not prediction.error
        automatic = non_rejected and prediction.action == 'auto'
        totals.update(errors=errors, characters=count, exact=int(exact), non_rejected=int(non_rejected),
                      automatic=int(automatic), inference_failures=int(bool(prediction.error)),
                      automatic_errors=errors if automatic else 0,
                      automatic_characters=count if automatic else 0)
        valid_time = (prediction.processing_seconds is not None and prediction.duration_seconds is not None
                      and math.isfinite(prediction.processing_seconds) and prediction.processing_seconds >= 0
                      and math.isfinite(prediction.duration_seconds) and prediction.duration_seconds > 0)
        rtf = prediction.processing_seconds / prediction.duration_seconds if valid_time else None
        if valid_time:
            seconds.append(prediction.processing_seconds)
            rtfs.append(rtf)
        else:
            issues.append(f'Missing/invalid warm latency or clip duration: {sample.id}.')
        row_prediction = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                          for key, value in asdict(prediction).items()}
        rows.append({**row_prediction, 'reference': sample.reference, 'speaker': sample.speaker,
                     'session': sample.session, 'label_source': sample.label_source, 'domain': sample.domain,
                     'split': sample.split, 'video_sha256': sample.sha256,
                     'articulation': sample.articulation, 'articulation_verified': sample.articulation_verified,
                     'character_errors': errors, 'reference_characters': count,
                     'cer': errors / count, 'exact': exact, 'candidate_offered': non_rejected,
                     'automatically_accepted': automatic, 'rtf': rtf})
    size = len(dataset.samples)
    metrics = {
        'samples': size, 'speakers': len(sessions),
        'sessions_per_speaker': {key: len(value) for key, value in sorted(sessions.items())},
        'character_errors': totals['errors'], 'reference_characters': totals['characters'],
        'cer': totals['errors'] / totals['characters'] if totals['characters'] else None,
        'exact_sentence_rate': totals['exact'] / size if size else None,
        # An offered candidate is not a verified-correct result.
        'non_rejected_coverage': totals['non_rejected'] / size if size else None,
        'rejection_rate': 1 - totals['non_rejected'] / size if size else None,
        'automatic_coverage': totals['automatic'] / size if size else None,
        'automatic_cer': totals['automatic_errors'] / totals['automatic_characters']
                         if totals['automatic_characters'] else None,
        'inference_failures': totals['inference_failures'],
        'warm_processing_seconds_p50': percentile(seconds, .5),
        'warm_processing_seconds_p95': percentile(seconds, .95),
        'rtf_p50': percentile(rtfs, .5), 'rtf_p95': percentile(rtfs, .95),
        'timed_samples': len(seconds),
    }
    gates = []

    def gate(name, observed, required, passed):
        gates.append({'criterion': name, 'observed': observed, 'required': required, 'passed': bool(passed)})

    gate('pure_visual', mode, 'silent', mode == 'silent')
    gate('minimum_samples', size, thresholds.min_samples, size >= thresholds.min_samples)
    gate('minimum_speakers', len(sessions), thresholds.min_speakers, len(sessions) >= thresholds.min_speakers)
    least_sessions = min((len(value) for value in sessions.values()), default=0)
    gate('minimum_sessions_per_speaker', least_sessions, thresholds.min_sessions_per_speaker,
         least_sessions >= thresholds.min_sessions_per_speaker)
    webcam_count = sum(sample.domain == 'webcam' and not sample.mouth_roi for sample in dataset.samples)
    gate('webcam_domain', webcam_count, size if thresholds.require_webcam_domain else 'optional',
         not thresholds.require_webcam_domain or size > 0 and webcam_count == size)
    silent_count = sum(sample.articulation == 'silent' and sample.articulation_verified for sample in dataset.samples)
    gate('silent_articulation', silent_count, size if thresholds.require_silent_articulation else 'optional',
         not thresholds.require_silent_articulation or size > 0 and silent_count == size)
    gate('independent_data', issues, 'No unresolved provenance, overlap, or timing issues', not issues)
    gate('no_inference_failures', totals['inference_failures'], 0, totals['inference_failures'] == 0)
    for name, required, direction in (
            ('cer', thresholds.max_cer, 'max'),
            ('exact_sentence_rate', thresholds.min_exact_sentence_rate, 'min'),
            ('non_rejected_coverage', thresholds.min_non_rejected_coverage, 'min'),
            ('rtf_p95', thresholds.max_p95_rtf, 'max'),
            ('warm_processing_seconds_p95', thresholds.max_p95_processing_seconds, 'max')):
        observed = metrics[name]
        passed = observed is not None and (observed <= required if direction == 'max' else observed >= required)
        gate(name, observed, {direction: required}, passed)
    ready = all(item['passed'] for item in gates)
    return {'schema_version': 1, 'mode': mode, 'description': dataset.description,
            'readiness': {'ready': ready, 'status': 'ready_under_declared_protocol' if ready else 'not_ready',
                          'thresholds': asdict(thresholds), 'gates': gates},
            'metrics': metrics, 'samples': rows,
            'limitations': [
                'Decoder scores/margins are not correctness probabilities.',
                'Candidate coverage includes incorrect outputs; automatic coverage is reported separately.',
                'Independence and webcam provenance require a truthful human audit; file hashes do not prove labels.',
                'Passing these configurable criteria supports only this declared test domain, not universal usability.',
                'Latency excludes model loading and one warmup, and includes per-clip preparation plus decoding.',
            ]}
