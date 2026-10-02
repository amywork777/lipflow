"""Paired raw-text comparison of two stored visual-recognition reports.

    python scripts/compare_chinese_reports.py baseline.json candidate.json \
        --output comparison.json

All IDs, references and speaker identities must agree. Failed/empty predictions
remain in the denominator. CER delta is candidate minus baseline. The default
95% interval resamples whole speakers 5,000 times with seed 0; it does not prove
silent-webcam usability or license deployment. No inference or decoding occurs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from lipflow.evaluation import character_error, characters, percentile


def _score(report: dict) -> dict[str, dict]:
    if not isinstance(report, dict) or not isinstance(report.get('samples'), list) or not report['samples']:
        raise ValueError('report must contain a nonempty samples list')
    if 'schema_version' in report and (type(report['schema_version']) is not int or report['schema_version'] != 1):
        raise ValueError('expected report schema_version=1')
    scored = {}
    for row in report['samples']:
        if not isinstance(row, dict):
            raise ValueError('samples must contain objects')
        identity, speaker = row.get('sample_id'), row.get('speaker')
        if not isinstance(identity, str) or not identity or identity in scored:
            raise ValueError('report sample_id must be unique and nonempty')
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f'{identity}: speaker required for clustered comparison')
        raw, reference = row.get('raw'), row.get('reference')
        if not isinstance(raw, str) or not isinstance(reference, str):
            raise ValueError(f'{identity}: raw and reference must be strings')
        errors, length = character_error(raw, reference)
        for key, expected in (('character_errors', errors), ('reference_characters', length)):
            if key in row and (type(row[key]) is not int or row[key] != expected):
                raise ValueError(f'{identity}: recorded {key} differs from recomputed raw CER')
        exact = characters(raw) == characters(reference)
        if 'exact' in row and (type(row['exact']) is not bool or row['exact'] != exact):
            raise ValueError(f'{identity}: recorded exact differs from recomputed raw match')
        error, action = row.get('error'), row.get('action', 'review')
        if error is not None and not isinstance(error, str):
            raise ValueError(f'{identity}: error must be a string or null')
        if action not in ('auto', 'review', 'retry'):
            raise ValueError(f'{identity}: action must be auto/review/retry')
        empty = not bool(characters(raw))
        scored[identity] = {'source': row, 'speaker': speaker, 'reference': reference,
                            'character_errors': errors, 'reference_characters': length,
                            'exact': exact, 'empty_output': empty, 'inference_failure': bool(error),
                            'candidate_offered': not empty and action != 'retry' and not error}
    recorded = report.get('metrics')
    if recorded is not None:
        if not isinstance(recorded, dict):
            raise ValueError('report metrics must be an object')
        observed = _metrics(list(scored.values()))
        for key in ('samples', 'reference_characters', 'character_errors', 'inference_failures'):
            if key in recorded and (type(recorded[key]) is not int or recorded[key] != observed[key]):
                raise ValueError(f'report metrics {key} differs from full sample rows')
        for key in ('cer', 'exact_sentence_rate', 'non_rejected_coverage'):
            if key in recorded and (type(recorded[key]) not in (float, int)
                                    or not math.isfinite(recorded[key])
                                    or abs(recorded[key] - observed[key]) > 1e-12):
                raise ValueError(f'report metrics {key} differs from recomputed raw scoring')
    return scored


def _metrics(rows: list[dict]) -> dict:
    references = sum(row['reference_characters'] for row in rows)
    errors = sum(row['character_errors'] for row in rows)
    exact = sum(row['exact'] for row in rows)
    return {'samples': len(rows), 'reference_characters': references, 'character_errors': errors,
            'cer': errors / references, 'exact_sentences': exact, 'exact_sentence_rate': exact / len(rows),
            'empty_outputs': sum(row['empty_output'] for row in rows),
            'inference_failures': sum(row['inference_failure'] for row in rows),
            'non_rejected_coverage': sum(row['candidate_offered'] for row in rows) / len(rows)}


def _delta(baseline: dict, candidate: dict) -> dict:
    difference = (candidate['character_errors'] - baseline['character_errors']) / baseline['reference_characters']
    return {'character_errors': candidate['character_errors'] - baseline['character_errors'],
            'cer': difference, 'cer_percentage_points': 100 * difference,
            'exact_sentences': candidate['exact_sentences'] - baseline['exact_sentences'],
            'exact_sentence_rate': candidate['exact_sentence_rate'] - baseline['exact_sentence_rate']}


def compare(baseline: dict, candidate: dict, *, seed: int = 0, resamples: int = 5000,
            manifest: dict | None = None) -> dict:
    if type(seed) is not int or type(resamples) is not int or not 1 <= resamples <= 1_000_000:
        raise ValueError('seed must be integer and resamples between 1 and 1,000,000')
    before, after = _score(baseline), _score(candidate)
    if before.keys() != after.keys():
        missing, unexpected = sorted(before.keys() - after.keys()), sorted(after.keys() - before.keys())
        raise ValueError(f'paired reports require identical sample IDs; missing={missing}, unexpected={unexpected}')
    if manifest is not None:
        if not isinstance(manifest, dict) or not isinstance(manifest.get('samples'), list) or not manifest['samples']:
            raise ValueError('manifest must contain a nonempty samples list')
        declared = {}
        for sample in manifest['samples']:
            if not isinstance(sample, dict) or not isinstance(sample.get('id'), str) or not sample['id'] or sample['id'] in declared:
                raise ValueError('manifest samples require unique nonempty IDs')
            declared[sample['id']] = sample
        if declared.keys() != before.keys():
            raise ValueError('reports must cover every frozen manifest sample exactly once')
        for identity, sample in declared.items():
            for key in ('reference', 'speaker'):
                if sample.get(key) != before[identity][key] or sample.get(key) != after[identity][key]:
                    raise ValueError(f'{identity}: frozen manifest {key} mismatch')
            for key in ('split', 'session', 'domain', 'articulation', 'articulation_verified', 'sha256'):
                report_key = 'video_sha256' if key == 'sha256' else key
                if key in sample and any(item['source'].get(report_key) != sample[key] for item in (before[identity], after[identity])):
                    raise ValueError(f'{identity}: frozen manifest {key} mismatch')
    if 'mode' in baseline or 'mode' in candidate:
        if baseline.get('mode') != candidate.get('mode'):
            raise ValueError('paired reports have different declared modes')
    speakers = defaultdict(list)
    for identity in sorted(before):
        left, right = before[identity], after[identity]
        for key in ('reference', 'speaker'):
            if left[key] != right[key]:
                raise ValueError(f'{identity}: paired {key} must agree exactly')
        # Optional corpus fields must agree wherever either report declares them.
        # This catches equal-text predictions paired to a changed video/session.
        for key in ('split', 'session', 'video_sha256', 'domain', 'articulation', 'articulation_verified'):
            if key in left['source'] or key in right['source']:
                if left['source'].get(key) != right['source'].get(key):
                    raise ValueError(f'{identity}: paired {key} mismatch')
        speakers[left['speaker']].append(identity)
    identities = sorted(before)
    baseline_metrics = _metrics([before[key] for key in identities])
    candidate_metrics = _metrics([after[key] for key in identities])
    per_speaker, clusters = {}, []
    for speaker, keys in sorted(speakers.items()):
        bm, cm = _metrics([before[key] for key in keys]), _metrics([after[key] for key in keys])
        per_speaker[speaker] = {'baseline': bm, 'candidate': cm, 'delta': _delta(bm, cm)}
        clusters.append((bm['character_errors'], cm['character_errors'], bm['reference_characters']))
    interval = {'method': 'percentile speaker-cluster bootstrap', 'confidence_level': .95,
                'seed': seed, 'resamples': resamples, 'speakers': len(clusters),
                'definition': 'candidate CER minus baseline CER; negative values mean fewer raw errors',
                'lower': None, 'upper': None, 'available': len(clusters) >= 2}
    if interval['available']:
        generator, deltas = random.Random(seed), []
        for _ in range(resamples):
            baseline_errors = candidate_errors = references = 0
            for _ in clusters:
                be, ce, count = clusters[generator.randrange(len(clusters))]
                baseline_errors += be
                candidate_errors += ce
                references += count
            deltas.append((candidate_errors - baseline_errors) / references)
        interval['lower'], interval['upper'] = percentile(deltas, .025), percentile(deltas, .975)
    else:
        interval['reason'] = 'At least two known speakers are required; one-cluster intervals are not informative.'
    return {'schema_version': 1, 'purpose': 'paired_raw_prediction_comparison',
            'product_readiness_established': False, 'references_used_for_decoding': False,
            'frozen_manifest_coverage_checked': manifest is not None,
            'baseline': baseline_metrics, 'candidate': candidate_metrics,
            'delta': _delta(baseline_metrics, candidate_metrics),
            'cer_difference_95pct_interval': interval, 'per_speaker': per_speaker,
            'samples': [{'sample_id': key, 'speaker': before[key]['speaker'],
                         'reference_characters': before[key]['reference_characters'],
                         'baseline_character_errors': before[key]['character_errors'],
                         'candidate_character_errors': after[key]['character_errors'],
                         'character_errors_delta': after[key]['character_errors'] - before[key]['character_errors'],
                         'baseline_exact': before[key]['exact'], 'candidate_exact': after[key]['exact'],
                         'baseline_inference_failure': before[key]['inference_failure'],
                         'candidate_inference_failure': after[key]['inference_failure'],
                         'baseline_empty_output': before[key]['empty_output'],
                         'candidate_empty_output': after[key]['empty_output']} for key in identities],
            'limitations': [
                'All paired samples, including failed, rejected and empty predictions, remain in raw CER and exact-match denominators.',
                'Paired-report completeness is checked against declared metrics; pass the frozen manifest to verify the intended full corpus.',
                'Whole speakers are resampled with replacement; CER is recomputed from pooled character counts, not averaged sentence CER.',
                'The interval assumes independent, representative speakers; few speakers or shared recording conditions limit its interpretation.',
                'Held-out labels must not select decoding, cleanup or training; freeze the candidate before accessing a fresh test.',
                'Statistical improvement does not establish usable silent Chinese speech, webcam performance or deployment rights.',
                'Normalization uses NFKC/casefold letters and numbers; unknown-token literal letters remain in raw CER.',
            ]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resamples', type=int, default=5000)
    parser.add_argument('--manifest', type=Path, help='Frozen test manifest; only IDs/labels/metadata are read, no videos')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = (args.baseline, args.candidate, *([args.manifest] if args.manifest else []))
        if args.output and args.output.resolve() in {path.resolve() for path in inputs}:
            raise ValueError('comparison output must not overwrite an input report')
        baseline_bytes, candidate_bytes = args.baseline.read_bytes(), args.candidate.read_bytes()
        manifest_bytes = args.manifest.read_bytes() if args.manifest else None
        result = compare(json.loads(baseline_bytes), json.loads(candidate_bytes), seed=args.seed, resamples=args.resamples,
                         manifest=json.loads(manifest_bytes) if manifest_bytes is not None else None)
        result['provenance'] = {'baseline_report_sha256': hashlib.sha256(baseline_bytes).hexdigest(),
                                'candidate_report_sha256': hashlib.sha256(candidate_bytes).hexdigest(),
                                'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None}
        serialized = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding='utf-8')
        print(serialized, end='')
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
