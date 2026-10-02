"""Diagnose stored Mandarin predictions without loading a model or changing text.

    python scripts/analyze_chinese_errors.py report.json --vocabulary char_units.txt
    python scripts/analyze_chinese_errors.py adapter/report.json \
        --stage candidate --split dev --manifest dev.json --output errors.json

Held-out labels are diagnostic evidence only. This program never emits a decoder
configuration, corrected transcript, training target or deployable model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from lipflow.evaluation import characters


def alignment(hypothesis: str, reference: str) -> dict:
    """Minimum CER alignment, deterministic diagonal/deletion/insertion ties.

    The total distance is unique; its edit-type decomposition can be ambiguous.
    Direction bytes bound memory, and giant transcripts are rejected explicitly.
    """
    ref, hyp = characters(reference), characters(hypothesis)
    if not ref:
        raise ValueError('reference must contain at least one letter or number')
    width, height = len(hyp) + 1, len(ref) + 1
    if width * height > 4_000_000:
        raise ValueError('transcript alignment exceeds 4,000,000 cells')
    trace = bytearray(width * height)
    previous = list(range(width))
    for j in range(1, width):
        trace[j] = 3  # insertion into reference
    for i in range(1, height):
        row = [i] + [0] * len(hyp)
        trace[i * width] = 2  # deletion from reference
        for j in range(1, width):
            diagonal = previous[j - 1] + (ref[i - 1] != hyp[j - 1])
            deletion, insertion = previous[j] + 1, row[j - 1] + 1
            best = min(diagonal, deletion, insertion)
            row[j] = best
            trace[i * width + j] = 1 if diagonal == best else 2 if deletion == best else 3
        previous = row
    counts = Counter(substitutions=0, deletions=0, insertions=0, matches=0)
    substitutions, deleted, inserted = Counter(), Counter(), Counter()
    i, j = len(ref), len(hyp)
    while i or j:
        operation = trace[i * width + j]
        if operation == 1:
            if ref[i - 1] == hyp[j - 1]:
                counts['matches'] += 1
            else:
                counts['substitutions'] += 1
                substitutions[ref[i - 1], hyp[j - 1]] += 1
            i, j = i - 1, j - 1
        elif operation == 2:
            counts['deletions'] += 1
            deleted[ref[i - 1]] += 1
            i -= 1
        elif operation == 3:
            counts['insertions'] += 1
            inserted[hyp[j - 1]] += 1
            j -= 1
        else:
            raise RuntimeError('invalid alignment trace')
    return {**counts, 'character_errors': previous[-1], 'reference_characters': len(ref),
            'hypothesis_characters': len(hyp), 'substitution_pairs': substitutions,
            'deleted_characters': deleted, 'inserted_characters': inserted}


def select_rows(report: dict, *, stage: str | None = None, split: str | None = None) -> list[dict]:
    if not isinstance(report, dict):
        raise ValueError('report must be an object')
    if 'baseline' in report or 'candidate' in report:
        if stage not in ('baseline', 'candidate') or split not in ('dev', 'test'):
            raise ValueError('adapter reports require explicit --stage and --split')
        section = report.get(stage, {}).get(split)
        if not isinstance(section, dict):
            raise ValueError('requested adapter stage/split not found')
    else:
        if stage is not None:
            raise ValueError('--stage only applies to adapter reports')
        section = report
    rows = section.get('samples')
    if not isinstance(rows, list) or not rows:
        raise ValueError('report samples must be a nonempty list')
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('report samples must contain objects')
    if split and any(row.get('split', split) != split for row in rows):
        raise ValueError('report contains a different declared split')
    return rows


def vocabulary_characters(path: str | Path) -> set[str]:
    """Read published unit rows; special/multichar tokens are not CER units."""
    result = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if fields and len(fields[0]) == 1:
            result.update(characters(fields[0]))
    if not result:
        raise ValueError('vocabulary contains no single-character CER units')
    return result


def _bucket(length: int) -> str:
    return '1-20' if length <= 20 else '21-40' if length <= 40 else '41-60' if length <= 60 else '61+'


def _metrics(rows: list[dict]) -> dict:
    totals = Counter()
    for row in rows:
        for name in ('character_errors', 'reference_characters', 'hypothesis_characters',
                     'substitutions', 'deletions', 'insertions', 'matches', 'empty_output',
                     'inference_failure', 'exact', 'unknown_token_markers'):
            totals[name] += int(row[name])
    count, references = len(rows), totals['reference_characters']
    return {'samples': count, **totals, 'cer': totals['character_errors'] / references if references else None,
            'exact_sentence_rate': totals['exact'] / count if count else None,
            'hypothesis_reference_length_ratio': totals['hypothesis_characters'] / references if references else None,
            'mean_reference_length': references / count if count else None}


def analyze(rows: list[dict], *, vocabulary: set[str] | None = None, manifest: dict | None = None) -> dict:
    """Preserve every sample and independently recompute recorded CER totals."""
    metadata = {}
    if manifest is not None:
        if not isinstance(manifest, dict) or not isinstance(manifest.get('samples'), list):
            raise ValueError('manifest must contain samples')
        for sample in manifest['samples']:
            if not isinstance(sample, dict) or not isinstance(sample.get('id'), str) or sample['id'] in metadata:
                raise ValueError('manifest samples must have unique string IDs')
            metadata[sample['id']] = sample
    seen, summaries = set(), []
    substitutions, deleted, inserted, oov = Counter(), Counter(), Counter(), Counter()
    groups, lengths, oov_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in rows:
        identity = row.get('sample_id')
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('prediction samples must have unique nonempty sample_id')
        seen.add(identity)
        if not isinstance(row.get('raw'), str) or not isinstance(row.get('reference'), str):
            raise ValueError(f'{identity}: raw and reference must be strings')
        source = metadata.get(identity, {})
        if manifest is not None and not source:
            raise ValueError(f'{identity}: not present in supplied manifest')
        for key in ('reference', 'speaker', 'split', 'video_sha256'):
            source_key = 'sha256' if key == 'video_sha256' else key
            if key in row and source_key in source and row[key] != source[source_key]:
                raise ValueError(f'{identity}: report/manifest {key} mismatch')
        edits = alignment(row['raw'], row['reference'])
        for key in ('character_errors', 'reference_characters'):
            if key in row and (type(row[key]) is not int or row[key] != edits[key]):
                raise ValueError(f'{identity}: recorded {key} differs from recomputed CER')
        substitutions.update(edits.pop('substitution_pairs'))
        deleted.update(edits.pop('deleted_characters'))
        inserted.update(edits.pop('inserted_characters'))
        ref = characters(row['reference'])
        if vocabulary is not None:
            unsupported = set(ref) - vocabulary
        elif 'reference_oov_characters' in row:
            declared = row['reference_oov_characters']
            if not isinstance(declared, list) or any(not isinstance(unit, str) or len(characters(unit)) != 1 for unit in declared):
                raise ValueError(f'{identity}: invalid reference_oov_characters')
            unsupported = {characters(unit)[0] for unit in declared}
            if not unsupported <= set(ref):
                raise ValueError(f'{identity}: declared OOV absent from reference')
        else:
            unsupported = None
        occurrence_count = sum(unit in unsupported for unit in ref) if unsupported is not None else None
        if unsupported:
            oov.update(unit for unit in ref if unit in unsupported)
        speaker = row.get('speaker', source.get('speaker', '(unreported)'))
        if not isinstance(speaker, str) or not speaker:
            raise ValueError(f'{identity}: speaker must be nonempty string')
        summary = {'sample_id': identity, 'speaker': speaker, 'split': row.get('split', source.get('split')),
                   **edits, 'empty_output': edits['hypothesis_characters'] == 0,
                   'inference_failure': bool(row.get('error')), 'exact': edits['character_errors'] == 0,
                   'unknown_token_markers': row['raw'].count('<unk>'),
                   'cer': edits['character_errors'] / edits['reference_characters'],
                   'oov_occurrences': occurrence_count,
                   'length_ratio': edits['hypothesis_characters'] / edits['reference_characters']}
        summaries.append(summary)
        groups[speaker].append(summary)
        lengths[_bucket(len(ref))].append(summary)
        oov_groups['unreported' if unsupported is None else 'contains_oov' if unsupported else 'in_vocabulary'].append(summary)
    return {'schema_version': 1, 'diagnostic_only': True, 'references_used_for_decoding': False,
            'metrics': _metrics(summaries),
            'per_speaker': {key: _metrics(value) for key, value in sorted(groups.items())},
            'per_reference_length': {key: _metrics(value) for key, value in sorted(lengths.items())},
            'per_oov_presence': {key: _metrics(value) for key, value in sorted(oov_groups.items())},
            'oov_characters': dict(oov.most_common()),
            'oov_occurrences': sum(oov.values()),
            'oov_diagnostics_available_samples': sum(row['oov_occurrences'] is not None for row in summaries),
            'top_substitutions': [{'reference': a, 'hypothesis': b, 'count': count}
                                  for (a, b), count in substitutions.most_common(30)],
            'top_deletions': dict(deleted.most_common(30)), 'top_insertions': dict(inserted.most_common(30)),
            'samples': summaries,
            'limitations': [
                'Minimum-distance alignments can have tied decompositions; counts use diagonal, deletion, insertion tie order.',
                'CER normalization is NFKC/casefold Unicode letters and numbers; punctuation is not evaluated.',
                'Literal <unk> markers are flagged separately; their letters still enter canonical CER and are not silently removed.',
                'Unknown vocabulary coverage is reported as unreported, never assumed to be in vocabulary.',
                'Held-out labels may diagnose failures but must not select corrections, decoding parameters or training examples.',
                'Changes selected from dev need a fresh held-out test if an earlier test has already informed development.',
                'No source videos, audio, camera, model, LLM or network are accessed.',
            ]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('report', type=Path)
    parser.add_argument('--stage', choices=('baseline', 'candidate'))
    parser.add_argument('--split', choices=('dev', 'test'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--vocabulary', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.report.read_bytes()
        report = json.loads(raw)
        manifest = json.loads(args.manifest.read_text(encoding='utf-8')) if args.manifest else None
        vocabulary = vocabulary_characters(args.vocabulary) if args.vocabulary else None
        result = analyze(select_rows(report, stage=args.stage, split=args.split), vocabulary=vocabulary, manifest=manifest)
        result['provenance'] = {'report_sha256': hashlib.sha256(raw).hexdigest(),
                                'stage': args.stage, 'split': args.split,
                                'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest() if args.manifest else None,
                                'vocabulary_sha256': hashlib.sha256(args.vocabulary.read_bytes()).hexdigest() if args.vocabulary else None}
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
