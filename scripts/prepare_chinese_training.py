"""Prepare a new train-only manifest for the pinned CNVSRC character vocabulary.

No model is loaded and no decoder results are read. With explicit permission,
whole unsupported training rows are excluded; reference text is never changed.
Development/test manifests are always rejected. The new manifest records every
excluded ID/unit and both source-manifest and pinned-vocabulary hashes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from lipflow.evaluation import load_manifest

# Sibling research helpers import Torch only inside actual model/training calls.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_chinese_adapter import target_units
from evaluate_cnvsrc import VOCABULARY_SHA256, _verified_file


def _relative_video(video: str | Path, output_parent: Path) -> str:
    resolved = Path(video).resolve(strict=True)
    try:
        # macOS /tmp -> /private/tmp aliases must use the same physical parent
        # as load_manifest(), otherwise relocation can invent /private/private.
        return Path(os.path.relpath(resolved, output_parent.resolve())).as_posix()
    except ValueError:  # Windows files on different drives cannot have a relative path.
        return str(resolved)


def prepare(manifest, vocabulary, output, *, exclude_unsupported_training_samples=False) -> dict:
    manifest, vocabulary, output = Path(manifest).resolve(), Path(vocabulary).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists; it will not be overwritten: {output}')
    vocab_hash = _verified_file(vocabulary, VOCABULARY_SHA256)
    vocab_bytes = vocabulary.read_bytes()
    if hashlib.sha256(vocab_bytes).hexdigest() != vocab_hash:
        raise ValueError('Vocabulary changed during verification; retry with stable source files')
    vocab = {line.split()[0] for line in vocab_bytes.decode('utf-8').splitlines() if line.strip()}
    vocab -= {'<blank>', '<unk>', '<eos>', '<space>'}
    source_bytes = manifest.read_bytes()
    source = json.loads(source_bytes)
    dataset = load_manifest(manifest)  # validates every source row and resolves/hashes existing videos
    if manifest.read_bytes() != source_bytes:
        raise ValueError('Manifest changed during validation; retry with stable source files')
    if dataset.split != 'train' or any(sample.split != 'train' for sample in dataset.samples):
        raise ValueError('Only an explicitly train-only manifest can be prepared; dev/test filtering is forbidden')
    ids = [sample.id for sample in dataset.samples]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Training manifest must be nonempty with unique sample IDs')
    excluded, kept = [], []
    for row, sample in zip(source['samples'], dataset.samples):
        units = target_units(sample.reference)
        unsupported = sorted(set(units) - vocab)
        if not units or unsupported:
            excluded.append({'sample_id': sample.id, 'unsupported_units': unsupported, 'empty_target': not units})
        else:
            original = copy.deepcopy(row)
            original['video'] = _relative_video(sample.video, output.parent)
            kept.append(original)
    if excluded and exclude_unsupported_training_samples is not True:
        raise ValueError('Unsupported training targets; no rows were discarded. Explicit '
                         '--exclude-unsupported-training-samples is required:\n' +
                         json.dumps(excluded, ensure_ascii=False, indent=2))
    if not kept:
        raise ValueError('No trainable samples remain; an empty prepared manifest will not be written')
    result = copy.deepcopy(source)
    result['samples'] = kept
    for row in result.get('development_samples', []):
        if 'video' in row:
            original_path = Path(row['video'])
            resolved = original_path if original_path.is_absolute() else manifest.parent / original_path
            row['video'] = _relative_video(resolved, output.parent)
    result['training_eligibility'] = {
        'criterion': 'Whole training rows with empty targets or units outside the pinned fixed vocabulary',
        'input_samples': len(dataset.samples), 'kept_samples': len(kept), 'excluded_samples': len(excluded),
        'original_manifest': str(manifest), 'original_manifest_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'vocabulary_sha256': vocab_hash, 'excluded_ids': [row['sample_id'] for row in excluded],
        'exclusions': excluded, 'explicit_exclusion_permission': exclude_unsupported_training_samples is True,
        'selection_used_decoder_results': False, 'references_unmodified': True, 'dev_test_filtered': False,
    }
    if 'training_eligibility' in source:
        result['training_eligibility']['source_training_eligibility'] = copy.deepcopy(source['training_eligibility'])
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.prepared-train-', suffix='.tmp', dir=output.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)  # publish exclusively; an existing destination is never overwritten
    finally:
        Path(temporary).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--vocabulary', required=True, help='Pinned CNVSRC datamodule/char_units.txt')
    parser.add_argument('--output', required=True, help='New manifest path; existing files are refused')
    parser.add_argument('--exclude-unsupported-training-samples', action='store_true')
    args = parser.parse_args()
    try:
        result = prepare(args.manifest, args.vocabulary, args.output,
                         exclude_unsupported_training_samples=args.exclude_unsupported_training_samples)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result['training_eligibility'], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
