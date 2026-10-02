r"""Research-only CNVSRC encoder adaptation on independently collected videos.

    uv run python scripts/train_chinese_adapter.py --accept-research-license \
        --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth --source-dir /local/CNVSRC2025 \
        --train-manifest train.json --dev-manifest dev.json --test-manifest test.json \
        --scope general --output /local/adapter-run

No download, audio input, GUI deployment, LLM cleanup or automatic activation.
The fixed training schedule is followed by one raw-text comparison on dev/test.
Diagnostic videos, shared labels and overlapping partitions are rejected.
Unknown base-model training overlap is reported; this experiment cannot establish
webcam readiness. Saved tensors remain subject to the source research license.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import unicodedata

from lipflow.evaluation import Dataset, Prediction, character_error, characters, load_manifest

from evaluate_cnvsrc import CHECKPOINT_SHA256, CONFIG_SHA256, SOURCE_REVISION, VOCABULARY_SHA256, _reader, _verified_file

# These eight publicly inspected AISHELL6 demos are diagnostic data, never a
# training or held-out acceptance set. Hashes also block relabelling/renaming.
DIAGNOSTIC_HASHES = frozenset({
    '3590ecb7ca6b8f0ee05bc56e7e9a015e2263556122129fedeaef8fdca21cc3ef',
    '9b9c55efff6960e825ef9084797ac481396c6afc977a8c4e6830866ed6bb93ba',
    'bdeaa8ed1660015054e5cb156ee67a4329792d331f3301301f661f5e8e768cc1',
    'e19597e222ee75d665e78e40cb4ccf5adf5e5b74c3e87834ba56f7d1459cd7cd',
    '61488be19724bc04e9146d542f32c99fee51bd5ab86edd074cafad4ada96a738',
    '6b4db3061c78b6a6e1096ce6bc41ef00d8b0b796dda254392d192b57834404c3',
    '58250a3bd2965fb2993848467ef6cfa8064355a61c194e692a627c2a07c552b5',
    '896991c5850b8c014433fb3843df3dbd1a9f9490bf99e3fd38ce9b943048857b',
})
IGNORED_PUNCTUATION = frozenset('，。！？；、：“”‘’（）【】《》,.!?;:\"\'()[]')


def target_units(text: str) -> list[str]:
    """Ignore delimiters, retain every digit, Latin character and other symbol.

    Unsupported units must raise in validation, never disappear into <unk> or
    have their sample removed. The base model has a fixed Han vocabulary.
    """
    normalized = unicodedata.normalize('NFKC', text)
    return [unit for unit in normalized if not unit.isspace() and unit not in IGNORED_PUNCTUATION]


def vocabulary_errors(datasets: dict[str, Dataset], token_list: list[str]) -> list[dict]:
    vocabulary = set(token_list) - {'<blank>', '<unk>', '<eos>', '<space>'}
    errors = []
    for partition, dataset in datasets.items():
        for sample in dataset.samples:
            units = target_units(sample.reference)
            unsupported = sorted(set(units) - vocabulary)
            if not units or unsupported:
                errors.append({'partition': partition, 'sample_id': sample.id,
                               'unsupported_units': unsupported, 'empty_target': not units})
    return errors


def partition_issues(datasets: dict[str, Dataset], scope: str) -> list[str]:
    """Validate a fixed adaptation protocol without consulting decoder results."""
    issues = []
    roles = {'train': {'train'}, 'dev': {'dev', 'val', 'validation'}, 'test': {'test'}}
    if set(datasets) != set(roles) or scope not in ('personal', 'general'):
        raise ValueError('Expected train/dev/test datasets and personal or general scope')
    all_samples = []
    for role, dataset in datasets.items():
        if not dataset.samples:
            issues.append(f'{role}: partition is empty')
        if dataset.split not in roles[role]:
            issues.append(f'{role}: explicit split must be one of {sorted(roles[role])}')
        ids = set()
        for sample in dataset.samples:
            all_samples.append((role, sample))
            if sample.split not in roles[role]:
                issues.append(f'{role}/{sample.id}: sample split conflicts with partition')
            if sample.id in ids:
                issues.append(f'{role}/{sample.id}: duplicate ID within partition')
            ids.add(sample.id)
            if not sample.label_verified or not sample.label_source:
                issues.append(f'{role}/{sample.id}: reference provenance is not verified')
            if not sample.sha256:
                issues.append(f'{role}/{sample.id}: video hash is missing')
            if (sample.sha256 in DIAGNOSTIC_HASHES or 'zutm.github.io/aishell6-whisper' in sample.label_source.casefold()
                    or sample.session.casefold() in ('published-demo', 'diagnostic')):
                issues.append(f'{role}/{sample.id}: known diagnostic material cannot be used')
    if scope == 'personal' and len({sample.speaker for _, sample in all_samples}) != 1:
        issues.append('personal scope requires one declared speaker across all partitions')
    for i, (role_a, a) in enumerate(all_samples):
        for role_b, b in all_samples[i + 1:]:
            same_content = a.sha256 == b.sha256 or a.video == b.video
            overlap = max(a.start, b.start) < min(a.end or math.inf, b.end or math.inf)
            if same_content and overlap:
                issues.append(f'{role_a}/{a.id}, {role_b}/{b.id}: overlapping video content')
            if role_a == role_b:
                continue
            if a.id == b.id:
                issues.append(f'{role_a}/{a.id}, {role_b}/{b.id}: shared sample ID')
            if characters(a.reference) == characters(b.reference):
                issues.append(f'{role_a}/{a.id}, {role_b}/{b.id}: shared normalized reference')
            if scope == 'general' and a.speaker == b.speaker:
                issues.append(f'{role_a}/{a.id}, {role_b}/{b.id}: general scope requires distinct speakers')
            if scope == 'personal' and a.speaker == b.speaker and a.session == b.session:
                issues.append(f'{role_a}/{a.id}, {role_b}/{b.id}: personal hold-out requires separate sessions')
    return sorted(set(issues))


def configure_trainable(model, last_layers: int = 1):
    """Freeze the base; adapt only final encoder blocks and their output norm."""
    import torch

    blocks = model.encoder.encoders
    if not 1 <= last_layers <= len(blocks):
        raise ValueError('last_layers must fit the encoder block count')
    # Warmup/held-out encode() uses inference_mode. Positional encodings cache
    # .pe as an ordinary attribute, so Module.to() cannot convert these caches
    # back into normal tensors. Training attention must save them for backward.
    with torch.inference_mode(False):
        for module in model.encoder.modules():
            positional_cache = getattr(module, 'pe', None)
            if isinstance(positional_cache, torch.Tensor) and positional_cache.is_inference():
                module.pe = positional_cache.clone()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = [f'encoder.encoders.{index}.' for index in range(len(blocks) - last_layers, len(blocks))]
    if getattr(model.encoder, 'after_norm', None) is not None:
        prefixes.append('encoder.after_norm.')
    named = {name: parameter for name, parameter in model.named_parameters() if name.startswith(tuple(prefixes))}
    if not named:
        raise ValueError('No compatible final encoder parameters')
    for parameter in named.values():
        parameter.requires_grad_(True)
    for block in list(blocks)[-last_layers:]:
        block.train()
    # Small adaptation sets must not alter inherited BN running statistics.
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
    return named


def ctc_loss(reader, inputs, target_ids: list[int]):
    """CPU CTC retains gradients through the device transfer to the encoder."""
    import torch
    import torch.nn.functional as functional

    encoded, _ = reader.model.encoder(inputs.unsqueeze(0).to(reader.enc_device), None)
    logits = reader.model.ctc.ctc_lo(encoded)
    log_probabilities = logits.float().log_softmax(-1).transpose(0, 1).cpu()
    required_frames = len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:]))
    if log_probabilities.shape[0] < required_frames:
        raise ValueError('Too few encoder frames for the full CTC target; sample is not silently dropped')
    targets = torch.tensor(target_ids, dtype=torch.long)
    loss = functional.ctc_loss(
        log_probabilities, targets, torch.tensor([log_probabilities.shape[0]]),
        torch.tensor([len(target_ids)]), blank=0, reduction='sum', zero_infinity=False,
    ) / len(target_ids)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite CTC loss; training stopped')
    return loss


def summarize(dataset: Dataset, predictions: list[Prediction]) -> dict:
    by_id = {prediction.sample_id: prediction for prediction in predictions}
    errors = count = exact = non_rejected = failures = 0
    rows = []
    for sample in dataset.samples:
        prediction = by_id[sample.id]
        error, length = character_error(prediction.raw, sample.reference)
        correct = characters(prediction.raw) == characters(sample.reference)
        offered = bool(characters(prediction.raw)) and prediction.action != 'retry' and not prediction.error
        errors += error
        count += length
        exact += correct
        non_rejected += offered
        failures += bool(prediction.error)
        rows.append({**asdict(prediction), 'reference': sample.reference,
                     'character_errors': error, 'reference_characters': length, 'exact': correct})
    return {'character_errors': errors, 'reference_characters': count, 'cer': errors / count,
            'exact_sentence_rate': exact / len(dataset.samples),
            'non_rejected_coverage': non_rejected / len(dataset.samples),
            'inference_failures': failures, 'samples': rows}


def improvement_gate(baseline: dict, candidate: dict) -> dict:
    """A relative improvement is research evidence, never readiness or activation."""
    checks = {
        'raw_cer_strictly_lower': candidate['cer'] < baseline['cer'],
        'exact_sentence_rate_not_lower': candidate['exact_sentence_rate'] >= baseline['exact_sentence_rate'],
        'candidate_coverage_not_lower': candidate['non_rejected_coverage'] >= baseline['non_rejected_coverage'],
        'no_inference_failures': baseline['inference_failures'] == candidate['inference_failures'] == 0,
    }
    return {'improved': all(checks.values()), 'checks': checks,
            'baseline_cer': baseline['cer'], 'candidate_cer': candidate['cer']}


def _progress(stage: str, completed: int, total: int, **details):
    """Progress has no reference text or recognition hypotheses."""
    print(json.dumps({'stage': stage, 'completed': completed, 'total': total, **details},
                     allow_nan=False), file=sys.stderr, flush=True)


def _predict(reader, dataset: Dataset, stage: str = 'prediction') -> list[Prediction]:
    from evaluate_chinese import ClipRejected, _visual_input
    from lipflow.confidence import assess

    reader.model.eval()
    predictions = []
    _progress(stage, 0, len(dataset.samples))
    for sample in dataset.samples:
        started, duration = time.monotonic(), None
        try:
            rois, duration, quality = _visual_input(sample, reader)
            encoded = reader.encode(rois)
            hypotheses = reader.hypotheses(encoded, nbest=5)
            decision = assess(hypotheses, reader.greedy(encoded), quality, policy='review', language='zh')
            prediction = Prediction(sample.id, hypotheses[0].text if hypotheses else '', duration,
                                    time.monotonic() - started, decision.action, decision.reason)
        except ClipRejected as exc:
            prediction = Prediction(sample.id, '', exc.duration, time.monotonic() - started, 'retry', reason=str(exc))
        except Exception as exc:
            prediction = Prediction(sample.id, '', duration, time.monotonic() - started, 'retry', error=str(exc))
        predictions.append(prediction)
        _progress(stage, len(predictions), len(dataset.samples))
    return predictions


def _manifest_evidence(path: str, dataset: Dataset) -> dict:
    raw_manifest = Path(path).read_bytes()
    declared = json.loads(raw_manifest)
    return {'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
            'split': dataset.split, 'sample_count': len(dataset.samples),
            'training_overlap_checked': dataset.training_overlap_checked,
            # These producer declarations remain inspectable evidence rather
            # than an assertion that base-model training overlap was audited.
            'source_metadata': declared.get('source'),
            'training_eligibility': declared.get('training_eligibility'),
            'samples': [{'id': sample.id, 'speaker': sample.speaker, 'session': sample.session,
                         'video_sha256': sample.sha256, 'start': sample.start, 'end': sample.end,
                         'articulation': sample.articulation,
                         'articulation_verified': sample.articulation_verified,
                         'label_verified': sample.label_verified,
                         'label_source': sample.label_source,
                         'reference_sha256': hashlib.sha256(sample.reference.encode()).hexdigest()}
                        for sample in dataset.samples]}


def run(args) -> dict:
    import numpy as np
    import torch
    from evaluate_chinese import _visual_input
    from lipflow.vsr import LipReader

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to preserve prior experiments')
    paths = {'train': args.train_manifest, 'dev': args.dev_manifest, 'test': args.test_manifest}
    datasets = {role: load_manifest(path) for role, path in paths.items()}
    issues = partition_issues(datasets, args.scope)
    if issues:
        raise ValueError('Invalid adaptation partitions:\n' + '\n'.join(issues))

    source = Path(args.source_dir)
    source = source / 'VSR' if (source / 'VSR').is_dir() else source
    vocabulary_path = source / 'datamodule/char_units.txt'
    _verified_file(vocabulary_path, VOCABULARY_SHA256)
    tokens = ['<blank>'] + [line.split()[0] for line in vocabulary_path.read_text(encoding='utf-8').splitlines()] + ['<eos>']
    unsupported = vocabulary_errors(datasets, tokens)
    untrainable = [row for row in unsupported if row['partition'] == 'train']
    if untrainable:
        raise ValueError('Unsupported training targets; no samples discarded:\n' + json.dumps(untrainable, ensure_ascii=False))

    output.mkdir(parents=True, exist_ok=True)
    evidence = {role: _manifest_evidence(paths[role], dataset) for role, dataset in datasets.items()}
    common_evidence = {
        'schema_version': 1, 'scope': args.scope,
        'base': {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': SOURCE_REVISION,
                 'configuration_sha256': CONFIG_SHA256, 'vocabulary_sha256': VOCABULARY_SHA256},
        'protocol': {'epochs': args.epochs, 'last_encoder_layers': args.last_layers,
                     'learning_rate': args.learning_rate, 'seed': args.seed,
                     'beam_size': args.beam_size, 'ctc_weight': args.ctc_weight,
                     'length_bonus': 0.0, 'external_lm': False, 'loss': 'CTC only',
                     'fixed_schedule_no_epoch_selection': True, 'test_scored_once_per_baseline_and_candidate': True,
                     'heldout_results_not_used_for_training': True},
        'partitions': evidence, 'vocabulary_diagnostics': unsupported,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _progress('loading_and_warmup', 0, 1)
    reader = _reader(Path(args.checkpoint), Path(args.source_dir), args.beam_size, args.ctc_weight, args.device)
    reader.warmup()
    _progress('loading_and_warmup', 1, 1)
    ids = {token: index for index, token in enumerate(reader.token_list)}
    prepared = []
    _progress('preparing', 0, len(datasets['train'].samples))
    for sample in datasets['train'].samples:
        rois, _, _ = _visual_input(sample, reader)
        targets = [ids[unit] for unit in target_units(sample.reference)]
        if len(rois) < len(targets) + sum(a == b for a, b in zip(targets, targets[1:])):
            raise ValueError(f'{sample.id}: too few frames for the complete target')
        prepared.append((sample.id, rois, targets))
        if len(prepared) % 20 == 0 or len(prepared) == len(datasets['train'].samples):
            _progress('preparing', len(prepared), len(datasets['train'].samples))
    baseline = {role: summarize(datasets[role], _predict(reader, datasets[role], f'baseline:{role}'))
                for role in ('dev', 'test')}
    # Preserve all baseline predictions and immutable protocol/data evidence
    # before the first optimizer step, even if an implementation error follows.
    baseline_report = {**common_evidence, 'status': 'baseline_complete', 'training_started': False,
                       'activation': {'automatic_activation': False,
                                      'eligible_for_manual_research_evaluation': False,
                                      'product_readiness_established': False}, 'baseline': baseline}
    (output / 'baseline.json').write_text(json.dumps(baseline_report, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + '\n', encoding='utf-8')

    # Freeze decoder/CTC head. The head follows the encoder device for logits;
    # only the CTC loss itself runs on CPU, with the autograd connection intact.
    reader.model.ctc.to(reader.enc_device)
    named = configure_trainable(reader.model, args.last_layers)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
    optimizer = torch.optim.AdamW(list(named.values()), lr=args.learning_rate, weight_decay=1e-4)
    epoch_rows = []
    started = time.monotonic()
    for epoch in range(args.epochs):
        configure_trainable(reader.model, args.last_layers)
        random.shuffle(prepared)
        total = 0.0
        _progress('training', 0, len(prepared), epoch=epoch + 1)
        for step, (sample_id, rois, targets) in enumerate(prepared, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = ctc_loss(reader, LipReader.to_tensor(rois), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(named.values()), 5.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
            if step % 20 == 0 or step == len(prepared):
                _progress('training', step, len(prepared), epoch=epoch + 1,
                          finite_mean_ctc_loss=total / step)
        row = {'epoch': epoch + 1, 'mean_ctc_loss': total / len(prepared), 'elapsed_seconds': time.monotonic() - started}
        epoch_rows.append(row)
        print(json.dumps(row, allow_nan=False), file=sys.stderr, flush=True)
    reader.model.eval()
    reader.model.ctc.to(reader.device)
    for parameter in reader.model.parameters():
        parameter.requires_grad_(False)
    candidate = {role: summarize(datasets[role], _predict(reader, datasets[role], f'candidate:{role}'))
                 for role in ('dev', 'test')}
    gates = {role: improvement_gate(baseline[role], candidate[role]) for role in ('dev', 'test')}
    eligible = all(gate['improved'] for gate in gates.values())
    changed_count = sum(not torch.equal(initial[name], parameter.detach().cpu())
                        for name, parameter in named.items())
    if eligible and not changed_count:
        raise ValueError('Measured improvement without changed parameters; artifact not saved')
    report = {
        **common_evidence, 'status': 'research_improvement' if eligible else 'rejected',
        'activation': {'automatic_activation': False, 'eligible_for_manual_research_evaluation': eligible,
                       'product_readiness_established': False},
        'training': epoch_rows, 'baseline': baseline, 'candidate': candidate,
        'improvement_gates': gates, 'changed_tensor_count': changed_count,
        'adapter_tensor_count': len(named),
        'limitations': [
            'Relative dev/test improvement does not establish calibrated correctness or webcam usability.',
            'Base-model pretraining overlap requires a separate audit; official corpus splits do not prove it.',
            'General scope requires speaker-disjoint partitions; personal scope proves only one held-out speaker/session protocol.',
            'Changing epochs/decoder/augmentation after observing test results requires a fresh independent test set.',
            'No known diagnostic video is used for training or held-out acceptance.',
            'Every dev/test sample remains in scoring, including references outside the fixed model vocabulary.',
            'Video-only inference on voiced corpus material does not establish recognition of silent articulation.',
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    if eligible:
        # Complete keys for the selected tail make strict reloading possible;
        # no frozen base tensors or optimizer state are included.
        adapter_parameters = {name: parameter.detach().cpu() for name, parameter in named.items()}
        torch.save({'schema_version': 1, 'language': 'zh', 'last_encoder_layers': args.last_layers,
                    'base_checkpoint_sha256': CHECKPOINT_SHA256,
                    'adapter_parameters': adapter_parameters, 'metadata': report}, output / 'encoder_adapter.pth')
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accept-research-license', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--dev-manifest', required=True)
    parser.add_argument('--test-manifest', required=True)
    parser.add_argument('--scope', choices=('personal', 'general'), default='general')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--last-layers', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--beam-size', type=int, default=40)
    parser.add_argument('--ctc-weight', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not args.accept_research_license:
        parser.error('Read the author research license before passing --accept-research-license')
    if args.epochs < 1 or args.last_layers < 1 or args.beam_size < 1:
        parser.error('epochs, last-layers and beam-size must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate must be finite and positive')
    if not math.isfinite(args.ctc_weight) or not 0 <= args.ctc_weight <= 1:
        parser.error('ctc-weight must be finite within [0,1]')
    try:
        report = run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'status': report['status'], 'activation': report['activation'],
                      'improvement_gates': report['improvement_gates']}, ensure_ascii=False, indent=2))
    return 0 if report['activation']['eligible_for_manual_research_evaluation'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
