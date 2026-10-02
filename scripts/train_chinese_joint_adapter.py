"""Train/dev-only, memory-bounded Mandarin encoder adaptation for research.

This script never loads a test partition. It selects the earliest best raw dev
CER subject to coverage, exact-match and failure gates against epoch zero.
The selected artifact is compatible with evaluate_cnvsrc._load_adapter, but is
never automatically activated. Source code/model research restrictions apply.

Example (all files already local):
  python scripts/train_chinese_joint_adapter.py --accept-research-license \
    --checkpoint /local/model_avg_cncvs_2_3_cnvsrc.pth --source-dir /local/CNVSRC2025 \
    --train-manifest train-supported.json --dev-manifest dev.json --output /local/run
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import threading
import time

from lipflow.evaluation import Dataset, load_manifest
from evaluate_cnvsrc import (CHECKPOINT_SHA256, CONFIG_SHA256, SOURCE_REVISION,
                            VOCABULARY_SHA256, _reader, _verified_file)
from train_chinese_adapter import (_manifest_evidence, _predict, _progress,
                                  configure_trainable, improvement_gate,
                                  partition_issues, summarize, target_units,
                                  vocabulary_errors)


def train_dev_issues(datasets: dict[str, Dataset], scope: str) -> list[str]:
    """Reuse the established overlap checks without accessing any test data."""
    if set(datasets) != {'train', 'dev'}:
        raise ValueError('Only train and dev partitions are accepted')
    # An empty third role lets the existing checker validate every train/dev
    # identity, reference and video pair. It introduces no test samples.
    checked = {**datasets, 'test': Dataset((), split='test')}
    return [issue for issue in partition_issues(checked, scope)
            if issue != 'test: partition is empty']


def configure_joint_trainable(model, last_layers: int = 2):
    """Keep all inherited buffers/heads frozen and repair inference caches."""
    import torch

    with torch.inference_mode(False):
        for module in model.modules():
            cache = getattr(module, 'pe', None)
            if isinstance(cache, torch.Tensor) and cache.is_inference():
                module.pe = cache.clone()
            if hasattr(module, '_mem_kv'):
                del module._mem_kv
    return configure_trainable(model, last_layers)


def _ctc_encoded_loss(reader, encoded, target_ids: list[int]):
    import torch
    import torch.nn.functional as functional

    logits = reader.model.ctc.ctc_lo(encoded)
    probabilities = logits.float().log_softmax(-1).transpose(0, 1).cpu()
    required = len(target_ids) + sum(a == b for a, b in zip(target_ids, target_ids[1:]))
    if not target_ids or probabilities.shape[0] < required:
        raise ValueError('Too few encoder frames or empty complete CTC target')
    return functional.ctc_loss(
        probabilities, torch.tensor(target_ids, dtype=torch.long),
        torch.tensor([probabilities.shape[0]]), torch.tensor([len(target_ids)]),
        blank=0, reduction='sum', zero_infinity=False,
    ) / len(target_ids)


def adaptation_loss(reader, inputs, target_ids: list[int], loss_type: str = 'joint'):
    """One visual encode; CPU losses preserve encoder autograd transfers.

    Teacher forcing receives training references only. Both attention terms are
    normalized per output token (including EOS), while CTC is normalized per
    reference token. Frozen decoders retain gradients with respect to memory.
    """
    import torch
    from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
    from espnet.nets.pytorch_backend.transformer.mask import target_mask

    if loss_type not in ('ctc', 'joint'):
        raise ValueError('loss_type must be ctc or joint')
    encoded, _ = reader.model.encoder(inputs.unsqueeze(0).to(reader.enc_device), None)
    ctc = _ctc_encoded_loss(reader, encoded, target_ids)
    components = {'ctc': ctc}
    loss = ctc
    if loss_type == 'joint':
        memory = encoded.to(reader.device)
        labels = torch.tensor([target_ids], dtype=torch.long, device=reader.device)
        attention_terms = []
        for name, decoder, sequence in (
                ('forward_attention', reader.model.decoder, labels),
                ('reverse_attention', reader.model.r_decoder, labels.flip(1))):
            ys_in, ys_out = add_sos_eos(sequence, reader.model.sos,
                                      reader.model.eos, reader.model.ignore_id)
            prediction, _ = decoder(ys_in, target_mask(ys_in, reader.model.ignore_id), memory, None)
            term = reader.model.criterion(prediction, ys_out)
            # The pinned criterion is batch-normalized, not token-normalized.
            if not reader.model.criterion.normalize_length:
                term = term / (len(target_ids) + 1)
            components[name] = term
            attention_terms.append(term.to(ctc.device))
        loss = .1 * ctc + .9 * (.7 * attention_terms[0] + .3 * attention_terms[1])
    if not all(torch.isfinite(term).all() for term in (loss, *components.values())):
        raise ValueError('Non-finite adaptation loss; stopped without dropping a sample')
    return loss, {name: float(term.detach()) for name, term in components.items()}


def should_select(baseline: dict, best: dict, candidate: dict) -> bool:
    """Strictly lower CER selects the earliest tied epoch; no quality regressions."""
    return (improvement_gate(baseline, candidate)['improved']
            and candidate['cer'] < best['cer'])


def _tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def _defer_sigint():
    """Commit one best copy/its metadata before delivering a pending Ctrl-C.

    CLI training runs on the main thread. Other threads cannot receive Python's
    SIGINT handler and therefore need no handler replacement.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending = []
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda number, frame: pending.append((number, frame)))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if pending:
            if callable(previous):
                previous(*pending[0])
            elif previous != signal.SIG_IGN:
                raise KeyboardInterrupt


def _capture_parameters(named) -> dict:
    return {name: parameter.detach().cpu().clone() for name, parameter in named.items()}


def _write_json(path: Path, value: dict):
    """Atomic latest status; an interruption cannot replace it with partial JSON."""
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save_adapter(path: Path, named, metadata: dict, last_layers: int):
    import torch

    descriptor, temporary = tempfile.mkstemp(prefix='.adapter-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save({'schema_version': 1, 'language': 'zh', 'last_encoder_layers': last_layers,
                        'base_checkpoint_sha256': CHECKPOINT_SHA256,
                        'adapter_parameters': named, 'metadata': metadata}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _persist_selection(output: Path, named, best_parameters, initial_hashes,
                       report: dict, last_layers: int, *, save_adapter: bool):
    """Save a completed-epoch choice before starting any further training.

    The adapter is independently atomic and contains its own selection evidence.
    The report also records its hash, so a crash between the two atomic renames
    is detectable. A baseline-only run produces no adapted model artifact.
    """
    changed = sum(_tensor_hash(parameter) != initial_hashes[name]
                  for name, parameter in best_parameters.items())
    eligible = (report['selected_epoch'] > 0 and changed > 0
                and improvement_gate(report['baseline'], report['candidate'])['improved'])
    report.update(adapter_tensor_count=len(named), changed_tensor_count=changed,
                  improvement_gate=improvement_gate(report['baseline'], report['candidate']))
    report['activation'] = {**report['activation'],
                            'eligible_for_manual_research_evaluation': eligible}
    if eligible and save_adapter:
        artifact = output / 'encoder_adapter.pth'
        boundary_artifact = output / f'encoder_adapter_epoch{report["selected_epoch"]}.pth'
        # Do not embed a stale artifact hash, or attempt a self-referential hash.
        metadata = {key: value for key, value in report.items() if key != 'selected_artifact'}
        _save_adapter(boundary_artifact, best_parameters, metadata, last_layers)
        _publish_adapter(boundary_artifact, artifact)
        report['selected_artifact'] = {
            'path': artifact.name, 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
            'boundary_path': boundary_artifact.name,
            'selected_epoch': report['selected_epoch'], 'durable_after_complete_epoch': True,
        }
    _write_json(output / 'report.json', report)


def _publish_adapter(boundary: Path, canonical: Path):
    """Atomic hard link keeps the resume boundary immutable with no duplicate bytes."""
    descriptor, temporary = tempfile.mkstemp(prefix='.selected-', dir=canonical.parent)
    os.close(descriptor)
    Path(temporary).unlink()
    try:
        os.link(boundary, temporary)
        os.replace(temporary, canonical)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _identity_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _implementation_identity() -> dict:
    """Bind a continuation to the local executable protocol and library versions."""
    import cv2
    import numpy as np
    import torch
    root = Path(__file__).resolve().parents[1]
    sources = ('scripts/train_chinese_joint_adapter.py', 'scripts/train_chinese_adapter.py',
               'scripts/evaluate_cnvsrc.py', 'scripts/evaluate_chinese.py',
               'lipflow/vsr.py', 'lipflow/evaluation.py', 'lipflow/confidence.py')
    return {
        'local_source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in sources},
        'runtime_versions': {'python_major_minor': f'{sys.version_info.major}.{sys.version_info.minor}',
                             'torch': str(torch.__version__), 'numpy': str(np.__version__),
                             'opencv': str(cv2.__version__)},
    }


def _cpu_tree(value):
    """Use only tensors and primitive containers accepted by weights_only load."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported resume state type: {type(value).__name__}')


def _rng_state(device) -> dict:
    import numpy as np
    import torch
    numpy_state = np.random.get_state()
    state = {'python': random.getstate(),
             'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
             'torch_cpu': torch.get_rng_state(), 'backend': str(device)}
    if device.type == 'mps':
        state['torch_device'] = torch.mps.get_rng_state().cpu()
    elif device.type == 'cuda':
        state['torch_device'] = torch.cuda.get_rng_state(device).cpu()
    return state


def _restore_rng(state: dict, device):
    import numpy as np
    import torch
    if state.get('backend') != str(device):
        raise ValueError('Resume encoder backend differs from the saved experiment')
    random.setstate(state['python'])
    numpy_state = state['numpy']
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'mps':
        torch.mps.set_rng_state(state['torch_device'])
    elif device.type == 'cuda':
        torch.cuda.set_rng_state(state['torch_device'], device)


def _validate_optimizer(saved: dict, optimizer, named, completed_epoch: int):
    """Validate Adam metadata/moments before applying the saved tail tensors."""
    import torch
    expected = optimizer.state_dict()
    if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
        raise ValueError('Invalid resume optimizer state')
    if saved['param_groups'] != expected['param_groups']:
        raise ValueError('Resume optimizer protocol/parameter order mismatch')
    parameters = dict(zip(expected['param_groups'][0]['params'], named.values()))
    if not isinstance(saved['state'], dict) or not set(saved['state']) <= set(parameters):
        raise ValueError('Resume optimizer parameter identities mismatch')
    if ((completed_epoch == 0 and saved['state'])
            or (completed_epoch > 0 and set(saved['state']) != set(parameters))):
        raise ValueError('Resume optimizer must contain every named moment after a complete epoch')
    for index, state in saved['state'].items():
        if not isinstance(state, dict) or set(state) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Invalid resume Adam moments')
        parameter = parameters[index]
        for key, tensor in state.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or not torch.isfinite(tensor).all()):
                raise ValueError('Invalid resume optimizer tensors')
            if key == 'step':
                if tensor.numel() != 1 or tensor.item() < 0:
                    raise ValueError('Invalid resume optimizer step')
            elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                raise ValueError('Resume optimizer moment shape/dtype mismatch')


def _save_resume(path: Path, identity: dict, reader, named, optimizer, training,
                 initial_hashes: dict, report: dict, elapsed: float):
    """A full epoch boundary; partial-epoch optimizer updates are never saved."""
    import torch
    state = {'schema_version': 1, 'kind': 'mandarin-complete-epoch-resume',
             'identity': identity, 'identity_sha256': _identity_hash(identity),
             'completed_epoch': len(report['training']),
             'current_parameters': _capture_parameters(named),
             'optimizer': _cpu_tree(optimizer.state_dict()),
             'rng': _rng_state(reader.enc_device),
             'training_order': [sample.id for sample, targets in training],
             'initial_hashes': initial_hashes, 'report': report,
             'active_budget_elapsed_seconds': elapsed}
    descriptor, temporary = tempfile.mkstemp(prefix='.resume-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_resume(output: Path, identity: dict) -> tuple[dict, float]:
    """Check identities and the budget ledger before touching experiment files."""
    import torch
    state = torch.load(output / 'training_resume.pth', map_location='cpu', weights_only=True)
    digest = _identity_hash(identity)
    if (state.get('schema_version') != 1 or state.get('kind') != 'mandarin-complete-epoch-resume'
            or state.get('identity') != identity or state.get('identity_sha256') != digest):
        raise ValueError('Resume dataset/model/source/protocol identity mismatch')
    report = state['report']
    epoch = state['completed_epoch']
    if (type(epoch) is not int or not 0 <= epoch <= identity['protocol']['maximum_epochs']
            or len(report['training']) != epoch
            or [row['epoch'] for row in report['training']] != list(range(1, epoch + 1))
            or not 0 <= report['selected_epoch'] <= epoch):
        raise ValueError('Resume checkpoint does not describe a complete epoch boundary')
    def checked_budget(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('Invalid resume active budget')
        return float(value)

    used = checked_budget(state['active_budget_elapsed_seconds'])
    ledger_path = output / 'active_budget.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get('identity_sha256') != digest:
            raise ValueError('Resume active-budget identity mismatch')
        used = max(used, checked_budget(ledger['active_budget_elapsed_seconds']))
    artifact = report.get('selected_artifact')
    if report['selected_epoch'] > 0:
        if not artifact or artifact['path'] != 'encoder_adapter.pth':
            raise ValueError('Resume best adapter evidence is missing')
        if artifact.get('boundary_path') != f'encoder_adapter_epoch{report["selected_epoch"]}.pth':
            raise ValueError('Resume best adapter boundary path is invalid')
        if hashlib.sha256((output / artifact['boundary_path']).read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError('Resume best adapter hash mismatch')
    return state, float(used)


def run(args) -> dict:
    import numpy as np
    import torch
    from evaluate_chinese import _visual_input
    from lipflow.vsr import LipReader

    output = Path(args.output)
    resuming = getattr(args, 'resume', False)
    if not resuming and output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to preserve previous experiments')
    paths = {'train': args.train_manifest, 'dev': args.dev_manifest}
    datasets = {role: load_manifest(path) for role, path in paths.items()}
    issues = train_dev_issues(datasets, args.scope)
    if issues:
        raise ValueError('Invalid train/dev protocol:\n' + '\n'.join(issues))
    source = Path(args.source_dir)
    source = source / 'VSR' if (source / 'VSR').is_dir() else source
    vocabulary = source / 'datamodule/char_units.txt'
    _verified_file(vocabulary, VOCABULARY_SHA256)
    tokens = ['<blank>'] + [line.split()[0] for line in vocabulary.read_text(encoding='utf-8').splitlines()] + ['<eos>']
    unsupported = vocabulary_errors(datasets, tokens)
    untrainable = [row for row in unsupported if row['partition'] == 'train']
    if untrainable:
        raise ValueError('Unsupported train targets; use explicit train-only preparation:\n' +
                         json.dumps(untrainable, ensure_ascii=False))
    protocol = {
        'maximum_epochs': args.epochs, 'last_encoder_layers': args.last_layers,
        'learning_rate': args.learning_rate, 'seed': args.seed, 'weight_decay': 1e-4,
        'gradient_clip_norm': 5.0, 'batch_size': 1, 'loss': args.loss,
        'joint_weights': {'ctc': .1, 'forward_attention': .63, 'reverse_attention': .27},
        'loss_normalization': 'CTC per reference token; attention per target token including EOS',
        'beam_size': args.beam_size, 'ctc_weight': args.ctc_weight, 'length_bonus': 0.0,
        'external_lm': False, 'selection': 'earliest strictly lowest raw dev CER with baseline quality gates',
        'baseline_epoch_zero_included': True, 'test_loaded_or_scored': False,
        'maximum_seconds': args.max_seconds, 'prepared_rois_cached': False,
        'budget_scope': ('Soft wall-clock budget from training start, including dev evaluations; '
                         'model startup and baseline excluded. Checked before training steps; '
                         'an active step or full-epoch dev evaluation may exceed the limit.'),
        'heads_and_decoders_frozen': True, 'batchnorm_statistics_frozen': True,
        'checkpoint_persistence': 'Atomic selected adapter and report after every completed epoch',
        'training_resume_supported': True,
        'resume_scope': 'Complete epoch only; partial epoch updates discarded; cumulative active budget retained',
        'abrupt_kill_budget_limit': 'Runtime since the last budget ledger update may be unrecorded',
    }
    evidence = {role: _manifest_evidence(path, datasets[role]) for role, path in paths.items()}
    common = {
        'schema_version': 1, 'scope': args.scope,
        'base': {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': SOURCE_REVISION,
                 'configuration_sha256': CONFIG_SHA256, 'vocabulary_sha256': VOCABULARY_SHA256},
        'protocol': protocol, 'partitions': evidence, 'vocabulary_diagnostics': unsupported,
        'implementation': _implementation_identity(),
        'activation': {'automatic_activation': False, 'product_readiness_established': False},
        'limitations': [
            'Dev-selected comparative research; no independent test result is claimed by this script.',
            'Voiced mouth-crop training does not establish silent articulation or webcam usability.',
            'Unknown base-model pretraining overlap remains unaudited.',
            'The base model and any derived tensors remain subject to the author research license.',
            'Every dev sample remains in scoring, including units outside the fixed vocabulary.',
        ],
    }
    identity = {key: common[key] for key in ('schema_version', 'scope', 'base', 'protocol',
                                            'partitions', 'implementation')}
    identity['requested_device'] = args.device
    identity_digest = _identity_hash(identity)
    resume_state, budget_used = _load_resume(output, identity) if resuming else (None, 0.0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _progress('loading_and_warmup', 0, 1)
    startup_started = time.monotonic()
    reader = _reader(Path(args.checkpoint), Path(args.source_dir), args.beam_size, args.ctc_weight, args.device)
    reader.warmup()
    startup_seconds = time.monotonic() - startup_started
    _progress('loading_and_warmup', 1, 1)
    if resuming:
        report = resume_state['report']
        baseline = report['baseline']
        if resume_state['rng']['backend'] != str(reader.enc_device):
            raise ValueError('Resume encoder backend differs from the saved experiment')
    else:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / 'protocol.json', common)
        baseline_started = time.monotonic()
        baseline = summarize(datasets['dev'], _predict(reader, datasets['dev'], 'dev:epoch0'))
        common['timing'] = {'model_startup_and_warmup_seconds': startup_seconds,
                            'baseline_dev_seconds': time.monotonic() - baseline_started}
        _write_json(output / 'protocol.json', common)
        report = {**common, 'status': 'baseline_complete', 'baseline': baseline,
                  'candidate': baseline, 'selected_epoch': 0, 'training': [],
                  'adapter_tensor_count': 0, 'changed_tensor_count': 0}
        _write_json(output / 'baseline.json', report)
        _write_json(output / 'report.json', report)
    named = configure_joint_trainable(reader.model, args.last_layers)
    initial_hashes = {name: _tensor_hash(parameter) for name, parameter in named.items()}
    # One best copy plus the current model; no initial third parameter snapshot.
    best_parameters = _capture_parameters(named)
    best, best_epoch = baseline, 0
    token_ids = {token: index for index, token in enumerate(reader.token_list)}
    # Only labels/paths are prepared in memory; decoded ROIs live for one step.
    training = [(sample, [token_ids[unit] for unit in target_units(sample.reference)])
                for sample in datasets['train'].samples]
    optimizer = torch.optim.AdamW(list(named.values()), lr=args.learning_rate, weight_decay=1e-4)
    rows = report['training'] if resuming else []
    first_epoch = 1
    if resuming:
        current = resume_state['current_parameters']
        if set(current) != set(named) or resume_state['initial_hashes'] != initial_hashes:
            raise ValueError('Resume trainable parameters differ from the pinned base')
        for name, parameter in named.items():
            if current[name].shape != parameter.shape or current[name].dtype != parameter.dtype:
                raise ValueError('Resume parameter shape/dtype mismatch')
            if not torch.isfinite(current[name]).all():
                raise ValueError('Resume parameters are not finite')
        by_id = {sample.id: (sample, targets) for sample, targets in training}
        order = resume_state['training_order']
        if len(order) != len(by_id) or set(order) != set(by_id):
            raise ValueError('Resume training order does not match the training manifest')
        _validate_optimizer(resume_state['optimizer'], optimizer, named, resume_state['completed_epoch'])
        if report['selected_epoch'] > 0:
            # Validate the saved best artifact with the production strict schema.
            from evaluate_cnvsrc import _load_adapter
            _load_adapter(reader, output / report['selected_artifact']['boundary_path'])
            best_parameters = _capture_parameters(named)
        with torch.no_grad():
            for name, parameter in named.items():
                parameter.copy_(current[name].to(parameter.device))
        optimizer.load_state_dict(resume_state['optimizer'])
        training = [by_id[sample_id] for sample_id in order]
        _restore_rng(resume_state['rng'], reader.enc_device)
        first_epoch = resume_state['completed_epoch'] + 1
        best, best_epoch = report['candidate'], report['selected_epoch']
        with _defer_sigint():
            if report['selected_epoch'] > 0:
                _publish_adapter(output / report['selected_artifact']['boundary_path'],
                                 output / 'encoder_adapter.pth')
            else:
                # Keep the canonical artifact aligned with the committed boundary.
                (output / 'encoder_adapter.pth').unlink(missing_ok=True)
            report.setdefault('resume_events', []).append({
                'from_complete_epoch': first_epoch - 1, 'active_budget_carried_seconds': budget_used,
                'partial_epoch_updates_discarded': True, 'startup_seconds_excluded': startup_seconds})
            report.update(status='resumed_at_complete_epoch', partial_epoch=None)
            _write_json(output / 'report.json', report)
        del resume_state
    started, stop_status = time.monotonic(), 'complete'
    active_epoch, active_step, active_completed_steps = None, None, 0
    total_completed_steps = len(rows) * len(training)

    def elapsed_budget():
        return budget_used + time.monotonic() - started

    def persist_budget():
        _write_json(output / 'active_budget.json', {
            'identity_sha256': identity_digest,
            'active_budget_elapsed_seconds': elapsed_budget(),
            'idle_downtime_included': False})

    if not resuming:
        with _defer_sigint():
            _save_resume(output / 'training_resume.pth', identity, reader, named, optimizer,
                         training, initial_hashes, report, elapsed_budget())
    persist_budget()
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            active_epoch, active_step, active_completed_steps = epoch, None, 0
            configure_joint_trainable(reader.model, args.last_layers)
            reader.model.ctc.to(reader.enc_device)
            random.shuffle(training)
            total, component_totals = 0.0, {}
            _progress('training', 0, len(training), epoch=epoch)
            for step, (sample, targets) in enumerate(training, 1):
                persist_budget()
                if elapsed_budget() >= args.max_seconds:
                    stop_status = 'time_limit'
                    break
                active_step = step
                rois, _, _ = _visual_input(sample, reader)
                optimizer.zero_grad(set_to_none=True)
                loss, components = adaptation_loss(reader, LipReader.to_tensor(rois), targets, args.loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(named.values()), 5.0, error_if_nonfinite=True)
                optimizer.step()
                active_completed_steps = step
                total_completed_steps += 1
                total += float(loss.detach())
                for key, value in components.items():
                    component_totals[key] = component_totals.get(key, 0.0) + value
                del rois, loss
                if step % 20 == 0 or step == len(training):
                    _progress('training', step, len(training), epoch=epoch, mean_loss=total / step)
            if stop_status == 'time_limit':
                break  # Never select a partial epoch using a smaller unseen schedule.
            reader.model.ctc.to(reader.device)
            reader.model.eval()
            candidate = summarize(datasets['dev'], _predict(reader, datasets['dev'], f'dev:epoch{epoch}'))
            selected = should_select(baseline, best, candidate)
            with _defer_sigint():
                if selected:
                    # A pending Ctrl-C is delivered only after copy and metrics
                    # agree. Releasing first avoids a third full tail snapshot.
                    best_parameters.clear()
                    best_parameters = _capture_parameters(named)
                    best, best_epoch = candidate, epoch
                rows.append({'epoch': epoch, 'mean_loss': total / len(training),
                             'mean_components': {key: value / len(training) for key, value in component_totals.items()},
                             'elapsed_seconds': elapsed_budget(), 'dev': candidate,
                             'selected_at_this_epoch': selected})
                report.update(status='training', training=rows, candidate=best, selected_epoch=best_epoch)
                active_epoch, active_step = None, None
                report.update(total_completed_training_steps=total_completed_steps, partial_epoch=None)
                _persist_selection(output, named, best_parameters, initial_hashes, report,
                                   args.last_layers, save_adapter=selected)
                _save_resume(output / 'training_resume.pth', identity, reader, named, optimizer,
                             training, initial_hashes, report, elapsed_budget())
                persist_budget()
    except KeyboardInterrupt:
        stop_status = 'interrupted'
    except Exception as exc:
        report.update(status='failed', training=rows, candidate=best, selected_epoch=best_epoch,
                      error=f'{type(exc).__name__}: {exc}',
                      total_completed_training_steps=total_completed_steps,
                      partial_epoch={'epoch': active_epoch, 'completed_steps': active_completed_steps,
                                     'attempted_step': active_step, 'total_steps': len(training),
                                     'dev_selection_committed': False})
        _write_json(output / 'report.json', report)
        raise
    finally:
        try:
            persist_budget()
        finally:
            reader.model.ctc.to(reader.device)
            reader.model.eval()
    report.update(status=stop_status, training=rows, candidate=best, selected_epoch=best_epoch,
                  improvement_gate=improvement_gate(baseline, best),
                  total_completed_training_steps=total_completed_steps,
                  partial_epoch=({'epoch': active_epoch, 'completed_steps': active_completed_steps,
                                  'attempted_step': active_step, 'total_steps': len(training),
                                  'dev_selection_committed': False} if active_epoch is not None else None))
    elapsed = elapsed_budget()
    report['timing'].update(soft_budget_elapsed_seconds=elapsed,
                            soft_budget_overrun_seconds=max(0.0, elapsed - args.max_seconds))
    with _defer_sigint():
        _persist_selection(output, named, best_parameters, initial_hashes, report,
                           args.last_layers, save_adapter=False)
    # Restore only after selection, without changing any frozen base tensor.
    with torch.no_grad():
        for name, parameter in named.items():
            parameter.copy_(best_parameters[name].to(parameter.device))
    reader.model.eval()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--accept-research-license', action='store_true')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--train-manifest', required=True)
    parser.add_argument('--dev-manifest', required=True)
    parser.add_argument('--scope', choices=('personal', 'general'), default='general')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--loss', choices=('ctc', 'joint'), default='joint')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--last-layers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--beam-size', type=int, default=40)
    parser.add_argument('--ctc-weight', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--resume', action='store_true',
                        help='Resume this output at its last completed epoch; fixed protocol and cumulative budget')
    args = parser.parse_args(argv)
    if not args.accept_research_license:
        parser.error('Read the author research license before passing --accept-research-license')
    if not 1 <= args.epochs <= 6 or args.last_layers < 1 or args.beam_size < 1:
        parser.error('epochs must be within [1,6], last-layers and beam-size positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('learning-rate must be finite and positive')
    if not math.isfinite(args.ctc_weight) or not 0 <= args.ctc_weight <= 1:
        parser.error('ctc-weight must be finite within [0,1]')
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 1800:
        parser.error('max-seconds must be finite within (0,1800]')
    try:
        report = run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'status': report['status'], 'selected_epoch': report['selected_epoch'],
                      'baseline_cer': report['baseline']['cer'], 'selected_cer': report['candidate']['cer'],
                      'activation': report['activation']}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report['activation']['eligible_for_manual_research_evaluation'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
