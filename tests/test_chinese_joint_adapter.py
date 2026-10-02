"""Tiny CPU gradients and dev-only checkpoint selection, with no real model/media."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_chinese_joint_adapter import (adaptation_loss, configure_joint_trainable,
                                        should_select, train_dev_issues)
from lipflow.evaluation import Dataset, Sample
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = torch.nn.Linear(4, 4)
        self.encoders = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.after_norm = torch.nn.LayerNorm(4)

    def forward(self, x, masks):
        x = self.frontend(x)
        for block in self.encoders:
            x = torch.tanh(block(x))
        return self.after_norm(x), masks


class _Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 4)
        self.output = torch.nn.Linear(4, 5)
        self.seen = []

    def forward(self, targets, mask, memory, memory_mask):
        self.seen.append(targets.detach().clone())
        return self.output(self.embedding(targets) + memory.mean(1, keepdim=True)), mask


def _reader():
    model = torch.nn.Module()
    model.encoder = _Encoder()
    model.ctc = torch.nn.Module()
    model.ctc.ctc_lo = torch.nn.Linear(4, 5)
    model.decoder, model.r_decoder = _Decoder(), _Decoder()
    model.criterion = LabelSmoothingLoss(5, -1, .1, normalize_length=False)
    model.sos = model.eos = 4
    model.ignore_id = -1
    return SimpleNamespace(model=model, enc_device=torch.device('cpu'), device=torch.device('cpu'))


def test_joint_gradients_change_tail_only_and_teacher_force_reversed_labels():
    torch.manual_seed(1)
    reader = _reader()
    before = {name: tensor.clone() for name, tensor in reader.model.state_dict().items()}
    named = configure_joint_trainable(reader.model, 2)
    loss, parts = adaptation_loss(reader, torch.randn(12, 4), [1, 2, 3], 'joint')
    assert set(parts) == {'ctc', 'forward_attention', 'reverse_attention'}
    assert float(loss.detach()) == pytest.approx(.1 * parts['ctc'] + .63 * parts['forward_attention']
                                                + .27 * parts['reverse_attention'])
    assert reader.model.decoder.seen[0].tolist() == [[4, 1, 2, 3]]
    assert reader.model.r_decoder.seen[0].tolist() == [[4, 3, 2, 1]]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in named.values())
    assert any(p.grad.abs().sum() > 0 for p in named.values())
    assert all(p.grad is None for name, p in reader.model.named_parameters() if name not in named)
    torch.optim.SGD(named.values(), lr=.1).step()
    after = reader.model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in named)
    assert all(torch.equal(value, before[name]) for name, value in after.items() if name not in named)


def test_attention_has_consistent_per_output_token_normalization():
    reader = _reader()
    configure_joint_trainable(reader.model)
    inputs, labels = torch.randn(12, 4), [1, 2, 3]
    _, parts = adaptation_loss(reader, inputs, labels)
    memory, _ = reader.model.encoder(inputs.unsqueeze(0), None)
    ys_in, ys_out = torch.tensor([[4, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4]])
    prediction, _ = reader.model.decoder(ys_in, None, memory, None)
    expected = reader.model.criterion(prediction, ys_out) / 4
    assert parts['forward_attention'] == pytest.approx(float(expected.detach()))


def test_ctc_only_does_not_teacher_force_any_decoder():
    reader = _reader()
    configure_joint_trainable(reader.model)
    loss, parts = adaptation_loss(reader, torch.randn(12, 4), [1, 2], 'ctc')
    assert set(parts) == {'ctc'}
    assert reader.model.decoder.seen == reader.model.r_decoder.seen == []
    loss.backward()


def test_impossible_ctc_targets_fail_instead_of_dropping_or_zeroing():
    reader = _reader()
    configure_joint_trainable(reader.model)
    with pytest.raises(ValueError, match='Too few'):
        adaptation_loss(reader, torch.randn(3, 4), [1, 1, 1])


def test_baseline_is_kept_for_ties_failures_and_quality_regressions():
    base = dict(cer=.7, exact_sentence_rate=.1, non_rejected_coverage=.9, inference_failures=0)
    good = {**base, 'cer': .6}
    assert should_select(base, base, good)
    assert not should_select(base, good, good)  # Earliest equal-CER epoch wins.
    assert not should_select(base, base, {**good, 'non_rejected_coverage': .8})
    assert not should_select(base, base, {**good, 'exact_sentence_rate': .0})
    assert not should_select(base, base, {**good, 'inference_failures': 1})
    assert not should_select(base, base, base)


def test_train_dev_only_checker_reuses_overlap_rules_and_rejects_test_role():
    def sample(role):
        return Sample(role, '/clips/' + role, '今天开会' if role == 'train' else '明天见面',
                      role, role, 'author', True, 'mouth_roi', True, sha256=role, split=role)
    datasets = {role: Dataset((sample(role),), split=role) for role in ('train', 'dev')}
    assert train_dev_issues(datasets, 'general') == []
    from dataclasses import replace
    datasets['dev'] = Dataset((replace(sample('dev'), speaker='train'),), split='dev')
    assert any('distinct speakers' in issue for issue in train_dev_issues(datasets, 'general'))
    with pytest.raises(ValueError, match='Only train and dev'):
        train_dev_issues({**datasets, 'test': Dataset((), split='test')}, 'general')


def test_inference_position_caches_are_cloned_for_both_decoders():
    reader = _reader()
    with torch.inference_mode():
        reader.model.decoder.pe = torch.randn(1, 5, 4)
        reader.model.r_decoder.pe = torch.randn(1, 5, 4)
    assert reader.model.decoder.pe.is_inference()
    configure_joint_trainable(reader.model)
    assert not reader.model.decoder.pe.is_inference()
    assert not reader.model.r_decoder.pe.is_inference()


def test_cli_rejects_test_manifest_and_more_than_six_epochs():
    from train_chinese_joint_adapter import main
    base = ['--accept-research-license', '--checkpoint', 'base', '--source-dir', 'source',
            '--train-manifest', 'train', '--dev-manifest', 'dev', '--output', 'output']
    with pytest.raises(SystemExit) as caught:
        main(base + ['--test-manifest', 'test'])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        main(base + ['--epochs', '7'])
    assert caught.value.code == 2


@pytest.mark.parametrize('interrupt', [False, True, 'hard_exit', 'budget_stop'])
def test_train_dev_workflow_saves_earliest_best_and_survives_interrupt(tmp_path, monkeypatch, interrupt):
    import json
    import numpy as np
    import evaluate_chinese
    import train_chinese_joint_adapter as trainer
    from evaluate_cnvsrc import _load_adapter
    from lipflow.evaluation import Prediction
    from lipflow.vsr import LipReader

    torch.manual_seed(3)
    datasets = {}
    paths = {}
    for role, reference in [('train', '甲乙'), ('dev', '甲丙')]:
        sample = Sample(role, '/clips/' + role, reference, role, role,
                        'author', True, 'mouth_roi', True, sha256=role, split=role)
        datasets[role] = Dataset((sample,), split=role)
        path = tmp_path / (role + '.json')
        path.write_text(json.dumps({'source': {'revision': 'pinned'}}), encoding='utf-8')
        paths[role] = str(path)
    source = tmp_path / 'source' / 'datamodule'
    source.mkdir(parents=True)
    (source / 'char_units.txt').write_text('甲 1\n乙 2\n丙 3\n', encoding='utf-8')
    reader = _reader()
    reader.token_list = ['<blank>', '甲', '乙', '丙', '<eos>']
    reader.language = 'zh'
    reader.warmup = lambda: None
    loaded, scored, visual_reads = [], [], []
    output = tmp_path / 'run'
    abort_after_resume = False

    def load(path):
        role = Path(path).stem
        loaded.append(role)
        return datasets[role]

    def predict(subject, dataset, stage):
        scored.append(stage)
        assert dataset.split == 'dev'
        raw = '' if stage == 'dev:epoch0' else '甲'
        return [Prediction('dev', raw, .48, .01, action='retry' if not raw else 'review')]

    def visual(sample, subject):
        visual_reads.append(sample.split)
        assert sample.split == 'train'
        if abort_after_resume:
            saved = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            assert saved['status'] == 'resumed_at_complete_epoch'
            assert saved['selected_epoch'] == 1
            assert saved['resume_events'][0]['from_complete_epoch'] == 1
            raise SystemExit(123)
        if len(visual_reads) == 2:
            # The preceding complete-epoch adapter must already be durable,
            # even when execution bypasses all normal end-of-run saves.
            saved = json.loads((output / 'report.json').read_text(encoding='utf-8'))
            assert saved['selected_epoch'] == 1
            artifact = torch.load(output / 'encoder_adapter.pth', weights_only=True)
            assert artifact['metadata']['selected_epoch'] == 1
            assert saved['selected_artifact']['selected_epoch'] == 1
            import hashlib
            assert saved['selected_artifact']['sha256'] == hashlib.sha256(
                (output / 'encoder_adapter.pth').read_bytes()).hexdigest()
            if interrupt == 'hard_exit':
                raise SystemExit(99)
            if interrupt:
                raise KeyboardInterrupt
        return np.zeros((12, 96, 96), dtype=np.uint8), .48, None

    monkeypatch.setattr(trainer, 'load_manifest', load)
    monkeypatch.setattr(trainer, '_reader', lambda *args: reader)
    monkeypatch.setattr(trainer, '_verified_file', lambda *args: 'verified')
    monkeypatch.setattr(trainer, '_predict', predict)
    monkeypatch.setattr(evaluate_chinese, '_visual_input', visual)
    monkeypatch.setattr(LipReader, 'to_tensor', staticmethod(lambda rois: torch.ones(12, 4)))
    args = SimpleNamespace(output=str(output), train_manifest=paths['train'], dev_manifest=paths['dev'],
                           scope='general', source_dir=str(source.parent), checkpoint='base',
                           device='cpu', loss='ctc', epochs=2, last_layers=2, learning_rate=.01,
                           seed=0, beam_size=1, ctc_weight=.5, max_seconds=1800)
    if interrupt == 'hard_exit':
        with pytest.raises(SystemExit) as caught:
            trainer.run(args)
        assert caught.value.code == 99
        report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    else:
        report = trainer.run(args)
    assert loaded == ['train', 'dev']
    assert all(split == 'train' for split in visual_reads)
    assert report['selected_epoch'] == 1
    assert report['candidate']['cer'] == .5
    assert report['status'] == ('training' if interrupt == 'hard_exit'
                                else 'interrupted' if interrupt else 'complete')
    assert scored == (['dev:epoch0', 'dev:epoch1'] if interrupt
                      else ['dev:epoch0', 'dev:epoch1', 'dev:epoch2'])
    assert report['protocol']['test_loaded_or_scored'] is False
    assert report['activation']['automatic_activation'] is False
    assert report['activation']['eligible_for_manual_research_evaluation'] is True
    assert json.loads((output / 'baseline.json').read_text(encoding='utf-8'))['selected_epoch'] == 0
    assert json.loads((output / 'report.json').read_text(encoding='utf-8'))['selected_epoch'] == 1
    fresh_reader = _reader()
    fresh_reader.language = 'zh'
    provenance = _load_adapter(fresh_reader, output / 'encoder_adapter.pth')
    assert provenance['strict_validation'] is True
    assert provenance['last_encoder_layers'] == 2
    if interrupt == 'budget_stop':
        args.resume = True
        ledger = json.loads((output / 'active_budget.json').read_text(encoding='utf-8'))
        ledger['active_budget_elapsed_seconds'] = 1900.0
        (output / 'active_budget.json').write_text(json.dumps(ledger), encoding='utf-8')
        torch.manual_seed(3)
        resumed_reader = _reader()
        resumed_reader.token_list = reader.token_list
        resumed_reader.language = 'zh'
        resumed_reader.warmup = lambda: None
        monkeypatch.setattr(trainer, '_reader', lambda *args: resumed_reader)
        reads_before = list(visual_reads)
        stopped = trainer.run(args)
        assert stopped['status'] == 'time_limit'
        assert stopped['selected_epoch'] == 1
        assert len(stopped['training']) == 1
        assert stopped['partial_epoch']['completed_steps'] == 0
        assert stopped['timing']['soft_budget_elapsed_seconds'] >= 1900
        assert visual_reads == reads_before
    if interrupt is True:
        # A protocol mismatch fails before loading the model or mutating files.
        args.resume = True
        original_rate = args.learning_rate
        args.learning_rate = original_rate * 2
        before_report = (output / 'report.json').read_bytes()
        with pytest.raises(ValueError, match='identity mismatch'):
            trainer.run(args)
        assert (output / 'report.json').read_bytes() == before_report
        args.learning_rate = original_rate

        original_identity = trainer._implementation_identity()
        changed_identity = {**original_identity, 'runtime_versions': {
            **original_identity['runtime_versions'], 'torch': 'changed-version'}}
        with monkeypatch.context() as patch:
            patch.setattr(trainer, '_implementation_identity', lambda: changed_identity)
            with pytest.raises(ValueError, match='identity mismatch'):
                trainer.run(args)
        changed_identity = {**original_identity, 'local_source_sha256': {
            **original_identity['local_source_sha256'], 'lipflow/vsr.py': 'changed-code'}}
        with monkeypatch.context() as patch:
            patch.setattr(trainer, '_implementation_identity', lambda: changed_identity)
            with pytest.raises(ValueError, match='identity mismatch'):
                trainer.run(args)
        assert (output / 'report.json').read_bytes() == before_report

        # Simulate a later, uncommitted canonical artifact. The immutable epoch
        # artifact lets resume restore the matching best model safely.
        (output / 'encoder_adapter.pth').unlink()
        (output / 'encoder_adapter.pth').write_bytes(b'uncommitted future artifact')
        future_report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        future_report['selected_epoch'] = 2
        (output / 'report.json').write_text(json.dumps(future_report), encoding='utf-8')
        torch.manual_seed(3)
        resumed_reader = _reader()
        resumed_reader.token_list = reader.token_list
        resumed_reader.language = 'zh'
        resumed_reader.warmup = lambda: None
        monkeypatch.setattr(trainer, '_reader', lambda *args: resumed_reader)
        abort_after_resume = True
        with pytest.raises(SystemExit) as caught:
            trainer.run(args)
        assert caught.value.code == 123
        restored = json.loads((output / 'report.json').read_text(encoding='utf-8'))
        assert restored['selected_epoch'] == 1
        assert restored['status'] == 'resumed_at_complete_epoch'
        abort_after_resume = False
        torch.manual_seed(3)
        resumed_reader = _reader()
        resumed_reader.token_list = reader.token_list
        resumed_reader.language = 'zh'
        resumed_reader.warmup = lambda: None
        resumed = trainer.run(args)
        assert resumed['status'] == 'complete'
        assert len(resumed['training']) == 2
        assert resumed['total_completed_training_steps'] == 2
        assert resumed['selected_epoch'] == 1
        assert resumed['resume_events'][0]['from_complete_epoch'] == 1
        assert resumed['resume_events'][0]['active_budget_carried_seconds'] > 0
        assert resumed['resume_events'][0]['partial_epoch_updates_discarded'] is True
        assert scored == ['dev:epoch0', 'dev:epoch1', 'dev:epoch2']
        _load_adapter(resumed_reader, output / 'encoder_adapter.pth')
        checkpoint = torch.load(output / 'training_resume.pth', weights_only=True)
        assert checkpoint['completed_epoch'] == 2
        assert checkpoint['training_order'] == ['train']
        assert checkpoint['optimizer']['state']
        assert checkpoint['rng']['backend'] == 'cpu'


def test_sigint_is_deferred_until_atomic_epoch_commit_finishes():
    import signal
    from train_chinese_joint_adapter import _defer_sigint
    events = []
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda number, frame: events.append('interrupt'))
    try:
        with _defer_sigint():
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            assert events == []
            events.append('committed')
        assert events == ['committed', 'interrupt']
    finally:
        signal.signal(signal.SIGINT, previous)


def test_resume_rng_reproduces_python_numpy_and_torch_next_draws():
    import numpy as np
    import random
    from train_chinese_joint_adapter import _rng_state, _restore_rng
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    state = _rng_state(torch.device('cpu'))
    expected = (random.random(), float(np.random.random()), torch.rand(4))
    _restore_rng(state, torch.device('cpu'))
    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(4), expected[2])


def test_complete_epoch_resume_requires_all_adam_moments():
    from train_chinese_joint_adapter import _validate_optimizer
    reader = _reader()
    named = configure_joint_trainable(reader.model)
    optimizer = torch.optim.AdamW(named.values(), lr=1e-5, weight_decay=1e-4)
    empty = optimizer.state_dict()
    _validate_optimizer(empty, optimizer, named, 0)
    with pytest.raises(ValueError, match='every named moment'):
        _validate_optimizer(empty, optimizer, named, 1)
    loss, _ = adaptation_loss(reader, torch.randn(12, 4), [1, 2], 'ctc')
    loss.backward()
    optimizer.step()
    saved = optimizer.state_dict()
    _validate_optimizer(saved, optimizer, named, 1)
    saved['state'].pop(next(iter(saved['state'])))
    with pytest.raises(ValueError, match='every named moment'):
        _validate_optimizer(saved, optimizer, named, 1)


@pytest.mark.parametrize('location', ['checkpoint', 'ledger'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1.0, True])
def test_resume_rejects_each_invalid_budget_before_max(tmp_path, location, invalid):
    import json
    from train_chinese_joint_adapter import _identity_hash, _load_resume
    identity = {'protocol': {'maximum_epochs': 1}}
    digest = _identity_hash(identity)
    checkpoint = {'schema_version': 1, 'kind': 'mandarin-complete-epoch-resume',
                  'identity': identity, 'identity_sha256': digest, 'completed_epoch': 0,
                  'report': {'training': [], 'selected_epoch': 0},
                  'active_budget_elapsed_seconds': invalid if location == 'checkpoint' else 10.0}
    torch.save(checkpoint, tmp_path / 'training_resume.pth')
    (tmp_path / 'active_budget.json').write_text(json.dumps({
        'identity_sha256': digest,
        'active_budget_elapsed_seconds': invalid if location == 'ledger' else 10.0}), encoding='utf-8')
    with pytest.raises(ValueError, match='Invalid resume active budget'):
        _load_resume(tmp_path, identity)
