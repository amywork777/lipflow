"""Training protocol leakage checks and actual CTC gradient/freeze behavior."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_chinese_adapter import (DIAGNOSTIC_HASHES, _manifest_evidence, configure_trainable, ctc_loss,
                                   improvement_gate, partition_issues, vocabulary_errors)
from lipflow.evaluation import Dataset, Sample


def _sample(identity, speaker, session, reference, split):
    return Sample(identity, f'/clips/{identity}.mp4', reference, speaker, session,
                  'author supplied labels', True, 'mouth_roi', True, sha256=identity, split=split)


def _partitions(scope='general'):
    refs = {'train': '今天开会', 'dev': '明天见面', 'test': '周末休息'}
    return {role: Dataset((_sample(role, role if scope == 'general' else 'one-person',
                                  role, reference, role),), split=role)
            for role, reference in refs.items()}


def test_general_requires_disjoint_speakers():
    datasets = _partitions()
    assert partition_issues(datasets, 'general') == []
    sample = datasets['dev'].samples[0]
    datasets['dev'] = Dataset((_sample(sample.id, 'train', sample.session, sample.reference, 'dev'),), split='dev')
    assert any('distinct speakers' in issue for issue in partition_issues(datasets, 'general'))


def test_personal_requires_separate_sessions():
    datasets = _partitions('personal')
    assert partition_issues(datasets, 'personal') == []
    sample = datasets['dev'].samples[0]
    datasets['dev'] = Dataset((_sample(sample.id, sample.speaker, 'train', sample.reference, 'dev'),), split='dev')
    assert any('separate sessions' in issue for issue in partition_issues(datasets, 'personal'))


def test_renamed_diagnostic_video_is_rejected():
    datasets = _partitions()
    sample = datasets['train'].samples[0]
    from dataclasses import replace
    datasets['train'] = Dataset((replace(sample, sha256=next(iter(DIAGNOSTIC_HASHES))),), split='train')
    assert any('diagnostic' in issue for issue in partition_issues(datasets, 'general'))


def test_overlap_and_repeated_reference_cannot_be_hidden_by_filename():
    datasets = _partitions()
    from dataclasses import replace
    train = datasets['train'].samples[0]
    dev = datasets['dev'].samples[0]
    datasets['dev'] = Dataset((replace(dev, sha256=train.sha256, reference=train.reference + '。'),), split='dev')
    issues = partition_issues(datasets, 'general')
    assert any('overlapping video' in issue for issue in issues)
    assert any('normalized reference' in issue for issue in issues)


def test_diagnostic_split_and_unverified_labels_are_rejected():
    datasets = _partitions()
    from dataclasses import replace
    sample = datasets['train'].samples[0]
    datasets['train'] = Dataset((replace(sample, split='diagnostic', label_verified=False),), split='diagnostic')
    issues = partition_issues(datasets, 'general')
    assert any('split' in issue for issue in issues)
    assert any('provenance' in issue for issue in issues)


def test_unsupported_numbers_names_and_currency_are_reported_without_dropping():
    sample = _sample('one', 'speaker', 'date', '张三，付给 Amy ￥２５０。', 'train')
    dataset = Dataset((sample,), split='train')
    errors = vocabulary_errors({'train': dataset}, ['<blank>', '张', '三', '付', '给', '<eos>'])
    assert errors[0]['sample_id'] == 'one'
    assert set(errors[0]['unsupported_units']) == {'A', 'm', 'y', '¥', '2', '5', '0'}
    assert dataset.samples == (sample,)


class _TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])
        self.after_norm = torch.nn.LayerNorm(4)

    def forward(self, inputs, masks):
        for block in self.encoders:
            inputs = torch.tanh(block(inputs))
        return self.after_norm(inputs), masks


def _tiny_reader():
    model = torch.nn.Module()
    model.encoder = _TinyEncoder()
    model.decoder = torch.nn.Linear(4, 4)
    model.ctc = torch.nn.Module()
    model.ctc.ctc_lo = torch.nn.Linear(4, 4)
    return SimpleNamespace(model=model, enc_device=torch.device('cpu'))


def test_ctc_backprop_changes_only_selected_encoder_parameters():
    torch.manual_seed(0)
    reader = _tiny_reader()
    before = {key: value.detach().clone() for key, value in reader.model.state_dict().items()}
    named = configure_trainable(reader.model)
    assert set(named) == {'encoder.encoders.1.weight', 'encoder.encoders.1.bias',
                          'encoder.after_norm.weight', 'encoder.after_norm.bias'}
    optimizer = torch.optim.SGD(named.values(), lr=.1)
    loss = ctc_loss(reader, torch.randn(12, 4), [1, 2, 1])
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in named.values())
    assert reader.model.ctc.ctc_lo.weight.grad is None
    optimizer.step()
    after = reader.model.state_dict()
    assert any(not torch.equal(after[key], before[key]) for key in named)
    assert all(torch.equal(value, before[key]) for key, value in after.items() if key not in named)


def test_ctc_impossible_alignment_stops_instead_of_zeroing_loss():
    reader = _tiny_reader()
    with pytest.raises(ValueError, match='Too few'):
        ctc_loss(reader, torch.randn(3, 4), [1, 1, 1])


def test_inference_cached_position_encoding_is_safe_for_training_backward():
    from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding

    class PositionalBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position = RelPositionalEncoding(4, 0)
            self.linear = torch.nn.Linear(4, 4)
            self.linear_pos = torch.nn.Linear(4, 4)

        def forward(self, inputs):
            inputs, positions = self.position(inputs)
            return self.linear(inputs) + self.linear_pos(positions[:, :inputs.size(1)])

    reader = _tiny_reader()
    block = PositionalBlock()
    reader.model.encoder.encoders[1] = block
    reader.model.double()
    inputs = torch.randn(12, 4, dtype=torch.float64)
    # A dtype transfer reproduces the first CPU->MPS warmup transfer without a
    # GPU: the cached .pe becomes an inference tensor under inference_mode.
    with torch.inference_mode():
        block.position(inputs.unsqueeze(0))
    original = block.position.pe.clone()
    assert block.position.pe.is_inference()
    with pytest.raises(RuntimeError, match='Inference tensors cannot be saved'):
        block(inputs.unsqueeze(0))
    named = configure_trainable(reader.model)
    assert not block.position.pe.is_inference()
    assert torch.equal(original, block.position.pe)
    loss = ctc_loss(reader, inputs, [1, 2, 1])
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in named.values())


def test_training_exclusions_and_source_provenance_survive_in_report(tmp_path):
    import json

    dataset = _partitions()['train']
    eligibility = {'input_samples': 2, 'kept_samples': 1, 'excluded_samples': 1,
                   'excluded': [{'sample_id': 'unsupported', 'unsupported_units': ['A']}],
                   'references_modified': False, 'dev_test_filtered': False}
    source = {'revision': 'pinned-author-revision', 'license': 'research only'}
    manifest = tmp_path / 'train.json'
    manifest.write_text(json.dumps({'training_eligibility': eligibility, 'source': source}), encoding='utf-8')
    evidence = _manifest_evidence(str(manifest), dataset)
    assert evidence['sample_count'] == 1
    assert evidence['training_eligibility'] == eligibility
    assert evidence['source_metadata'] == source


def test_baseline_evidence_is_saved_before_first_training_step(tmp_path, monkeypatch):
    import json
    import numpy as np
    import evaluate_chinese
    import train_chinese_adapter as trainer
    from lipflow.evaluation import Prediction

    datasets = _partitions()
    manifest_paths = {}
    for role in datasets:
        path = tmp_path / f'{role}.json'
        path.write_text(json.dumps({'source': {'revision': 'source-pin'}}), encoding='utf-8')
        manifest_paths[role] = str(path)
    monkeypatch.setattr(trainer, 'load_manifest', lambda path: datasets[Path(path).stem])
    source = tmp_path / 'source'
    (source / 'datamodule').mkdir(parents=True)
    vocabulary = sorted(set('今天开会明见面周末休息'))
    (source / 'datamodule/char_units.txt').write_text('\n'.join(f'{unit} {index}'
                                                             for index, unit in enumerate(vocabulary, 1)), encoding='utf-8')
    monkeypatch.setattr(trainer, '_verified_file', lambda *args: 'verified')
    reader = _tiny_reader()
    reader.token_list = ['<blank>', *vocabulary, '<eos>']
    reader.device = torch.device('cpu')
    reader.warmup = lambda: None
    monkeypatch.setattr(trainer, '_reader', lambda *args: reader)
    monkeypatch.setattr(evaluate_chinese, '_visual_input',
                        lambda *args: (np.zeros((12, 96, 96), dtype=np.uint8), .48, None))
    monkeypatch.setattr(trainer, '_predict', lambda reader, dataset, stage: [
        Prediction(sample.id, sample.reference, .48, .01) for sample in dataset.samples])
    output = tmp_path / 'run'

    def fail_first_loss(*args):
        baseline = json.loads((output / 'baseline.json').read_text(encoding='utf-8'))
        assert baseline['status'] == 'baseline_complete'
        assert baseline['training_started'] is False
        assert baseline['protocol']['epochs'] == 2
        assert baseline['base']['checkpoint_sha256'] == trainer.CHECKPOINT_SHA256
        assert set(baseline['baseline']) == {'dev', 'test'}
        assert baseline['partitions']['train']['source_metadata']['revision'] == 'source-pin'
        raise RuntimeError('simulated implementation failure')

    monkeypatch.setattr(trainer, 'ctc_loss', fail_first_loss)
    args = SimpleNamespace(output=str(output), train_manifest=manifest_paths['train'],
                           dev_manifest=manifest_paths['dev'], test_manifest=manifest_paths['test'],
                           source_dir=str(source), checkpoint='/local/base.pth', scope='general',
                           epochs=2, last_layers=1, learning_rate=1e-5, seed=0,
                           beam_size=40, ctc_weight=.5, device='cpu')
    with pytest.raises(RuntimeError, match='simulated implementation failure'):
        trainer.run(args)
    assert (output / 'baseline.json').is_file()
    assert not (output / 'encoder_adapter.pth').exists()
    with pytest.raises(ValueError, match='empty'):
        trainer.run(args)


def test_relative_gain_cannot_hide_coverage_or_exact_sentence_regression():
    baseline = dict(cer=.9, exact_sentence_rate=.2, non_rejected_coverage=1.0, inference_failures=0)
    candidate = dict(cer=.8, exact_sentence_rate=.2, non_rejected_coverage=1.0, inference_failures=0)
    assert improvement_gate(baseline, candidate)['improved']
    assert not improvement_gate(baseline, {**candidate, 'exact_sentence_rate': .1})['improved']
    assert not improvement_gate(baseline, {**candidate, 'non_rejected_coverage': .5})['improved']
    assert not improvement_gate(baseline, {**candidate, 'inference_failures': 1})['improved']
