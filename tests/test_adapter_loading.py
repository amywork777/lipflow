"""CPU-only strict research artifact loading; no inference or datasets."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_cnvsrc import (CHECKPOINT_SHA256, CONFIG_SHA256, SOURCE_REVISION,
                            VOCABULARY_SHA256, _load_adapter)


def reader():
    model = torch.nn.Module()
    model.encoder = torch.nn.Module()
    model.encoder.frontend = torch.nn.Linear(4, 4)
    model.encoder.encoders = torch.nn.ModuleList([
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)),
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)),
        torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)),
    ])
    model.encoder.after_norm = torch.nn.LayerNorm(4)
    model.ctc = torch.nn.Linear(4, 5)
    model.decoder = torch.nn.Linear(4, 5)
    model.r_decoder = torch.nn.Linear(4, 5)
    return SimpleNamespace(language='zh', model=model)


def payload(subject):
    selected = {name: parameter.detach().clone() + 0.1 for name, parameter in subject.model.named_parameters()
                if name.startswith(('encoder.encoders.2.', 'encoder.after_norm.'))}
    return {'schema_version': 1, 'language': 'zh', 'last_encoder_layers': 1,
            'base_checkpoint_sha256': CHECKPOINT_SHA256, 'adapter_parameters': selected,
            'metadata': {'base': {'checkpoint_sha256': CHECKPOINT_SHA256,
                                 'configuration_sha256': CONFIG_SHA256,
                                 'vocabulary_sha256': VOCABULARY_SHA256,
                                 'source_revision': SOURCE_REVISION},
                         'protocol': {'last_encoder_layers': 1},
                         'adapter_tensor_count': len(selected),
                         'changed_tensor_count': len(selected)}}


def saved(tmp_path, artifact):
    path = tmp_path / 'encoder_adapter.pth'
    torch.save(artifact, path)
    return path


def test_valid_complete_state_changes_only_declared_parameters_and_reports_sha(tmp_path, monkeypatch):
    subject = reader()
    artifact = payload(subject)
    before = {key: value.clone() for key, value in subject.model.state_dict().items()}
    path = saved(tmp_path, artifact)
    original, calls = torch.load, []
    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', checked_load)
    provenance = _load_adapter(subject, path)
    assert calls[0]['weights_only'] is True
    assert provenance['file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance['base_checkpoint_sha256'] == CHECKPOINT_SHA256
    assert provenance['automatic_activation'] is False
    assert provenance['loaded_parameter_count'] == len(artifact['adapter_parameters'])
    assert not subject.model.training
    for key, value in subject.model.state_dict().items():
        expected = artifact['adapter_parameters'].get(key, before[key])
        assert torch.equal(value, expected)


@pytest.mark.parametrize('field,value', [('schema_version', True), ('schema_version', 2),
                                         ('language', 'en'), ('base_checkpoint_sha256', 'wrong-base'),
                                         ('last_encoder_layers', True), ('last_encoder_layers', 0),
                                         ('last_encoder_layers', 4)])
def test_wrong_schema_language_base_or_layer_count_rejected(tmp_path, field, value):
    subject = reader()
    artifact = payload(subject)
    artifact[field] = value
    with pytest.raises(ValueError):
        _load_adapter(subject, saved(tmp_path, artifact))


@pytest.mark.parametrize('foreign_key', ['decoder.weight', 'r_decoder.weight', 'ctc.weight',
                                        'encoder.frontend.weight', 'encoder.encoders.0.0.weight',
                                        'encoder.encoders.2.1.running_mean'])
def test_decoder_ctc_frontend_earlier_layers_and_frozen_buffers_cannot_be_overridden(tmp_path, foreign_key):
    subject = reader()
    artifact = payload(subject)
    artifact['adapter_parameters'][foreign_key] = subject.model.state_dict()[foreign_key].clone()
    artifact['metadata']['adapter_tensor_count'] += 1
    with pytest.raises(ValueError, match='complete selected encoder'):
        _load_adapter(subject, saved(tmp_path, artifact))


def test_missing_selected_parameter_rejected(tmp_path):
    subject = reader()
    artifact = payload(subject)
    del artifact['adapter_parameters']['encoder.after_norm.bias']
    with pytest.raises(ValueError, match='missing='):
        _load_adapter(subject, saved(tmp_path, artifact))


@pytest.mark.parametrize('corruption', ['shape', 'dtype', 'nan', 'inf', 'non-tensor', 'sparse'])
def test_bad_tensor_rejected_before_any_parameter_copy(tmp_path, corruption):
    subject = reader()
    artifact = payload(subject)
    # Corrupt the last declared tensor to catch implementations that mutate
    # earlier good parameters before validating the entire file.
    key = 'encoder.after_norm.bias'
    bad = artifact['adapter_parameters'][key]
    if corruption == 'shape':
        bad = bad[:1]
    elif corruption == 'dtype':
        bad = bad.double()
    elif corruption == 'nan':
        bad[0] = float('nan')
    elif corruption == 'inf':
        bad[0] = float('inf')
    elif corruption == 'non-tensor':
        bad = 'not a tensor'
    else:
        bad = bad.to_sparse()
    artifact['adapter_parameters'][key] = bad
    before = {name: parameter.detach().clone() for name, parameter in subject.model.named_parameters()}
    with pytest.raises(ValueError):
        _load_adapter(subject, saved(tmp_path, artifact))
    assert all(torch.equal(parameter, before[name]) for name, parameter in subject.model.named_parameters())


@pytest.mark.parametrize('change', ['base', 'protocol', 'boolean-protocol', 'count', 'changed'])
def test_inconsistent_provenance_and_counts_rejected(tmp_path, change):
    subject = reader()
    artifact = payload(subject)
    if change == 'base':
        artifact['metadata']['base']['vocabulary_sha256'] = 'different-vocab'
    elif change == 'protocol':
        artifact['metadata']['protocol']['last_encoder_layers'] = 2
    elif change == 'boolean-protocol':
        artifact['metadata']['protocol']['last_encoder_layers'] = True
    elif change == 'count':
        artifact['metadata']['adapter_tensor_count'] -= 1
    else:
        artifact['metadata']['changed_tensor_count'] = 1000
    with pytest.raises(ValueError):
        _load_adapter(subject, saved(tmp_path, artifact))


def test_non_artifact_file_rejected(tmp_path):
    path = tmp_path / 'invalid.pth'
    path.write_bytes(b'not a torch file')
    with pytest.raises(ValueError, match='weights_only=True'):
        _load_adapter(reader(), path)
