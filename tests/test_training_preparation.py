"""Manifest preparation tests use only local text/bytes and no model inference."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_chinese_training as preparation
from lipflow.evaluation import load_manifest


def fixture(tmp_path, monkeypatch, *, split='train', row_split=None):
    vocab = tmp_path / 'char_units.txt'
    vocab.write_text('<unk> 1\n请 2\n发 3\n文 4\n件 5\n', encoding='utf-8')
    monkeypatch.setattr(preparation, 'VOCABULARY_SHA256', hashlib.sha256(vocab.read_bytes()).hexdigest())
    rows = []
    for index, reference in enumerate([' 请，发文件。 ', '请发A文件']):
        video = tmp_path / f'video{index}.mp4'
        video.write_bytes(f'local test video bytes {index}'.encode())
        rows.append(dict(id=str(index), video=video.name, reference=reference, speaker='s1', session='session1',
                         label_source='Independent local annotation', label_verified=True, domain='full_face',
                         split=row_split or split, custom={'must_survive': index}))
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(dict(schema_version=1, split=split, training_overlap_checked=False,
                                     source_note='preserved', samples=rows,
                                     development_samples=[dict(id='earlier', speaker='s2', session='previous',
                                                               video=rows[0]['video'], custom=True)])), encoding='utf-8')
    return source, vocab


def test_explicit_train_selection_preserves_text_metadata_and_relocates_all_video_paths(tmp_path, monkeypatch):
    source, vocab = fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    output = tmp_path / 'different' / 'prepared.json'
    result = preparation.prepare(source, vocab, output, exclude_unsupported_training_samples=True)
    assert source.read_bytes() == original
    assert len(result['samples']) == 1
    row = result['samples'][0]
    assert row['reference'] == ' 请，发文件。 ' and row['custom'] == {'must_survive': 0}
    assert (output.parent / row['video']).resolve() == tmp_path / 'video0.mp4'
    dev = result['development_samples'][0]
    assert dev['custom'] and (output.parent / dev['video']).resolve() == tmp_path / 'video0.mp4'
    assert result['source_note'] == 'preserved'
    evidence = result['training_eligibility']
    assert (evidence['input_samples'], evidence['kept_samples'], evidence['excluded_samples']) == (2, 1, 1)
    assert evidence['exclusions'] == [dict(sample_id='1', unsupported_units=['A'], empty_target=False)]
    assert evidence['original_manifest_sha256'] == hashlib.sha256(original).hexdigest()
    assert not evidence['selection_used_decoder_results'] and not evidence['dev_test_filtered']
    assert evidence['references_unmodified']
    assert load_manifest(output).samples[0].reference == row['reference']


def test_default_fails_and_lists_oov_without_creating_output(tmp_path, monkeypatch):
    source, vocab = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'new.json'
    with pytest.raises(ValueError, match='Unsupported training targets') as error:
        preparation.prepare(source, vocab, output)
    assert 'unsupported_units' in str(error.value) and 'A' in str(error.value)
    assert not output.exists()


def test_output_parent_symlink_alias_keeps_loadable_video_paths(tmp_path, monkeypatch):
    physical = tmp_path / 'physical'
    physical.mkdir()
    source, vocab = fixture(physical, monkeypatch)
    alias = tmp_path / 'alias'
    alias.symlink_to(physical, target_is_directory=True)
    output = alias / 'prepared.json'
    result = preparation.prepare(source, vocab, output, exclude_unsupported_training_samples=True)
    assert result['samples'][0]['video'] == 'video0.mp4'
    actual = load_manifest(output)
    assert Path(actual.samples[0].video) == physical / 'video0.mp4'
    assert (output.resolve().parent / result['development_samples'][0]['video']).resolve() == physical / 'video0.mp4'


@pytest.mark.parametrize('split,row_split', [('dev', None), ('test', None), ('train', 'test')])
def test_dev_test_cannot_be_filtered_even_with_permission(tmp_path, monkeypatch, split, row_split):
    source, vocab = fixture(tmp_path, monkeypatch, split=split, row_split=row_split)
    with pytest.raises(ValueError, match='train-only'):
        preparation.prepare(source, vocab, tmp_path / 'new.json', exclude_unsupported_training_samples=True)
    assert not (tmp_path / 'new.json').exists()


def test_existing_output_and_wrong_vocabulary_are_refused(tmp_path, monkeypatch):
    source, vocab = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'new.json'
    output.write_bytes(b'existing operator file')
    with pytest.raises(FileExistsError):
        preparation.prepare(source, vocab, output, exclude_unsupported_training_samples=True)
    assert output.read_bytes() == b'existing operator file'
    monkeypatch.setattr(preparation, 'VOCABULARY_SHA256', 'wrong')
    with pytest.raises(ValueError, match='SHA256'):
        preparation.prepare(source, vocab, tmp_path / 'other.json', exclude_unsupported_training_samples=True)
