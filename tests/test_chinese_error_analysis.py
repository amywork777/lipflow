"""Stored-prediction diagnostics must preserve denominators and held-out labels."""
from copy import deepcopy
import itertools
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_chinese_errors import alignment, analyze, select_rows, vocabulary_characters
from lipflow.evaluation import character_error


def _row(identity, raw, reference, **extras):
    return {'sample_id': identity, 'raw': raw, 'reference': reference, **extras}


def test_edit_decomposition_and_empty_denominator():
    assert alignment('甲乙', '甲丙')['substitutions'] == 1
    assert alignment('甲', '甲乙')['deletions'] == 1
    assert alignment('甲乙', '甲')['insertions'] == 1
    empty = alignment('', '甲乙')
    assert empty['character_errors'] == empty['deletions'] == 2
    with pytest.raises(ValueError, match='reference'):
        alignment('甲', '！！')


def test_alignment_matches_canonical_cer_for_repeated_and_tied_sequences():
    # Exhaustive short sequences detect reversal/tie errors without mocking CER.
    texts = [''.join(units) for length in range(4) for units in itertools.product('甲乙', repeat=length)]
    for reference in texts[1:]:
        for hypothesis in texts:
            result = alignment(hypothesis, reference)
            assert (result['character_errors'], result['reference_characters']) == character_error(hypothesis, reference)
            assert sum(result[key] for key in ('substitutions', 'deletions', 'insertions')) == result['character_errors']
            assert result['matches'] + result['substitutions'] + result['deletions'] == len(reference)
            assert result['matches'] + result['substitutions'] + result['insertions'] == len(hypothesis)


def test_unicode_oov_counts_speakers_and_empty_failures_preserve_all_samples():
    rows = [_row('a', '今天A２', '今天，a2。', speaker='one'),
            _row('b', '', '金额１２２', speaker='two', error='decode failed')]
    before = deepcopy(rows)
    result = analyze(rows, vocabulary=set('今天a2金额'))
    assert rows == before
    assert result['metrics']['samples'] == 2
    assert result['metrics']['reference_characters'] == 9
    assert result['metrics']['character_errors'] == 5
    assert result['metrics']['empty_output'] == result['metrics']['inference_failure'] == 1
    assert result['oov_characters'] == {'1': 1}
    assert result['per_oov_presence']['contains_oov']['samples'] == 1
    assert result['per_speaker']['two']['cer'] == 1
    assert result['references_used_for_decoding'] is False


def test_oov_repetitions_and_missing_oov_are_not_silently_treated_as_supported():
    result = analyze([_row('a', '甲', '甲乙乙', reference_oov_characters=['乙']), _row('b', '甲', '甲')])
    assert result['oov_characters'] == {'乙': 2}
    assert result['oov_diagnostics_available_samples'] == 1
    assert result['per_oov_presence']['unreported']['samples'] == 1
    with pytest.raises(ValueError, match='absent'):
        analyze([_row('a', '甲', '甲', reference_oov_characters=['乙'])])


def test_unknown_emissions_are_flagged_without_silently_cleaning_stored_evidence():
    row = _row('a', '<unk>甲', '甲乙')
    result = analyze([row])
    assert result['metrics']['unknown_token_markers'] == 1
    assert result['metrics']['hypothesis_characters'] == 4
    assert result['metrics']['character_errors'] == character_error(row['raw'], row['reference'])[0]
    assert row['raw'] == '<unk>甲'


def test_recomputed_errors_reject_inconsistent_reports_and_manifest_identity():
    with pytest.raises(ValueError, match='recorded character_errors'):
        analyze([_row('a', '甲', '乙', character_errors=0)])
    row = _row('a', '甲', '甲', speaker='one', split='test')
    source = {'samples': [{'id': 'a', 'reference': '甲', 'speaker': 'one', 'split': 'test'}]}
    assert analyze([row], manifest=source)['per_speaker']['one']['cer'] == 0
    source['samples'][0]['reference'] = '乙'
    with pytest.raises(ValueError, match='reference mismatch'):
        analyze([row], manifest=source)
    with pytest.raises(ValueError, match='unique'):
        analyze([row, row])


def test_nested_reports_require_explicit_stage_and_split():
    report = {'baseline': {'dev': {'samples': [_row('a', '甲', '甲')]}},
              'candidate': {'test': {'samples': [_row('b', '', '乙')]}}}
    with pytest.raises(ValueError, match='explicit'):
        select_rows(report)
    assert select_rows(report, stage='candidate', split='test')[0]['sample_id'] == 'b'
    with pytest.raises(ValueError, match='different'):
        select_rows({'samples': [_row('a', '甲', '甲', split='test')]}, split='dev')


def test_vocabulary_ignores_special_tokens_and_normalizes_cer_units(tmp_path):
    path = tmp_path / 'units.txt'
    path.write_text('<unk> 1\n甲 2\nA 3\n２ 4\n<eos> 5\n', encoding='utf-8')
    assert vocabulary_characters(path) == {'甲', 'a', '2'}
