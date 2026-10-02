"""Paired report statistics cannot hide missing, failed or changed samples."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_chinese_reports import compare, main


def _row(identity, raw, reference, speaker='one', **extras):
    return {'sample_id': identity, 'raw': raw, 'reference': reference, 'speaker': speaker, **extras}


def _report(*rows):
    return {'schema_version': 1, 'samples': list(rows)}


def test_failed_prediction_remains_in_denominator_and_exact_delta_is_independent():
    baseline = _report(_row('a', '甲', '甲乙', 'one'), _row('b', '乙', '丙', 'two'))
    candidate = _report(_row('b', '丙', '丙', 'two'),
                        _row('a', '', '甲乙', 'one', error='decode failed', action='retry'))
    saved = deepcopy((baseline, candidate))
    result = compare(baseline, candidate)
    assert (baseline, candidate) == saved
    assert result['baseline']['reference_characters'] == result['candidate']['reference_characters'] == 3
    assert result['candidate']['samples'] == 2
    assert result['candidate']['character_errors'] == 2
    assert result['candidate']['empty_outputs'] == result['candidate']['inference_failures'] == 1
    assert result['delta']['cer'] == 0
    assert result['delta']['exact_sentence_rate'] == .5
    assert result['product_readiness_established'] is False


def test_missing_sample_or_changed_reference_or_speaker_cannot_be_compared():
    baseline = _report(_row('a', '甲', '甲'), _row('b', '乙', '乙'))
    with pytest.raises(ValueError, match='identical sample IDs'):
        compare(baseline, _report(_row('a', '甲', '甲')))
    with pytest.raises(ValueError, match='reference must agree exactly'):
        compare(_report(_row('a', '甲', '甲。')), _report(_row('a', '甲', '甲')))
    with pytest.raises(ValueError, match='speaker must agree exactly'):
        compare(_report(_row('a', '甲', '甲')), _report(_row('a', '甲', '甲', speaker='other')))


def test_same_text_on_different_video_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match='video_sha256 mismatch'):
        compare(_report(_row('a', '甲', '甲', video_sha256='one')),
                _report(_row('a', '甲', '甲', video_sha256='two')))
    with pytest.raises(ValueError, match='unique'):
        compare(_report(_row('a', '甲', '甲'), _row('a', '甲', '甲')), _report(_row('a', '甲', '甲')))


def test_symmetric_omission_is_caught_by_declared_metrics_or_frozen_manifest():
    report = _report(_row('a', '甲', '甲'))
    report['metrics'] = {'samples': 2}
    with pytest.raises(ValueError, match='full sample rows'):
        compare(report, report)
    report.pop('metrics')
    manifest = {'samples': [{'id': 'a', 'reference': '甲', 'speaker': 'one'},
                            {'id': 'b', 'reference': '乙', 'speaker': 'two'}]}
    with pytest.raises(ValueError, match='every frozen manifest sample'):
        compare(report, report, manifest=manifest)
    manifest['samples'].pop()
    assert compare(report, report, manifest=manifest)['frozen_manifest_coverage_checked']
    manifest['samples'][0]['reference'] = '乙'
    with pytest.raises(ValueError, match='frozen manifest reference'):
        compare(report, report, manifest=manifest)


def test_bootstrap_resamples_speakers_and_weights_pooled_reference_counts():
    # Cluster A improves by -.5 CER; cluster B worsens by +.25. Equal sentence
    # averaging would give -.125, but pooled character counts give exactly 0.
    baseline = _report(_row('a', '甲丙', '甲乙', 'A'), _row('b', '甲乙丙丁', '甲乙丙丁', 'B'))
    candidate = _report(_row('a', '甲乙', '甲乙', 'A'), _row('b', '甲乙丙戊', '甲乙丙丁', 'B'))
    result = compare(baseline, candidate)
    ci = result['cer_difference_95pct_interval']
    assert result['delta']['cer'] == 0
    assert ci['available'] and ci['resamples'] == 5000 and ci['seed'] == 0
    assert ci['lower'] == -.5 and ci['upper'] == .25
    assert ci == compare(baseline, candidate)['cer_difference_95pct_interval']
    reversed_candidate = _report(*reversed(candidate['samples']))
    assert compare(baseline, reversed_candidate) == result


def test_clusters_keep_all_sentences_together_and_one_speaker_has_no_interval():
    baseline = _report(_row('a', '甲', '甲乙', 'A'), _row('b', '', '丙丁', 'A'),
                       _row('c', '甲乙', '甲乙', 'B'))
    candidate = _report(_row('a', '甲乙', '甲乙', 'A'), _row('b', '丙丁', '丙丁', 'A'),
                        _row('c', '甲丙', '甲乙', 'B'))
    result = compare(baseline, candidate)
    ci = result['cer_difference_95pct_interval']
    assert ci['lower'] == -.75 and ci['upper'] == .5
    assert result['per_speaker']['A']['delta']['character_errors'] == -3
    single = compare(_report(_row('a', '', '甲')), _report(_row('a', '甲', '甲')))
    assert not single['cer_difference_95pct_interval']['available']
    assert single['cer_difference_95pct_interval']['lower'] is None


def test_recorded_counts_and_exact_matches_are_recomputed_not_trusted():
    with pytest.raises(ValueError, match='recorded character_errors'):
        compare(_report(_row('a', '乙', '甲', character_errors=0)), _report(_row('a', '甲', '甲')))
    with pytest.raises(ValueError, match='recorded exact'):
        compare(_report(_row('a', '乙', '甲', exact=True)), _report(_row('a', '甲', '甲')))
    result = compare(_report(_row('a', 'Ａ２，', 'a2')), _report(_row('a', '<unk>', 'a2')))
    assert result['baseline']['cer'] == 0
    assert result['candidate']['character_errors'] == 3


def test_cluster_identity_and_bootstrap_configuration_are_required():
    report = _report(_row('a', '甲', '甲', speaker=''))
    with pytest.raises(ValueError, match='speaker required'):
        compare(report, report)
    report = _report(_row('a', '甲', '甲'))
    for count in (0, True, 1_000_001):
        with pytest.raises(ValueError, match='resamples'):
            compare(report, report, resamples=count)


def test_cli_hashes_inputs_and_prevents_overwriting_prediction_evidence(tmp_path, capsys):
    baseline, candidate, output = (tmp_path / name for name in ('before.json', 'after.json', 'comparison.json'))
    baseline.write_text(json.dumps(_report(_row('a', '', '甲'))), encoding='utf-8')
    candidate.write_text(json.dumps(_report(_row('a', '甲', '甲'))), encoding='utf-8')
    assert main([str(baseline), str(candidate), '--output', str(output)]) == 0
    result = json.loads(output.read_text())
    assert result['delta']['cer'] == -1
    assert len(result['provenance']['baseline_report_sha256']) == 64
    capsys.readouterr()
    original = baseline.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main([str(baseline), str(candidate), '--output', str(baseline)])
    assert exc.value.code == 2
    assert baseline.read_bytes() == original
