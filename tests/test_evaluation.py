"""Accuracy/gate tests use independently specified predictions, not model output."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from lipflow.evaluation import (Dataset, Exclusion, Prediction, Sample, Thresholds,
                                character_error, characters, evaluate, independence_issues,
                                load_manifest, percentile)


def sample(index=0, **changes):
    original = Sample(str(index), f'/local/video{index}.mp4', '明天不要发货', f'speaker{index % 5}',
                      f'session{index // 5 % 2}', 'Human annotation record', True, 'webcam',
                      sha256=f'content{index}', articulation='silent', articulation_verified=True)
    return replace(original, **changes)


def test_cer_keeps_han_latin_digits_and_normalizes_only_formatting():
    assert characters('１２元，ＡＢＣ！') == list('12元abc')
    assert character_error('明天不要发货。', '明天，不要发货！') == (0, 6)
    # A complete English word is not collapsed into one CER unit.
    assert character_error('AC19', 'AB12') == (2, 4)
    assert character_error('明天发货', '明天不要发货') == (2, 6)
    assert character_error('後天', '后天') == (1, 2)


@pytest.mark.parametrize('reference', ['', '，。 ！', None])
def test_empty_or_missing_reference_is_invalid(reference):
    with pytest.raises(ValueError):
        character_error('hello', reference)


def test_blank_predictions_count_deletions_and_missing_predictions_never_disappear():
    dataset = Dataset((sample(0), sample(1)), training_overlap_checked=True)
    report = evaluate(dataset, [Prediction('0', '', 4, 1, 'retry')])
    metrics = report['metrics']
    assert metrics['samples'] == 2
    assert metrics['reference_characters'] == 12
    assert metrics['character_errors'] == 12 and metrics['cer'] == 1
    assert metrics['non_rejected_coverage'] == 0 and metrics['rejection_rate'] == 1
    assert metrics['automatic_coverage'] == 0 and metrics['automatic_cer'] is None
    assert metrics['inference_failures'] == 1
    assert not report['readiness']['ready']


def test_empty_dataset_does_not_claim_zero_error_or_readiness():
    report = evaluate(Dataset((), training_overlap_checked=True), [])
    assert report['metrics']['cer'] is None
    assert report['metrics']['exact_sentence_rate'] is None
    assert report['metrics']['rtf_p95'] is None
    assert not report['readiness']['ready']


@pytest.mark.parametrize('articulation,verified', [('voiced', True), ('whispered', True),
                                                  ('unknown', False), ('silent', False)])
def test_visual_only_predictions_do_not_prove_deliberately_silent_articulation(articulation, verified):
    data = Dataset(tuple(sample(i, articulation=articulation, articulation_verified=verified)
                         for i in range(50)), training_overlap_checked=True)
    predictions = [Prediction(s.id, s.reference, 4, 1) for s in data.samples]
    report = evaluate(data, predictions)
    assert report['metrics']['cer'] == 0
    assert not report['readiness']['ready']
    gate = next(g for g in report['readiness']['gates'] if g['criterion'] == 'silent_articulation')
    assert not gate['passed'] and gate['observed'] == 0


def test_coverage_is_not_accuracy_and_automatic_subset_is_separate():
    dataset = Dataset((sample(0), sample(1), sample(2)), training_overlap_checked=True)
    predictions = [Prediction('0', '明天发货', 4, .5, 'review'),
                   Prediction('1', '明天不要发货', 4, 1, 'auto'),
                   Prediction('2', '', 4, 1.5, 'retry')]
    metrics = evaluate(dataset, predictions)['metrics']
    assert metrics['cer'] == pytest.approx(8 / 18)
    assert metrics['non_rejected_coverage'] == pytest.approx(2 / 3)
    assert metrics['exact_sentence_rate'] == pytest.approx(1 / 3)
    assert metrics['automatic_coverage'] == pytest.approx(1 / 3)
    assert metrics['automatic_cer'] == 0
    assert metrics['warm_processing_seconds_p50'] == 1
    assert metrics['warm_processing_seconds_p95'] == pytest.approx(1.45)
    assert metrics['rtf_p95'] == pytest.approx(1.45 / 4)


def test_readiness_requires_sufficient_independent_speakers_sessions_and_camera():
    samples = tuple(sample(i) for i in range(50))
    dataset = Dataset(samples, training_overlap_checked=True)
    predictions = [Prediction(s.id, s.reference, 4, 1) for s in samples]
    report = evaluate(dataset, predictions)
    assert report['readiness']['ready']
    assert report['readiness']['status'] == 'ready_under_declared_protocol'
    assert report['metrics']['sessions_per_speaker'] == {f'speaker{i}': 2 for i in range(5)}
    # Two exact author demo clips are only a smoke test, even at 0% CER.
    demos = tuple(replace(s, domain='mouth_roi', mouth_roi=True) for s in samples[:2])
    smoke = evaluate(Dataset(demos, training_overlap_checked=True), predictions[:2])
    failed = {gate['criterion'] for gate in smoke['readiness']['gates'] if not gate['passed']}
    assert {'minimum_samples', 'minimum_speakers', 'minimum_sessions_per_speaker', 'webcam_domain'} <= failed


def test_thresholds_can_be_explicitly_changed_without_implying_camera_readiness():
    s = sample(domain='mouth_roi', mouth_roi=True)
    thresholds = Thresholds(min_samples=1, min_speakers=1, min_sessions_per_speaker=1,
                            require_webcam_domain=False)
    report = evaluate(Dataset((s,), training_overlap_checked=True),
                      [Prediction(s.id, s.reference, 4, 1)], thresholds)
    assert report['readiness']['ready']
    assert report['readiness']['thresholds']['require_webcam_domain'] is False
    assert not evaluate(Dataset((s,), training_overlap_checked=True),
                        [Prediction(s.id, s.reference, 4, 1)], thresholds, mode='whisper')['readiness']['ready']


def test_default_accuracy_and_wait_thresholds_are_enforced():
    samples = tuple(sample(i) for i in range(50))
    dataset = Dataset(samples, training_overlap_checked=True)
    predictions = [Prediction(s.id, '明天发货', 4, 3) for s in samples]
    report = evaluate(dataset, predictions)
    failed = {gate['criterion'] for gate in report['readiness']['gates'] if not gate['passed']}
    assert {'cer', 'exact_sentence_rate', 'warm_processing_seconds_p95'} <= failed
    assert 'rtf_p95' not in failed


def test_known_training_speaker_or_duplicate_content_blocks_independence():
    samples = (sample(0), sample(1))
    dev = Exclusion('training', samples[0].speaker, 'different_session')
    dataset = Dataset(samples, training_overlap_checked=True, development_samples=(dev,))
    assert any('overlaps' in issue for issue in independence_issues(dataset))
    # Content hash catches renamed copies.
    copies = (samples[0], replace(samples[1], sha256=samples[0].sha256))
    assert any('Overlapping/duplicate' in issue for issue in independence_issues(Dataset(copies)))
    # Adjacent independent segments of one source video are not duplicated.
    adjacent = (replace(copies[0], start=0, end=2), replace(copies[1], start=2, end=4))
    assert not any('Overlapping/duplicate' in issue for issue in independence_issues(Dataset(adjacent)))


def test_provenance_and_unfinished_overlap_audit_block_readiness():
    dataset = Dataset((sample(label_verified=False),), split='calibration')
    issues = independence_issues(dataset)
    assert len(issues) == 3
    assert any('verified label' in issue for issue in issues)
    assert any('audit' in issue for issue in issues)
    assert any('split' in issue for issue in issues)


@pytest.mark.parametrize('time_value,duration', [(math.nan, 4), (-1, 4), (1, 0), (1, None)])
def test_invalid_latency_cannot_pass(time_value, duration):
    s = sample()
    report = evaluate(Dataset((s,), training_overlap_checked=True),
                      [Prediction(s.id, s.reference, duration, time_value)])
    assert report['metrics']['timed_samples'] == 0
    assert not report['readiness']['ready']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('changes', [{'min_samples': 0}, {'min_speakers': True},
                                    {'max_cer': math.nan}, {'min_exact_sentence_rate': 1.1},
                                    {'max_p95_rtf': 0}, {'max_p95_processing_seconds': -1}])
def test_invalid_thresholds_are_rejected(changes):
    with pytest.raises(ValueError):
        Thresholds(**changes)


def test_percentiles_are_deterministic_and_undefined_for_no_measurements():
    assert percentile([], .95) is None
    assert percentile([3], .95) == 3
    assert percentile([4, 1, 2, 3], .95) == pytest.approx(3.85)


def manifest_item():
    return {'id': 'sample1', 'video': 'local.mp4', 'reference': '１２元谢谢',
            'speaker': 'speaker1', 'session': 'session1', 'label_source': 'Local human transcript',
            'label_verified': True, 'domain': 'webcam'}


def write_manifest(tmp_path, item):
    (tmp_path / 'local.mp4').write_bytes(b'video data with no model content')
    manifest = tmp_path / 'evaluation.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'split': 'test',
                                   'training_overlap_checked': True, 'samples': [item]}), encoding='utf-8')
    return manifest


def test_manifest_preserves_independent_labels_and_resolves_relative_files(tmp_path):
    dataset = load_manifest(write_manifest(tmp_path, manifest_item()))
    assert dataset.samples[0].video == str(tmp_path / 'local.mp4')
    assert dataset.samples[0].reference == '１２元谢谢'
    assert len(dataset.samples[0].sha256) == 64
    assert dataset.samples[0].label_verified


def test_training_record_in_test_manifest_cannot_evade_split_gate(tmp_path):
    item = {**manifest_item(), 'split': 'train'}
    dataset = load_manifest(write_manifest(tmp_path, item))
    assert dataset.samples[0].split == 'train'
    assert any('split' in issue for issue in independence_issues(dataset))


@pytest.mark.parametrize('changes', [{'reference': ''}, {'reference': '，。'}, {'video': 'missing.mp4'},
                                    {'label_source': ''}, {'speaker': ''}, {'session': ''},
                                    {'label_verified': 'true'}, {'start': -1}, {'end': 0},
                                    {'domain': 'mouth_roi', 'mouth_roi': False}])
def test_invalid_manifest_inputs_are_not_silently_excluded(tmp_path, changes):
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, {**manifest_item(), **changes}))


def test_duplicate_and_unknown_predictions_are_reported():
    s = sample()
    report = evaluate(Dataset((s,), training_overlap_checked=True),
                      [Prediction(s.id, s.reference, 4, 1), Prediction(s.id, s.reference, 4, 1),
                       Prediction('other', s.reference, 4, 1)])
    independence = next(g for g in report['readiness']['gates'] if g['criterion'] == 'independent_data')
    assert not independence['passed']
    assert any('Duplicate prediction' in issue for issue in independence['observed'])
    assert any('not present' in issue for issue in independence['observed'])


def test_batch_model_is_warmed_once_never_receives_reference_and_keeps_failed_clips(tmp_path, monkeypatch):
    from lipflow.confidence import Hypothesis

    spec = importlib.util.spec_from_file_location('evaluate_chinese_script',
               Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_chinese.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    events = []

    class FakeReader:
        token_list = ['<blank>', '猜', '<eos>']
        enc_device = device = 'cpu'
        beam = SimpleNamespace(weights={'decoder': .9, 'ctc': .1})

        def __init__(self, **options):
            assert options['language'] == 'zh' and not options['personal']
            assert not any('reference' in key or 'prompt' in key for key in options)
            events.append('load')

        def warmup(self):
            events.append('warmup')

        def encode(self, pixels):
            assert pixels == 'unlabelled pixels'
            events.append('encode')
            return 'encoded pixels'

        def hypotheses(self, encoded, nbest):
            assert encoded == 'encoded pixels' and nbest == 5
            return [Hypothesis('模型猜测', -1, 4)]

        def greedy(self, encoded):
            assert encoded == 'encoded pixels'
            return '模型猜测'

    monkeypatch.setitem(sys.modules, 'lipflow.vsr', SimpleNamespace(LipReader=FakeReader))
    item = manifest_item()
    manifest = write_manifest(tmp_path, item)
    data = json.loads(manifest.read_text(encoding='utf-8'))
    # Same-source adjacent ranges are permitted, while different references are
    # scored only after decoding and can never alter the fake reader's output.
    data['samples'] = [{**item, 'end': 4}, {**item, 'id': 'failed', 'start': 4, 'end': 8}]
    manifest.write_text(json.dumps(data), encoding='utf-8')

    def visual_input(sample, reader):
        from lipflow.confidence import Quality
        assert events[:2] == ['load', 'warmup']
        if sample.id == 'failed':
            raise ValueError('video cannot be decoded')
        return 'unlabelled pixels', 4, Quality()

    monkeypatch.setattr(script, '_visual_input', visual_input)
    limits = Thresholds()
    options = vars(limits).copy()
    options.pop('require_webcam_domain')
    options.pop('require_silent_articulation')
    args = SimpleNamespace(manifest=manifest, beam_size=10, device='cpu', model_dir=str(tmp_path / 'fake-model'),
                           allow_non_webcam_domain=False, allow_voiced_articulation=False, **options)
    report = script._batch(args)
    assert events == ['load', 'warmup', 'encode']
    assert report['samples'][0]['raw'] == '模型猜测'
    assert report['samples'][1]['error'] == 'ValueError: video cannot be decoded'
    assert report['metrics']['inference_failures'] == 1 and report['metrics']['samples'] == 2
    assert report['metrics']['automatic_coverage'] == 0
    assert report['model']['vocabulary_size'] == 3
    assert report['model']['decoder_weights'] == {'decoder': .9, 'ctc': .1}
    assert not report['readiness']['ready']


def test_full_face_batch_obeys_live_no_movement_rejection_without_decoder(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    np = pytest.importorskip('numpy')
    spec = importlib.util.spec_from_file_location('evaluate_chinese_motion_script',
               Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_chinese.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    cleanup = []

    class StaticTracker:
        def detect(self, frame, timestamp):
            return SimpleNamespace(anchors=np.ones((4, 2)), outer_lips=np.array([[0, 0], [40, 10]]),
                                   mouth_open=.1)

        def close(self):
            cleanup.append('tracker closed')

    class Capture:
        remaining = 25

        def isOpened(self):
            return True

        def get(self, prop):
            assert prop == cv2.CAP_PROP_FPS
            return 25

        def read(self):
            self.remaining -= 1
            return (True, np.zeros((32, 32, 3), dtype=np.uint8)) if self.remaining >= 0 else (False, None)

        def release(self):
            cleanup.append('capture released')

    monkeypatch.setattr(cv2, 'VideoCapture', lambda video: Capture())
    monkeypatch.setitem(sys.modules, 'lipflow.face', SimpleNamespace(FaceTracker=StaticTracker, mouth_rois=None))
    with pytest.raises(script.ClipRejected, match='No lip movement') as caught:
        script._visual_input(sample(), SimpleNamespace())
    assert caught.value.duration == 1
    assert cleanup == ['capture released', 'tracker closed']
