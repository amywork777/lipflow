from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_chinese_decoders import (Candidate, compare, rerank_candidates,
                                      decode_candidates, pure_ctc_workspace_bytes,
                                      reverse_log_probability, validate_protocol)
from lipflow.evaluation import Dataset, Sample
from lipflow.confidence import Quality


def dataset(split='dev'):
    samples = tuple(Sample(str(i), f'/clips/{i}.mp4', text, f'person-{i}', 'date',
                           'author label', True, 'mouth_roi', True, split=split)
                    for i, text in enumerate(('你好', '明天𠀀')))
    return Dataset(samples, split=split)


@pytest.mark.parametrize('split', ['test', 'heldout', 'train', 'diagnostic'])
def test_rejects_non_development_data(split):
    with pytest.raises(ValueError, match='Only dev'):
        validate_protocol(dataset(split), (.3, .5), 40, 10, .3)


def test_rejects_test_sample_hidden_in_dev_manifest():
    data = dataset()
    data = replace(data, samples=(replace(data.samples[0], split='test'),))
    with pytest.raises(ValueError, match='Only dev'):
        validate_protocol(data, (.3, .5), 40, 10, .3)


@pytest.mark.parametrize('weights,beam,nbest,reverse', [
    ((), 40, 10, .3), ((0,), 40, 10, .3), ((float('nan'),), 40, 10, .3),
    ((1.1,), 40, 10, .3), ((True,), 40, 10, .3), ((.3, .3), 40, 10, .3),
    ((.3,), 0, 1, .3), ((.3,), 2, 3, .3), ((.3,), True, 1, .3),
    ((.3,), 40, 10, float('inf')), ((.3,), 40, 10, -.1),
])
def test_invalid_parameters_fail_before_inference(weights, beam, nbest, reverse):
    with pytest.raises(ValueError):
        validate_protocol(dataset(), weights, beam, nbest, reverse)


class Encoded:
    shape = (2, 1)
    def numel(self):
        return 2

    def element_size(self):
        return 4


def test_encodes_once_keeps_failure_oov_and_hides_reference():
    inputs, encodes, decoded = [], [], []
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '明', '天', '<eos>'],
                             greedy=lambda _: '你好')
    def visual(sample, passed_reader):
        inputs.append(sample)
        assert sample.reference == ''
        return sample.id, 1.0, Quality()
    def encode(frames):
        encodes.append(frames)
        if frames == '1':
            raise RuntimeError('video failed')
        return Encoded()
    reader.encode = encode
    def decode(passed_reader, encoded, weight, beam_size, nbest):
        decoded.append((encoded, weight))
        return [Candidate('你好', -1, (1, 2), -1, -1)]
    report = compare(dataset(), reader, visual, weights=(.3, .5), reverse_weight=0, decoder=decode)
    assert len(inputs) == len(encodes) == 2
    assert len(decoded) == 2 and decoded[0][0] is decoded[1][0]
    for configuration in report['configurations']:
        assert configuration['metrics']['samples'] == 2
        assert configuration['metrics']['reference_characters'] == 5
        assert configuration['metrics']['character_errors'] == 3
        assert configuration['metrics']['inference_failures'] == 1
        assert configuration['samples'][1]['reference_oov_characters'] == ['𠀀']
        assert configuration['readiness']['ready'] is False
        assert configuration['diagnostic']['oracle_used_for_selection'] is False


def test_decoder_failure_affects_only_its_configuration_and_cache_is_bounded():
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '<eos>'], encode=lambda _: Encoded(), greedy=lambda _: '你好')
    def visual(sample, passed_reader):
        return 'pixels', 1.0, Quality()
    def decode(passed_reader, encoded, weight, beam_size, nbest):
        if weight == .3:
            raise RuntimeError('one decoder failed')
        return [Candidate('你好', -1, (1, 2), -1, -1)]
    report = compare(dataset(), reader, visual, weights=(.3, .5), reverse_weight=0,
                     decoder=decode, cache_max_bytes=8)
    assert report['protocol']['cache_bytes'] == 8
    assert report['configurations'][0]['metrics']['inference_failures'] == 2
    assert report['configurations'][1]['metrics']['inference_failures'] == 1
    assert 'quota' in report['configurations'][1]['samples'][1]['error']


def test_reverse_reranks_on_candidate_ids_preserving_score_scale():
    candidates = [Candidate('甲', -2, (1,), -2, -2), Candidate('乙', -3, (2,), -3, -3)]
    seen = []
    def reverse(ids):
        seen.append(ids)
        return -20 if ids == (1,) else -1
    result = rerank_candidates(candidates, .3, .3, reverse)
    assert seen == [(1,), (2,)]
    assert result[0].text == '乙'
    assert result[0].score == pytest.approx(-3 + .7 * .3 * 2)
    assert result[0].beam_score == -3
    assert candidates[0].reverse_score is None
    assert rerank_candidates(candidates, 1, .3, reverse) == candidates
    assert rerank_candidates(candidates, .3, 0, reverse) == candidates


def test_reverse_teacher_forcing_boundaries_eos_causality_and_logits():
    torch = pytest.importorskip('torch')
    class Decoder(torch.nn.Module):
        def forward(self, inputs, mask, memory, memory_mask):
            assert inputs.tolist() == [[4, 2, 1]]
            assert mask.tolist() == [[[True, False, False], [True, True, False], [True, True, True]]]
            assert memory.shape == (1, 2, 3) and memory_mask is None
            logits = torch.zeros(1, 3, 5)
            logits[0, 0, 2] = logits[0, 1, 1] = logits[0, 2, 4] = 2
            return logits, mask
    result = reverse_log_probability(Decoder(), torch.zeros(2, 3), (1, 2), 4)
    expected = 3 * torch.log_softmax(torch.tensor([0., 0., 2., 0., 0.]), -1)[2].item()
    assert result == pytest.approx(expected)
    with pytest.raises(ValueError, match='ordinary'):
        reverse_log_probability(Decoder(), torch.zeros(2, 3), (4,), 4)


def test_reverse_only_uses_dev_best_base_and_reuses_encoded_frames():
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '明', '天', '<eos>'],
                             greedy=lambda _: '你好')
    encoded_objects = []
    def encode(frames):
        encoded = Encoded()
        encoded_objects.append(encoded)
        return encoded
    reader.encode = encode
    def visual(sample, passed_reader):
        return sample.id, 1.0, Quality()
    def decode(passed_reader, encoded, weight, beam_size, nbest):
        # .5 is best by first candidate; the oracle is equally good for both.
        candidates = [Candidate('你好', -1, (1, 2), -1, -1), Candidate('明天', -2, (3, 4), -2, -2)]
        if encoded_objects.index(encoded) == 1:
            candidates.reverse()
        return list(reversed(candidates)) if weight == .3 else candidates
    seen = []
    def reverse(passed_reader, encoded, ids):
        assert encoded in encoded_objects
        seen.append((encoded, ids))
        return -20 if ids == (1, 2) else -1
    report = compare(dataset(), reader, visual, weights=(.3, .5), decoder=decode,
                     reverse_scorer=reverse)
    assert len(encoded_objects) == 2 and len(seen) == 4
    assert report['protocol']['reverse_base_ctc_weight'] == .5
    assert len(report['configurations']) == 3
    assert report['configurations'][-1]['parameters'] == {
        'ctc_weight': .5, 'reverse_weight': .3, 'beam_size': 40, 'nbest': 10,
        'length_bonus': 0., 'external_lm': False}
    assert report['configurations'][-1]['samples'][0]['raw'] == '明天'


def test_fresh_beam_ctc_scorers_and_boundary_preservation(monkeypatch):
    torch = pytest.importorskip('torch')
    import espnet.asr.asr_utils
    import espnet.nets.batch_beam_search
    beams, scorer_calls = [], []
    class Decoder(torch.nn.Module):
        pass
    decoder = Decoder()
    decoder._mem_kv = 'stale utterance'
    def scorers():
        result = {'decoder': decoder, 'ctc': object()}
        scorer_calls.append(result)
        return result
    class Beam:
        def __init__(self, **kwargs):
            beams.append(kwargs)
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, encoded):
            assert not hasattr(decoder, '_mem_kv')
            decoder._mem_kv = 'current utterance'
            return [SimpleNamespace(yseq=torch.tensor([3, 1, 2, 3]), score=-4.,
                                    scores={'decoder': -3., 'ctc': -5.},
                                    asdict=lambda: {'yseq': [3, 1, 2, 3], 'score': -4.})]
    monkeypatch.setattr(espnet.nets.batch_beam_search, 'BatchBeamSearch', Beam)
    monkeypatch.setattr(espnet.asr.asr_utils, 'add_results_to_json', lambda hs, tokens: '你好')
    reader = SimpleNamespace(model=SimpleNamespace(decoder=decoder, scorers=scorers),
                             token_list=['<blank>', '你', '好', '<eos>'],
                             device=torch.device('cpu'), _clean=lambda s: s)
    for weight in (.3, 1.):
        candidates = decode_candidates(reader, torch.zeros(4, 3), weight, 40, 10)
        assert candidates[0].token_ids == (1, 2) and candidates[0].beam_score == -4
        assert not hasattr(decoder, '_mem_kv')
    assert scorer_calls[0]['ctc'] is not scorer_calls[1]['ctc']
    assert beams[0]['weights'] == {'decoder': .7, 'ctc': .3, 'length_bonus': 0.}
    assert beams[1]['pre_beam_score_key'] is None
    assert beams[0]['sos'] == beams[0]['eos'] == 3


def test_pure_ctc_budget_blocks_decoder_without_dropping_samples():
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '<eos>'], encode=lambda _: Encoded(), greedy=lambda _: '你好')
    def visual(sample, passed_reader):
        return 'pixels', 1., Quality()
    invoked = []
    def decode(passed_reader, encoded, weight, beam_size, nbest):
        invoked.append(weight)
        assert weight != 1
        return [Candidate('你好', -1, (1, 2), -1, -1)]
    estimate = pure_ctc_workspace_bytes(Encoded(), 40, 4)
    assert estimate == 6 * 2 * 2 * 40 * 4 * 4
    report = compare(dataset(), reader, visual, weights=(.3, 1.), decoder=decode,
                     reverse_weight=0, pure_ctc_max_workspace_bytes=estimate - 1)
    assert invoked == [.3, .3]
    limited = report['configurations'][1]
    assert limited['resource_limited_samples'] == 2
    assert limited['metrics']['samples'] == 2
    assert limited['metrics']['reference_characters'] == 5
    assert limited['metrics']['character_errors'] == 5
    assert all(row['resource_limited'] and 'not invoked' in row['error'] for row in limited['samples'])
    assert all(row['estimated_pure_ctc_workspace_bytes'] == estimate for row in limited['samples'])


def test_invalid_ctc_resource_budget_fails_before_encoding():
    reader = SimpleNamespace(encode=lambda _: pytest.fail('must not encode'))
    with pytest.raises(ValueError, match='workspace'):
        compare(dataset(), reader, lambda *_: pytest.fail('must not read video'),
                pure_ctc_max_workspace_bytes=0)


@pytest.mark.parametrize('raw,quality', [
    ('你<unk>好', Quality()),
    ('你好', Quality(brightness=20)),
    ('你好你好你好', Quality()),
])
def test_shared_retry_routing_keeps_raw_and_caches_quality_greedy(raw, quality):
    calls = {'visual': 0, 'greedy': 0, 'encode': 0}
    def visual(sample, reader):
        assert sample.reference == ''
        calls['visual'] += 1
        return 'pixels', 1., quality
    def encode(_):
        calls['encode'] += 1
        return Encoded()
    def greedy(_):
        calls['greedy'] += 1
        return raw
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '<unk>', '<eos>'],
                             encode=encode, greedy=greedy)
    def decode(*_):
        return [Candidate(raw, -1, (1, 2), -1, -1)]
    report = compare(dataset(), reader, visual, weights=(.3, .5), decoder=decode,
                     reverse_scorer=lambda *_: -1)
    assert calls == {'visual': 2, 'greedy': 2, 'encode': 2}
    for config in report['configurations']:
        assert config['metrics']['samples'] == 2
        assert config['metrics']['non_rejected_coverage'] == 0
        assert all(row['action'] == 'retry' and row['raw'] == raw for row in config['samples'])
        assert all(row['quality'] == quality.__dict__ and row['greedy'] == raw for row in config['samples'])


def test_any_forced_candidate_skips_whole_utterance_preserving_order_and_scores():
    candidates = [Candidate('你好', -1, (1, 2), -1, -1),
                  Candidate('明天', -2, (3, 4), -2, -2, forced_eos=True)]
    result = rerank_candidates(candidates, .5, .3,
                               lambda *_: pytest.fail('forced EOS must not be rescored'))
    assert [(c.text, c.score) for c in result] == [('你好', -1), ('明天', -2)]
    assert all(c.reverse_score is None and c.reverse_rerank_skipped_reason == 'forced_eos_in_beam'
               for c in result)
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '明', '天', '<eos>'],
                             encode=lambda _: Encoded(), greedy=lambda _: '你好')
    report = compare(dataset(), reader, lambda *_: ('pixels', 1., Quality()), weights=(.5,),
                     decoder=lambda *_: candidates,
                     reverse_scorer=lambda *_: pytest.fail('entire utterance must skip reverse'))
    rows = report['configurations'][-1]['samples']
    assert all(row['raw'] == '你好' and row['reverse_reranking_skipped'] for row in rows)
    assert all([h['score'] for h in row['hypotheses']] == [-1, -2] for row in rows)


@pytest.mark.parametrize('forced_ids,trailing', [([3, 1, 2, 1, 3], 1), ([3, 1, 2, 3, 3], 2)])
def test_final_loop_eos_stripped_and_evidence_retained_even_beyond_nbest(monkeypatch, forced_ids, trailing):
    torch = pytest.importorskip('torch')
    import espnet.asr.asr_utils
    import espnet.nets.batch_beam_search
    decoder = torch.nn.Identity()
    class Beam:
        def __init__(self, **kwargs):
            pass
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, encoded):
            def hyp(ids, score):
                return SimpleNamespace(yseq=torch.tensor(ids), score=score,
                                       scores={'decoder': score - 1, 'ctc': score - 2},
                                       asdict=lambda: {'yseq': ids, 'score': score})
            return [hyp([3, 1, 3], -1), hyp(forced_ids, -2)]
    monkeypatch.setattr(espnet.nets.batch_beam_search, 'BatchBeamSearch', Beam)
    def render(hs, tokens):
        assert hs[0]['yseq'][-1] == 3
        assert 3 not in hs[0]['yseq'][1:-1]
        return ''.join(tokens[i] for i in hs[0]['yseq'][1:-1])
    monkeypatch.setattr(espnet.asr.asr_utils, 'add_results_to_json', render)
    reader = SimpleNamespace(model=SimpleNamespace(decoder=decoder,
                                                   scorers=lambda: {'decoder': decoder, 'ctc': object()}),
                             token_list=['<blank>', '你', '好', '<eos>'],
                             device=torch.device('cpu'), _clean=lambda s: s)
    result = decode_candidates(reader, torch.zeros(3, 2), .5, 40, 2)
    assert result[1].forced_eos and result[1].trailing_eos_count == trailing
    assert result[1].beam_token_ids == tuple(forced_ids)
    assert result[1].token_ids == tuple(forced_ids[1:len(forced_ids) - trailing])
    assert result[1].score == result[1].beam_score == -2
    only_top = decode_candidates(reader, torch.zeros(3, 2), .5, 40, 1)
    assert len(only_top) == 1 and not only_top[0].forced_eos
    assert only_top[0].reverse_rerank_skipped_reason == 'forced_eos_in_beam'
    assert rerank_candidates(only_top, .5, .3, lambda *_: pytest.fail('must skip')) == only_top


def test_reverse_failure_retains_base_raw_and_denominator():
    reader = SimpleNamespace(token_list=['<blank>', '你', '好', '<eos>'],
                             encode=lambda _: Encoded(), greedy=lambda _: '你好')
    def fail(*_):
        raise RuntimeError('reverse failed')
    report = compare(dataset(), reader, lambda *_: ('pixels', 1., Quality()), weights=(.5,),
                     decoder=lambda *_: [Candidate('你好', -1, (1, 2), -1, -1)], reverse_scorer=fail)
    reranked = report['configurations'][-1]
    assert reranked['metrics']['reference_characters'] == 5
    assert reranked['metrics']['inference_failures'] == 2
    assert all(row['raw'] == '你好' and row['action'] == 'retry' for row in reranked['samples'])
    assert all(row['hypotheses'][0]['score'] == -1 for row in reranked['samples'])
