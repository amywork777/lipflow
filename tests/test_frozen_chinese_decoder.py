from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_cnvsrc import configured_hypotheses
from lipflow.confidence import Hypothesis


def test_zero_reverse_preserves_original_hypotheses():
    expected = [Hypothesis('原始', -2.0, 2)]
    calls = []
    reader = SimpleNamespace(hypotheses=lambda encoded, nbest: calls.append((encoded, nbest)) or expected)
    actual, evidence = configured_hypotheses(reader, 'encoded', beam_size=40, ctc_weight=.5,
                                            nbest=10, reverse_weight=0)
    assert actual is expected
    assert evidence == [asdict(expected[0])]
    assert calls == [('encoded', 10)]


def test_frozen_reverse_uses_candidates_and_keeps_original_scores(monkeypatch):
    import compare_chinese_decoders as helper
    first = helper.Candidate('甲', -2, (1,), -2, -2, beam_score=-2)
    second = helper.Candidate('乙', -3, (2,), -3, -3, beam_score=-3)
    reader = SimpleNamespace(token_list=['blank', '甲', '乙', 'eos'],
                             model=SimpleNamespace(r_decoder='reverse-decoder'))
    calls = []
    monkeypatch.setattr(helper, 'decode_candidates',
                        lambda *args: calls.append(args) or [first, second])
    def reverse(decoder, encoded, ids, eos):
        calls.append((decoder, encoded, ids, eos))
        return -100 if ids == (1,) else 0
    monkeypatch.setattr(helper, 'reverse_log_probability', reverse)
    actual, evidence = configured_hypotheses(reader, 'visual-encoded', beam_size=40,
                                            ctc_weight=.3, nbest=10, reverse_weight=.3)
    assert actual[0].text == '乙'
    assert evidence[0]['beam_score'] == -3
    assert evidence[0]['reverse_score'] == 0
    assert actual[0].score == pytest.approx(-3 + .7 * .3 * 3)
    assert actual[0].token_count == 1
    assert calls[0] == (reader, 'visual-encoded', .3, 40, 10)
    assert calls[1:] == [('reverse-decoder', 'visual-encoded', (1,), 3),
                        ('reverse-decoder', 'visual-encoded', (2,), 3)]


@pytest.mark.parametrize('flags', [['--reverse-weight', 'nan'], ['--reverse-weight', '-.1'],
                                  ['--nbest', '0'], ['--beam-size', '2', '--nbest', '3']])
def test_invalid_frozen_parameters_fail_before_model_load(monkeypatch, flags):
    import evaluate_cnvsrc
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--accept-research-license', '--checkpoint', 'missing',
                                    '--source-dir', 'missing', '--manifest', 'missing', *flags])
    monkeypatch.setattr(evaluate_cnvsrc, '_reader', lambda *args: pytest.fail('model must not load'))
    with pytest.raises(SystemExit) as exc:
        evaluate_cnvsrc.main()
    assert exc.value.code == 2
