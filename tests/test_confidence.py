import math
from lipflow.confidence import Hypothesis as H, Quality as Q, assess


def test_review_is_default_even_for_large_gap():
    assert assess([H('hello', -1, 1), H('yellow', -4, 1)], 'hello', Q()).action == 'review'


def test_opt_in_auto_requires_margin_and_ctc_agreement():
    hs = [H('hello there', -2, 2), H('yellow pear', -6, 2)]
    assert assess(hs, 'hello there', Q(), 'auto').action == 'auto'
    assert assess(hs, 'yellow pear', Q(), 'auto').action == 'review'
    assert assess([H('hello there', -2, 2), H('yellow pear', -2.1, 2)], 'hello there', Q(), 'auto').action == 'review'


def test_unvalidated_mandarin_never_auto_pastes_even_with_decoder_agreement():
    hs = [H('明天发送', -1, 4), H('今天发送', -100, 4)]
    d = assess(hs, hs[0].text, Q(), 'auto', language='zh')
    assert d.action == 'review' and d.agreement and d.margin > 20
    assert 'Experimental Mandarin' in d.reason
    # Han output also stays protected if a caller omits the language flag.
    assert assess(hs, hs[0].text, Q(), 'auto').action == 'review'
    assert assess([H('hello', -1, 1), H('yellow', -9, 1)], 'hello', Q(),
                  'auto', language='zh').action == 'review'


def test_missing_nonfinite_or_single_hypothesis_never_auto():
    for hs in ([H('hello', -1, 1)], [H('hello', math.nan, 1), H('yellow', -5, 1)],
               [H('hello', -1, 0), H('yellow', -5, 1)]):
        assert assess(hs, 'hello', Q(), 'auto').action == 'review'
    assert assess([], '', Q()).action == 'retry'


def test_quality_advice_takes_priority_over_decoder_scores():
    hs = [H('hello', -1, 1), H('yellow', -9, 1)]
    for q, expected in [(Q(face_ratio=.5), 'facing'), (Q(mouth_pixels=20), 'closer'),
                        (Q(brightness=20), 'lighting'), (Q(contrast=1), 'lighting')]:
        result = assess(hs, 'hello', q, 'auto')
        assert result.action == 'retry' and expected in result.reason


def test_nonfinite_quality_and_unknown_output_require_retry():
    hs = [H('hello', -1, 1), H('yellow', -9, 1)]
    assert assess(hs, 'hello', Q(brightness=math.nan), 'auto').action == 'retry'
    assert assess([H('<unk>', -1, 1)], '<unk>', Q()).action == 'retry'


def test_actual_chinese_decoder_phrase_loop_is_rejected_without_rejecting_normal_repetition():
    loop = '我是中国人我是中国人我是中国人我是中国人'
    assert assess([H(loop, -1, len(loop))], loop, Q(), language='zh').action == 'retry'
    for text in ('好好学习天天向上我们明天见', '我说不要发送，他也说不要发送'):
        assert assess([H(text, -1, len(text))], text, Q(), language='zh').action == 'review'
