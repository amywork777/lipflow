import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from lipflow.text import tokens
from lipflow.train_vsr import _targets
from lipflow.practice import practice_sentences
from lipflow.chinese import install


def test_unicode_tokens_and_chinese_training_targets():
    assert tokens('不要发送 API v2。') == ['不', '要', '发', '送', 'api', 'v2']
    r = SimpleNamespace(language='zh', token_list=['<blank>', '<unk>', '不', '要', '发', '送', '<eos>'])
    assert _targets(r, '不要，发送。') == [2, 3, 4, 5]
    with pytest.raises(ValueError, match='Han vocabulary'):
        _targets(r, '不要发送 API')


def test_chinese_practice_and_data_are_separate(tmp_path, monkeypatch):
    from lipflow import practice, paths
    from lipflow.paths import personal_vsr
    monkeypatch.setattr(practice, 'CLIPS', str(tmp_path))
    practice.save_clip(np.zeros((12,96,96),np.uint8), '你好', language='zh')
    assert practice.saved_clips('en') == []
    assert practice.saved_clips('zh')[0]['text'] == '你好'
    assert len(practice_sentences(24, 'zh')) == 24
    assert personal_vsr('en') != personal_vsr('zh')


def test_chinese_correction_does_not_include_new_typing():
    from lipflow.corrections import find_correction
    assert find_correction('这个方案没有问题', '这个方案没有问题另外还有一件事') is None
    assert find_correction('明天给张三发文件', '明天给李四发文件') == '明天给李四发文件'


def test_research_weights_require_explicit_license_acceptance():
    with pytest.raises(ValueError, match='research-only'):
        install(False)


def test_silent_audio_never_loads_or_hallucinates():
    from lipflow.whisper import ChineseWhisper
    r = ChineseWhisper()
    assert r.hypotheses(None) == []
    assert r.hypotheses(np.zeros(16000,np.float32)) == []
    assert r.model is None


def test_chinese_terms_rerank_inside_sentences():
    from lipflow.vocab import rerank
    assert rerank(['请发给李四', '请发给张三'], ['张三'])[0] == '请发给张三'


@pytest.mark.skipif(not Path('models/zh/vsr/model.pth').exists(), reason='optional research weights not installed')
def test_real_chinese_checkpoint_strict_loads_and_infers():
    from lipflow.vsr import LipReader
    reader = LipReader(language='zh', beam_size=2, use_lm=False, personal=False)
    enc = reader.encode(np.zeros((25,96,96),np.uint8))
    hs = reader.hypotheses(enc, 2)
    assert len(reader.token_list) == 3363
    assert all(np.isfinite(h.score) and h.token_count > 0 for h in hs)
    # Synthetic frames test compatibility only; they are NOT an accuracy evaluation.


def test_chinese_recipient_context_and_embedded_latin_term():
    from lipflow.context import extract_names
    from lipflow.text import contains
    assert '张三' in extract_names('收件人：张三')
    assert '李四' in extract_names('李四（私聊）')
    assert extract_names('我们明天讨论这个问题') == []
    assert contains('请发送给Miguel', 'Miguel')
