import pytest
from lipflow.guard import check
from lipflow.cleanup import Cleaner
from lipflow.personal import Personal


@pytest.mark.parametrize('original,proposed', [
    ("DON'T SEND IT TOMORROW", 'Send it tomorrow.'),
    ('Pay $100', 'Pay 100'), ('Pay 100 yuan', 'Pay 1000 yuan'),
    ('Meet tomorrow at 4:30', 'Meet today at 4:30'),
    ('不要明天发送', '明天发送'), ('明天给张三转账五百元', '明天给张三转账五千元'),
    ('会议在十月二日', '会议在十月三日'), ('我没有同意这个方案', '我同意这个方案'),
])
def test_sensitive_changes_require_review(original, proposed):
    assert check(original, proposed, [proposed])


def test_name_change_requires_review_even_if_it_is_a_decoder_alternative():
    assert check('Hi Miguel', 'Hi Priya', ['Hi Priya'], ['Miguel', 'Priya'])
    assert check('发给张三', '发给李四', ['发给李四'], ['张三', '李四'])


def test_formatting_and_equivalent_english_numbers_are_allowed():
    assert not check('HELLO WORLD', 'Hello, world.')
    assert not check('你好世界', '你好，世界。')
    assert not check('IN NINETEEN FORTY THREE', 'In 1943.')


def test_polish_always_reviews_changed_wording():
    assert check('I like the plan', 'The plan looks good', mode='polish')


@pytest.mark.parametrize('backend', ['local', 'claude', 'ollama'])
def test_every_backend_applies_same_guard_without_network(backend, monkeypatch):
    cleaner = Cleaner('basic', language='zh')
    cleaner.personal = Personal('/nonexistent')
    cleaner.backend = backend
    monkeypatch.setattr(cleaner, '_'+backend, lambda *args: '明天发送。')
    result = cleaner.process(['不要明天发送'], words=[])
    assert result.needs_review
    assert result.raw == '不要明天发送' and result.text == '不要明天发送。'
    assert result.proposed == '明天发送。'


def test_empty_gibberish_and_large_edits_fall_back_to_raw(monkeypatch):
    c = Cleaner('basic')
    c.backend = 'ollama'
    monkeypatch.setattr(c, '_ollama', lambda *args: 'Here is a completely unrelated answer')
    r = c.process(['THE PLAN IS READY'], words=[])
    assert r.needs_review and r.text == 'The plan is ready.'


def test_chinese_custom_terms_are_not_snapped_with_english_visemes():
    c = Cleaner('basic', language='zh')
    c.personal = Personal('/nonexistent')
    r = c.process(['请发给张三', '请发给李四'], words=['张三'])
    assert r.text == '请发给张三。'
