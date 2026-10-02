from types import SimpleNamespace
from lipflow.cleanup import Cleaner
from lipflow.confidence import Hypothesis as H, Quality, Assessment
from lipflow.delivery import choose_result
from lipflow.personal import Personal


def test_candidates_include_raw_even_when_cleanup_changes_sensitive_word(monkeypatch):
    c = Cleaner('basic', language='zh'); c.personal = Personal('/nonexistent')
    c.backend = 'ollama'; monkeypatch.setattr(c, '_ollama', lambda *args:'明天发送')
    decision, result, choices = choose_result([H('不要明天发送', -1, 6), H('不要今天发送', -2, 6)],
        '不要明天发送', Quality(), c, None, '')
    assert decision.action == 'review' and result.needs_review
    assert ('Raw / 原始识别', '不要明天发送。') in choices
    assert len(choices) <= 3


def test_camera_failure_skips_cleanup():
    c = SimpleNamespace(process=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('should not call')))
    d,r,choices = choose_result([H('hello', -1, 1)],'hello',Quality(face_ratio=.1),c,None,'')
    assert d.action == 'retry' and r is None and choices == []


def test_mac_stale_result_never_pastes(monkeypatch):
    import pytest
    pytest.importorskip('AppKit')
    from lipflow.app import Lipflow
    # Invoke the pure Python method with a test double; no GUI or keyboard side effects.
    app = SimpleNamespace(session=2)
    rec = SimpleNamespace(session=1)
    Lipflow._deliver(app, rec, None, [], Assessment('auto','',1,True), None, [], 0)
    Lipflow._publish(app, rec, None, [], None, None, 0, 'hello')


def test_changed_target_copies_instead_of_pasting(monkeypatch):
    import pytest
    pytest.importorskip('AppKit')
    import lipflow.app as A
    import lipflow.delivery as D
    seen=[]
    app=SimpleNamespace(session=2, opts=SimpleNamespace(paste=True),
        hud=SimpleNamespace(show=lambda *args:None))
    rec=SimpleNamespace(session=2,ctx=None)
    monkeypatch.setattr(D,'target_is_current',lambda ctx:False)
    monkeypatch.setattr(A,'copy_text',seen.append)
    monkeypatch.setattr(A,'paste_text',lambda text: (_ for _ in ()).throw(AssertionError('must not paste')))
    A.Lipflow._publish(app, rec, None, [], None, None, 0, '你好')
    assert seen==['你好'] and app.last_output=='你好'
