"""Push-to-talk timing (shared by macOS and Windows) and the Windows key mapping. Runs everywhere."""
import pytest

try:
    from pynput.keyboard import Key, KeyCode
except Exception:
    Key = KeyCode = None

from lipflow.ptt import DOUBLE_TAP, TAP_MAX, PushToTalkState

T = 1_000_000.0  # a realistic clock: last_tap starts at 0


def make(cls=PushToTalkState, *args):
    log = []
    ptt = cls(*args, lambda hands_free: log.append(("start", hands_free)),
              lambda: log.append(("stop",)), lambda silent=False: log.append(("cancel", silent)))
    return ptt, log


def test_hold_to_talk():
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.key_up(T + TAP_MAX + 0.1)
    assert log == [("start", False), ("stop",)]


def test_key_repeat_is_one_press():
    ptt, log = make()
    for t in (0.0, 0.05, 0.1, 0.15):
        ptt.key_down(T + t)
    ptt.key_up(T + 1.0)
    assert log == [("start", False), ("stop",)]


def test_single_tap_is_silently_ignored():
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.key_up(T + 0.1)
    assert log == [("start", False), ("cancel", True)]


def test_double_tap_enters_hands_free_until_next_tap():
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.key_up(T + 0.1)
    ptt.key_down(T + 0.2); ptt.key_up(T + 0.3)
    assert log[-1] == ("start", True) and ptt.hands_free
    ptt.key_down(T + 3.0); ptt.key_up(T + 3.1)
    assert log[-1] == ("stop",) and not ptt.hands_free


def test_slow_taps_are_not_a_double_tap():
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.key_up(T + 0.1)
    ptt.key_down(T + 0.1 + DOUBLE_TAP + 0.1); ptt.key_up(T + 0.2 + DOUBLE_TAP + 0.1)
    assert ("start", True) not in log


def test_shortcut_cancels_and_escape_cancels():
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.other_key(is_esc=False); ptt.key_up(T + 1.0)
    assert log == [("start", False), ("cancel", False)]
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.other_key(is_esc=True)
    assert log[-1] == ("cancel", False)


def test_escape_cancels_hands_free():
    ptt, log = make()
    ptt.key_down(T + 0.0); ptt.key_up(T + 0.1); ptt.key_down(T + 0.2); ptt.key_up(T + 0.3)
    ptt.other_key(is_esc=True)
    assert log[-1] == ("cancel", False) and not ptt.hands_free


@pytest.mark.skipif(Key is None, reason="No display server for pynput")
def test_windows_keys_map_through_pynput(monkeypatch):
    from lipflow.win.hotkey import KEYS, PushToTalk
    assert Key.ctrl_r in KEYS["right_control"] and Key.alt_gr in KEYS["right_alt"]
    ptt, log = make(PushToTalk, "right_control")
    ptt.press(Key.ctrl_r); ptt.press(Key.ctrl_r)  # key repeat
    ptt.press(Key.shift)                           # a modifier alone is not a shortcut
    ptt.release(Key.shift)
    ptt.down_at -= 1.0                             # held for a second
    ptt.release(Key.ctrl_r)
    assert log == [("start", False), ("stop",)]
    ptt, log = make(PushToTalk, "right_control")
    ptt.press(Key.ctrl_r); ptt.press(KeyCode.from_char("c"))
    assert log == [("start", False), ("cancel", False)]


@pytest.mark.skipif(Key is None, reason="No display server for pynput")
def test_windows_altgr_fake_ctrl_does_not_cancel(monkeypatch):
    from lipflow.win.hotkey import PushToTalk
    monkeypatch.setattr(PushToTalk, "_mask_alt", staticmethod(lambda: None))  # would inject a real key
    ptt, log = make(PushToTalk, "right_alt")
    ptt.press(Key.ctrl_l); ptt.press(Key.alt_gr)   # AltGr = a synthetic Left Ctrl + Right Alt
    ptt.press(Key.ctrl_l); ptt.press(Key.alt_gr)   # both repeat while held
    ptt.down_at -= 1.0
    ptt.release(Key.alt_gr)
    assert log == [("start", False), ("stop",)]
