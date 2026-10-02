"""Windows front end: clipboard, window context, overlay, tray icon and setup window.
Runs on Windows only (CI: .github/workflows/windows.yml). Nothing here presses keys."""
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


def test_data_lives_in_appdata(monkeypatch):
    import importlib
    import lipflow.paths as paths
    monkeypatch.delenv("LIPFLOW_HOME", raising=False)
    monkeypatch.setenv("APPDATA", r"C:\Users\me\AppData\Roaming")
    try:
        assert importlib.reload(paths).HOME == r"C:\Users\me\AppData\Roaming\Lipflow"
    finally:
        importlib.reload(paths)


def test_clipboard_round_trip_with_unicode():
    from lipflow.win import paste
    try:
        before = paste.get_text()
    except OSError:
        pytest.skip("no clipboard in this session")
    paste.set_text("lipflow · naïve ✓")
    assert paste.get_text() == "lipflow · naïve ✓"
    if before is not None:
        paste.set_text(before)


def test_context_capture_never_raises():
    from lipflow.context import capture
    ctx = capture()
    assert isinstance(ctx.title, str) and isinstance(ctx.app, str)


def test_word_list_ships_for_name_snapping():
    from lipflow.visemes import is_word
    assert is_word("plank") and is_word("meeting") and not is_word("mccall")


def test_camera_backend_and_names():
    from lipflow.camera import list_cameras, resolve_camera
    assert list_cameras() == [] and resolve_camera("auto") == 0 and resolve_camera(2) == 2


@pytest.fixture
def tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    root.withdraw()
    yield root
    root.destroy()


def test_overlay_shows_text_and_video_without_taking_focus(tk_root):
    from lipflow.win.hud import HUD
    hud = HUD(tk_root)
    assert hud.hwnd
    hud.show("listening", "Listening", "")
    hud.set_frame(np.zeros((70, 112, 3), np.uint8))
    hud.set_text("the birch canoe slid on the smooth planks and kept on going for a while")
    tk_root.update()
    assert hud.body_text.startswith("the birch") and hud._visible
    hud.show("done", "Pasted", "Hello.", hide_after=0.01)
    hud.hide()
    tk_root.update()
    assert not hud._visible


def test_tray_icons():
    from lipflow.win.hud import tray_image
    assert tray_image(False).size == (64, 64) and tray_image(True).getpixel((32, 32))[3] == 255


def test_setup_window_pages(tk_root, tmp_path, monkeypatch):
    import queue
    monkeypatch.setattr("lipflow.practice.CLIPS", str(tmp_path))
    from lipflow.win.setup import Setup
    camera = types.SimpleNamespace(track_always=False, ensure_open=lambda: None)
    app = types.SimpleNamespace(root=tk_root, key_name="Right Ctrl", camera=camera, onboarding=None,
                                opts=types.SimpleNamespace(language="en"),
                                onboarding_text="", reader=None, jobs=queue.Queue(), settings={},
                                ui=lambda fn, *a, **k: fn(*a, **k),
                                hud=types.SimpleNamespace(show=lambda *a, **k: None))
    s = Setup(app)
    s.show()
    s.goWords()
    s.goPractice()
    assert app.onboarding is s and app.onboarding_text
    s.set_frame(np.zeros((130, 208, 3), np.uint8))
    s.clip_done(False, "Too short. Hold the key.")
    s.clip_done(True, "", np.zeros((30, 88, 88), np.uint8), app.onboarding_text, "THE BIRCH")
    assert s.i == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    s.redo()
    assert s.i == 0 and not list(tmp_path.glob("*.npz"))
    s.goTrain()
    assert app.jobs.get_nowait()[0] == "train"
    s.report(50, "Training")
    s.finished(0.4, 0.3, True, "")
    tk_root.update()
    s.close()


def test_windows_app_imports():
    import lipflow.win.app as A
    assert A.key_label("right_control") == "Right Ctrl"
    assert "-m lipflow" in A.Lipflow._autostart_command()
