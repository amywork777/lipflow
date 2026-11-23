"""Picking a Windows capture backend that actually shows a picture (issue #24: some cameras give
DirectShow nothing but black frames). Runs everywhere: OpenCV is replaced by a fake."""
import numpy as np
import pytest

import lipflow.camera as C

DSHOW, MSMF, ANY = C.cv2.CAP_DSHOW, C.cv2.CAP_MSMF, C.cv2.CAP_ANY
BLACK = np.zeros((4, 4, 3), np.uint8)
FACE = np.full((4, 4, 3), 120, np.uint8)


@pytest.fixture
def fake_cv(monkeypatch, tmp_path):
    """behaviour[backend] -> 'picture' | 'black' | 'closed' | 'picture-default' (black once a
    resolution is forced). Returns the list of (backend, configured) opens."""
    behaviour, opened = {}, []

    class Cap:
        def __init__(self, idx=None, api=None):
            self.api, self.configured = api, False
            self.mode = behaviour.get(api, "closed") if idx is not None else "closed"
            opened.append(self)

        def set(self, *_):
            self.configured = True
            return True

        def isOpened(self):
            return self.mode != "closed"

        def read(self):
            if self.mode == "picture" or (self.mode == "picture-default" and not self.configured):
                return True, FACE
            return True, BLACK

        def release(self):
            pass

    monkeypatch.setattr(C.cv2, "VideoCapture", Cap)
    monkeypatch.setattr(C, "BACKEND_FILE", str(tmp_path / "camera_backend.json"))
    monkeypatch.setattr(C, "PROBE_SECONDS", 0.02)
    monkeypatch.delenv("LIPFLOW_CAMERA_BACKEND", raising=False)
    return behaviour, opened


def test_dshow_is_used_when_it_works(fake_cv):
    behaviour, opened = fake_cv
    behaviour.update({DSHOW: "picture", MSMF: "picture"})
    _, name, status = C.windows_open(0)
    assert (name, status) == ("dshow", "picture") and [c.api for c in opened] == [DSHOW]


def test_black_dshow_falls_back_to_msmf_and_is_remembered(fake_cv):
    behaviour, opened = fake_cv
    behaviour.update({DSHOW: "black", MSMF: "picture"})
    assert C.windows_open(0)[1:] == ("msmf", "picture")
    opened.clear()
    assert C.windows_open(0)[1:] == ("msmf", "picture")
    assert [c.api for c in opened] == [MSMF]  # next time straight to the one that worked


def test_retries_at_default_resolution(fake_cv):
    behaviour, _ = fake_cv
    behaviour.update({DSHOW: "closed", MSMF: "picture-default"})
    cap, name, status = C.windows_open(0)
    assert (name, status) == ("msmf", "picture") and not cap.configured


def test_all_dark_uses_first_that_sends_frames_without_remembering(fake_cv):
    behaviour, _ = fake_cv
    behaviour.update({DSHOW: "closed", MSMF: "black", ANY: "black"})
    assert C.windows_open(0)[1:] == ("msmf", "black")
    assert C._load_known() == {}


def test_nothing_opens(fake_cv):
    cap, _, status = C.windows_open(0)
    assert status == "none" and not cap.isOpened()


def test_env_var_forces_a_backend(fake_cv, monkeypatch):
    behaviour, opened = fake_cv
    behaviour.update({DSHOW: "picture", MSMF: "picture"})
    monkeypatch.setenv("LIPFLOW_CAMERA_BACKEND", "MSMF")
    assert C.windows_open(0)[1] == "msmf" and {c.api for c in opened} == {MSMF}


def test_has_picture():
    assert not C.has_picture(None) and not C.has_picture(BLACK) and C.has_picture(FACE)
