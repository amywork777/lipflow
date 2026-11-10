"""Run the real capture/helper protocol against a driver that can deliberately hang."""
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from lipflow import mic


@pytest.fixture
def fake_driver(tmp_path, monkeypatch):
    (tmp_path / "sounddevice.py").write_text('''
import os
import time
import numpy as np

def stall(stage):
    if os.environ.get("TEST_AUDIO_STALL") == stage:
        time.sleep(60)

def query_devices():
    stall("enumerate")
    return [{"max_input_channels": 1, "name": "MacBook microphone"}]

class InputStream:
    latency = 0.01
    def __init__(self, *, callback, device, **kwargs):
        self.callback, self.device = callback, device
        stall("open")
    def start(self):
        stall("start")
        if os.environ.get("TEST_AUDIO_ERROR"):
            raise RuntimeError("Microphone unavailable for test")
        self.callback(np.full((160, 1), self.device + 1, np.float32), 160, None, None)
    def stop(self):
        self.callback(np.full((160, 1), 99, np.float32), 160, None, None)
        stall("stop")
    def close(self):
        stall("close")
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.delenv("LIPFLOW_APP", raising=False)
    return tmp_path


@pytest.mark.parametrize("stage", ["stop", "close"])
def test_driver_stall_cannot_block_recording_completion(fake_driver, monkeypatch, stage):
    monkeypatch.setenv("TEST_AUDIO_STALL", stage)
    recorder = mic.Mic(device=0)
    recorder.start()
    capture = recorder._capture
    try:
        assert capture.ready.wait(3), recorder.error
        start = time.monotonic()
        chunks = recorder.stop()
        assert time.monotonic() - start < 0.5, "Driver shutdown blocked dictation"
        assert len(chunks) == 1 and np.all(chunks[0][1] == 1)
        assert capture.process.wait(timeout=3) == 2
        assert recorder.stop() == []
    finally:
        recorder.stop()
        if capture.process.poll() is None:
            capture.process.kill()
        capture.process.wait()


@pytest.mark.parametrize("stage", ["enumerate", "open", "start"])
def test_cancel_during_driver_startup_cannot_hang(fake_driver, monkeypatch, stage):
    monkeypatch.setenv("TEST_AUDIO_STALL", stage)
    recorder = mic.Mic()
    started = time.monotonic()
    recorder.start()
    capture = recorder._capture
    try:
        assert time.monotonic() - started < 0.5
        time.sleep(0.2)
        assert recorder.stop() == []
        capture.process.wait(timeout=3)
    finally:
        recorder.stop()
        if capture.process.poll() is None:
            capture.process.kill()
        capture.process.wait()


@pytest.mark.parametrize("launcher", ["console", "windowed"])
def test_repeat_recordings_are_isolated_and_timestamps_are_preserved(fake_driver, monkeypatch, launcher):
    if launcher == "windowed":
        if os.name != "nt":
            pytest.skip("Windows tray launcher")
        monkeypatch.setattr(sys, "executable", str(Path(sys.executable).with_name("pythonw.exe")))
    recorder = mic.Mic(device=0)
    for device in (0, 1, 2):
        recorder.device = device
        started = time.time()
        recorder.start()
        capture = recorder._capture
        try:
            recorder.start()
            assert recorder._capture is capture
            assert capture.ready.wait(3), recorder.error
            chunks = recorder.stop()
            assert len(chunks) == 1
            timestamp, samples = chunks[0]
            assert started - 0.1 <= timestamp <= time.time()
            assert samples.dtype == np.float32 and samples.shape == (160,)
            assert np.all(samples == device + 1)
            assert capture.process.wait(timeout=3) == 0
        finally:
            recorder.stop()
            if capture.process.poll() is None:
                capture.process.kill()
            capture.process.wait()


def test_unavailable_microphone_does_not_lose_lip_recording(fake_driver, monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_ERROR", "1")
    recorder = mic.Mic(device=0)
    recorder.start()
    capture = recorder._capture
    capture.process.wait(timeout=3)
    deadline = time.monotonic() + 1
    while recorder.error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorder.error == "Microphone unavailable for test"
    assert recorder.stop() == []


def test_parent_disappearance_also_reclaims_a_stalled_microphone(fake_driver, monkeypatch):
    monkeypatch.setenv("TEST_AUDIO_STALL", "stop")
    child = subprocess.Popen(mic._command(0), stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        import pickle
        assert pickle.load(child.stdout)[0] == "audio"
        child.stdin.close()
        assert child.wait(timeout=3) == 2
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        child.stdout.close()


def test_launch_failure_is_a_recoverable_microphone_error(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("No helper executable")
    monkeypatch.setattr(mic.subprocess, "Popen", fail)
    recorder = mic.Mic()
    recorder.start()
    assert recorder.error == "No helper executable"
    assert recorder.stop() == []
