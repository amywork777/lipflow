"""Microphone for whisper mode: on only while you hold the key, timestamped on the same clock as
the camera frames, so the audio can be cut to exactly the recorded video.

Timing matters: on test clips, audio 0.4 s off the video doubled the word error rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import threading

import numpy as np

from .mic_worker import RATE, STOP_TIMEOUT

CAMERA_LATENCY = 0.06  # a frame is read ~2 frames after it was exposed


@dataclass
class _Capture:
    process: subprocess.Popen
    chunks: list = field(default_factory=list)
    ready: threading.Event = field(default_factory=threading.Event)


def _command(device):
    # The native .app embeds Python, so sys.executable need not be a Python CLI.
    python = sys.executable
    if os.environ.get("LIPFLOW_APP"):
        python = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    elif Path(python).name.lower() == "pythonw.exe":
        # The Windows tray launcher has no standard streams; the helper needs pipes.
        python = str(Path(python).with_name("python.exe"))
    return [python, str(Path(__file__).with_name("mic_worker.py")), json.dumps(device)]


class Mic:
    def __init__(self, device=None):
        # Device enumeration can hang too; all audio-driver calls belong to the child.
        self.device = device
        self._capture: _Capture | None = None
        self._lock = threading.Lock()
        self.error: str | None = None

    def start(self):
        with self._lock:
            if self._capture is not None:
                return
            self.error = None
            try:
                process = subprocess.Popen(_command(self.device), stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                           close_fds=True,
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except OSError as exc:
                self.error = str(exc)
                print(f"[lipflow] microphone unavailable: {exc}")
                return
            capture = self._capture = _Capture(process)
        threading.Thread(target=self._receive, args=(capture,), name="lipflow-mic", daemon=True).start()

    def _receive(self, capture: _Capture):
        try:
            # This private pipe receives only messages from our own capture helper.
            with capture.process.stdout as stream:
                while True:
                    message = pickle.load(stream)
                    with self._lock:
                        if self._capture is not capture:
                            continue  # Never let a late callback bleed into the next recording.
                        if message[0] == "audio":
                            capture.chunks.append((message[1], message[2]))
                            capture.ready.set()
                        elif message[0] == "error":
                            self.error = message[1]
                            print(f"[lipflow] microphone unavailable: {self.error}")
        except (EOFError, OSError, pickle.UnpicklingError):
            with self._lock:
                if self._capture is capture and self.error is None:
                    self.error = "Microphone capture stopped unexpectedly"

    def stop(self) -> list[tuple[float, np.ndarray]]:
        """Detach the completed recording immediately; never wait for an audio driver."""
        with self._lock:
            capture, self._capture = self._capture, None
        if capture is None:
            return []
        capture.process.stdin.close()  # EOF requests shutdown, including during a stuck start.
        threading.Thread(target=_reap, args=(capture.process,), name="lipflow-mic-close", daemon=True).start()
        return capture.chunks


def _reap(process):
    try:
        process.wait(timeout=STOP_TIMEOUT + 1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def segment(chunks: list[tuple[float, np.ndarray]], t_start: float, n_frames: int, fps: int = 25) -> "np.ndarray | None":
    """Audio for n_frames video frames starting at t_start (video clock), 640 samples per frame."""
    if not chunks:
        return None
    t_start -= CAMERA_LATENCY
    want = n_frames * RATE // fps
    out = np.zeros(want, np.float32)
    got = 0
    for t0, a in chunks:
        off = int(round((t0 - t_start) * RATE))  # where this chunk lands in the output
        lo, hi = max(off, 0), min(off + len(a), want)
        if hi > lo:
            out[lo:hi] = a[lo - off:hi - off]
            got += hi - lo
    return out if got > want * 0.5 else None
