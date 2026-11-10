"""Private microphone helper. Native audio hangs must not hold the dictation UI hostage.

Audio travels over an anonymous pipe, never a file or network connection. Closing
stdin stops capture. A watchdog exits even if PortAudio hangs in start/stop/close,
so the OS can reclaim the microphone. This also works if the parent app exits.
"""
from __future__ import annotations

import json
import os
import pickle
import queue
import sys
import threading
import time

import numpy  # noqa: F401  (see capture: native imports must precede the stdin watcher)

RATE = 16000
STOP_TIMEOUT = 1.0


def _builtin(sd):
    """Prefer the built-in microphone over a Continuity iPhone."""
    for i, device in enumerate(sd.query_devices()):
        name = device["name"].lower()
        if device["max_input_channels"] > 0 and "iphone" not in name:
            if "macbook" in name or "built-in" in name:
                return i
    return None


def capture(device=None):
    # On Windows, loading a native DLL (numpy, PortAudio) deadlocks while another thread
    # has a blocking read pending on the stdin pipe, so import before starting the watcher.
    try:
        import sounddevice as sd
    except Exception as exc:
        sd, import_error = None, exc
    stop, closed = threading.Event(), threading.Event()

    def watch_parent():
        sys.stdin.buffer.read(1)
        stop.set()
        if not closed.wait(STOP_TIMEOUT):
            os._exit(2)  # Do not run PortAudio's atexit handlers after a native deadlock.

    threading.Thread(target=watch_parent, daemon=True).start()
    messages = queue.SimpleQueue()
    stream = None
    latency = 0.0

    def callback(data, frames, timing, status):
        if not stop.is_set():
            timestamp = time.time() - frames / RATE - latency
            messages.put(("audio", timestamp, data[:, 0].copy()))

    def send(message):
        pickle.dump(message, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.flush()

    try:
        if sd is None:
            raise import_error
        if stop.is_set():
            return
        if device is None:
            device = _builtin(sd)
        stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", device=device,
                                blocksize=0, callback=callback)
        latency = float(stream.latency or 0.0)
        if stop.is_set():
            return
        stream.start()
        while not stop.is_set():
            try:
                send(messages.get(timeout=0.02))
            except queue.Empty:
                pass
    except BrokenPipeError:
        os._exit(0)  # The parent exited; there is nowhere to deliver audio.
    except Exception as exc:
        try:
            send(("error", str(exc)))
        except BrokenPipeError:
            os._exit(0)
    finally:
        # Bound cleanup even when startup failed before the parent requested stop.
        def deadline():
            if not closed.wait(STOP_TIMEOUT):
                os._exit(2)
        threading.Thread(target=deadline, daemon=True).start()
        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            closed.set()
            # PortAudio also has an atexit termination path that can hang on macOS.
            os._exit(0)


if __name__ == "__main__":
    capture(json.loads(sys.argv[1]))
