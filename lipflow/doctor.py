"""`lipflow doctor`: check everything the app needs before you hit the hotkey."""
from __future__ import annotations

import os
import sys

from .vsr import MODELS
from .paths import WHO


def doctor(language="en") -> int:
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {what}" + ("" if good else f"\n      → {fix}"))

    print("Lipflow doctor\n")
    model_files = ([("zh/vsr/model.pth", 200e6), ("zh/lm/model.pth", 180e6)] if language == "zh" else
                   [("vsr/model.pth", 900e6), ("lm/model.pth", 200e6)]) + [("face_landmarker.task", 3e6)]
    for rel, size in model_files:
        path = os.path.join(MODELS, rel)
        line(os.path.exists(path) and os.path.getsize(path) > size, f"model file {rel}",
             "run .\\setup.ps1" if sys.platform == "win32" else "run ./setup.sh")

    import torch
    dev = "cuda (NVIDIA GPU)" if torch.cuda.is_available() else \
        "mps (Apple GPU)" if torch.backends.mps.is_available() else "cpu"
    line(True, f"torch {torch.__version__}, encoder on {dev}")

    if sys.platform == "win32":
        _windows_checks(line)
    else:
        _mac_checks(line)

    from .cleanup import Cleaner
    c = Cleaner()
    line(True, f"cleanup backend: {c.describe()}"
         + ("  (set ANTHROPIC_API_KEY or run Ollama for much better accuracy)" if c.backend == "basic" else ""))
    print("\nAll good." if ok else "\nFix the ✗ items above, then run `lipflow`.")
    return 0 if ok else 1


def _mac_checks(line):
    import Quartz
    line(Quartz.CGPreflightListenEventAccess(), "Input Monitoring (for the push-to-talk key)",
         f"System Settings → Privacy & Security → Input Monitoring → enable {WHO}, then restart it")
    line(Quartz.CGPreflightPostEventAccess(), "Accessibility (to paste at your cursor)",
         f"System Settings → Privacy & Security → Accessibility → enable {WHO}")

    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    names = {0: "not asked yet (you'll be prompted on first use)", 1: "restricted", 2: "denied", 3: "granted"}
    line(status in (0, 3), f"Camera: {names.get(status, status)}",
         f"System Settings → Privacy & Security → Camera → enable {WHO}")


def _windows_checks(line):
    """Windows grants keyboard hooks and pasting to every desktop app; only the camera can be off."""
    import cv2
    from .camera import resolve_camera
    from .dictation import load_settings
    idx = resolve_camera(load_settings().get("camera", "auto"))
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    line(ok, f"Camera {idx} opens and sends frames",
         "Settings → Privacy & security → Camera → turn on \"Let desktop apps access your camera\", "
         "and close other apps using the camera")
