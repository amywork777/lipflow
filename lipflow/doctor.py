"""`lipflow doctor`: check everything the app needs before you hit the hotkey."""
from __future__ import annotations

import os
import shutil
import sys

from .vsr import MODELS
from .paths import WHO


def doctor() -> int:
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {what}" + ("" if good else f"\n      → {fix}"))

    if sys.platform == "win32":
        fix_models = "run .\\setup.ps1"
    elif sys.platform == "linux":
        fix_models = "run ./scripts/download-models.sh"
    else:
        fix_models = "run ./setup.sh"

    print("Lipflow doctor\n")
    for rel, size in [("vsr/model.pth", 900e6), ("lm/model.pth", 200e6), ("face_landmarker.task", 3e6)]:
        path = os.path.join(MODELS, rel)
        line(os.path.exists(path) and os.path.getsize(path) > size, f"model file {rel}", fix_models)

    import torch
    from .vsr import apple_gpu
    dev = "cuda (NVIDIA GPU)" if torch.cuda.is_available() else \
        "mps (Apple GPU)" if apple_gpu() else "cpu"
    line(True, f"torch {torch.__version__}, encoder on {dev}")

    if sys.platform == "win32":
        _windows_checks(line)
    elif sys.platform == "linux":
        _linux_checks(line)
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


def _linux_checks(line):
    from .linux.paste import wayland_session
    if wayland_session():
        from .linux.hotkey import keyboards_readable
        line(shutil.which("wl-copy") is not None, "wl-copy (clipboard)",
             "install wl-clipboard")
        line(any(shutil.which(tool) for tool in ("wtype", "ydotool", "dotool")),
             "Wayland paste (wtype, ydotool, or dotool)",
             "install wtype (Hyprland, Sway) or ydotool with ydotoold running (GNOME, KDE, other Wayland)")
        line(keyboards_readable(), "Wayland push-to-talk key (/dev/input)",
             "add your user to the input group and log in again")
    else:
        line(shutil.which("xclip") is not None, "xclip (clipboard)",
             "install xclip")
        line(shutil.which("xdotool") is not None, "xdotool (paste at cursor)",
             "install xdotool")
    line(shutil.which("wl-paste") is not None or shutil.which("xclip") is not None,
         "clipboard read (optional, for restore)", "same as above")

    import cv2
    from .camera import resolve_camera
    from .dictation import load_settings
    idx = resolve_camera(load_settings().get("camera", "auto"))
    cap = cv2.VideoCapture(idx)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    line(ok, f"Camera {idx} opens and sends frames",
         "check PipeWire/v4l2, close other apps using the camera, try --camera 0")
