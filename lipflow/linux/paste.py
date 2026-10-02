"""Insert text at the cursor on Linux: clipboard + Ctrl+V (wtype on Wayland, xdotool on X11)."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time


def _wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"


def get_text() -> str | None:
    if _wayland() and shutil.which("wl-paste"):
        r = subprocess.run(["wl-paste", "-n"], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    if shutil.which("xclip"):
        r = subprocess.run(["xclip", "-selection", "clipboard", "-o"], capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else None
    return None


def set_text(text: str) -> None:
    if _wayland() and shutil.which("wl-copy"):
        subprocess.run(["wl-copy", text], check=True)
        return
    if shutil.which("xclip"):
        subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=True)
        return
    raise OSError("install wl-clipboard (Wayland) or xclip (X11)")


def _press_ctrl_v() -> None:
    if _wayland() and shutil.which("wtype"):
        subprocess.run(["wtype", "-M", "ctrl", "-k", "v"], check=True)
        return
    if shutil.which("xdotool"):
        subprocess.run(["xdotool", "key", "ctrl+v"], check=True)
        return
    raise OSError("install wtype (Wayland) or xdotool (X11)")


def paste_text(text: str, restore_after: float = 0.6) -> None:
    if not text:
        return
    try:
        saved = get_text()
    except OSError:
        saved = None
    set_text(text)
    _press_ctrl_v()

    def restore():
        time.sleep(restore_after)
        if saved is not None:
            try:
                set_text(saved)
            except OSError:
                pass

    threading.Thread(target=restore, daemon=True).start()


def copy_text(text: str) -> None:
    set_text(text)
