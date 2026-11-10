"""Unix domain socket IPC triggers for Wayland and X11 hotkey bindings."""
from __future__ import annotations

import os
import socket
import tempfile
import threading
from typing import Callable

SOCKET_NAME = "lipflow.sock"


def socket_path() -> str:
    """Return path to the Lipflow IPC socket. Prefers $XDG_RUNTIME_DIR."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and os.path.isdir(runtime_dir):
        return os.path.join(runtime_dir, SOCKET_NAME)
    return os.path.join(tempfile.gettempdir(), f"lipflow-{os.getuid() if hasattr(os, 'getuid') else 1000}.sock")


def send_trigger(cmd: str) -> bool:
    """Send an IPC trigger command ('start', 'stop', 'toggle', 'cancel') to a running Lipflow instance."""
    path = socket_path()
    if not os.path.exists(path):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect(path)
            s.sendall(cmd.strip().encode("utf-8"))
            try:
                s.recv(16)
            except (TimeoutError, OSError):
                pass
            return True
    except OSError:
        return False


class IPCServer:
    """Background listener for trigger commands sent by the CLI."""

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_toggle: Callable[[], None],
        on_cancel: Callable[[], None],
    ):
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_toggle = on_toggle
        self.on_cancel = on_cancel
        self.sock_path = socket_path()
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> bool:
        """Start listening on the domain socket with 0600 permissions."""
        if os.path.exists(self.sock_path):
            try:
                os.remove(self.sock_path)
            except OSError:
                pass
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(self.sock_path)
            try:
                os.chmod(self.sock_path, 0o600)
            except OSError:
                pass
            sock.listen(5)
            self._server_sock = sock
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="lipflow-ipc", daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            print(f"[lipflow] warning: could not bind IPC socket at {self.sock_path}: {e}")
            return False

    def _loop(self):
        while self._running and self._server_sock:
            try:
                conn, _ = self._server_sock.accept()
            except OSError:
                break
            with conn:
                try:
                    conn.settimeout(0.5)
                    data = conn.recv(64).decode("utf-8").strip()
                    if data == "start":
                        self.on_start()
                    elif data == "stop":
                        self.on_stop()
                    elif data == "toggle":
                        self.on_toggle()
                    elif data == "cancel":
                        self.on_cancel()
                    conn.sendall(b"ok\n")
                except Exception:
                    pass

    def stop(self):
        """Stop listener and clean up socket file."""
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None
        if os.path.exists(self.sock_path):
            try:
                os.remove(self.sock_path)
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.5)
