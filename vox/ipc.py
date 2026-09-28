"""Tiny Unix-socket control channel: `vox ctl listen` etc. Also keeps vox single-instance.

Bind `vox ctl listen` to a keyboard shortcut in KDE for push-to-talk.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
from pathlib import Path

from .config import runtime_socket

log = logging.getLogger(__name__)

COMMANDS = ("listen", "stop", "mute", "unmute", "toggle-mute", "reset", "ask", "quit", "ping")


def send(command: str, path: Path | None = None, timeout: float = 2.0) -> str:
    path = path or runtime_socket()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        s.sendall(command.encode() + b"\n")
        return s.recv(4096).decode().strip()


def is_running(path: Path | None = None) -> bool:
    try:
        return send("ping", path, timeout=0.5) == "pong"
    except OSError:
        return False


class Server:
    def __init__(self, handler, path: Path | None = None):
        """handler(cmd: str, arg: str) -> str"""
        self.path = path or runtime_socket()
        self.handler = handler
        if self.path.exists():
            self.path.unlink()  # stale socket from a crashed run (is_running() was checked before)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(self.path))
        os.chmod(self.path, 0o600)
        self.sock.listen(4)
        threading.Thread(target=self._serve, name="vox-ipc", daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    line = conn.recv(8192).decode().strip()
                    cmd, _, arg = line.partition(" ")
                    reply = "pong" if cmd == "ping" else self.handler(cmd, arg.strip())
                    conn.sendall((reply or "ok").encode())
                except Exception as e:  # noqa: BLE001
                    log.warning("ipc error: %s", e)

    def close(self):
        try:
            self.sock.close()
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def dispatch(engine, cmd: str, arg: str, on_quit=None) -> str:
    if cmd == "listen":
        engine.listen_now()
    elif cmd == "stop":
        engine.stop()
    elif cmd == "mute":
        engine.set_muted(True)
    elif cmd == "unmute":
        engine.set_muted(False)
    elif cmd == "toggle-mute":
        engine.set_muted(not engine.muted)
    elif cmd == "reset":
        engine.new_conversation()
    elif cmd == "ask":
        if not arg:
            return "usage: vox ctl ask <question>"
        engine.ask(arg)
    elif cmd == "quit":
        (on_quit or engine.shutdown)()
    else:
        return f"unknown command {cmd!r}; try: {', '.join(COMMANDS)}"
    return "ok"
