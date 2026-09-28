import contextlib
import os
import time

import pytest

from vox import ipc
from vox.brain import Brain
from vox.config import load_config, set_api_key, tomllib


class FakeStream:
    def __init__(self, chunks):
        self.text_stream = iter(chunks)


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    @contextlib.contextmanager
    def stream(self, **kw):
        self.calls.append({**kw, "messages": [dict(m) for m in kw["messages"]]})
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        yield FakeStream(r)


def test_brain_memory_prompt_and_tools():
    client = FakeClient([["Cześć", " Szymon!"], ["Dobrze."]])
    b = Brain("", "claude-haiku-4-5-20251001", web_search=True, user_name="Szymon", location="Poznań, Poland",
              _client=client)
    assert "".join(b.ask("Cześć", "pl")) == "Cześć Szymon!"
    assert "".join(b.ask("Jak leci?", "pl")) == "Dobrze."
    first, second = client.calls
    assert "Reply in Polish" in first["system"] and "Poznań" in first["system"] and "Szymon" in first["system"]
    assert first["tools"][0]["name"] == "web_search" and first["tools"][0]["user_location"]["city"] == "Poznań"
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]


def test_brain_forgets_after_idle_and_recovers_from_errors():
    client = FakeClient([["A"], RuntimeError("boom"), ["B"]])
    b = Brain("", "m", memory_minutes=0.0001, _client=client)
    "".join(b.ask("one", "en"))
    with pytest.raises(RuntimeError):
        "".join(b.ask("two", "en"))
    time.sleep(0.02)
    "".join(b.ask("three", "en"))
    assert [m["content"] for m in client.calls[-1]["messages"]] == ["three"]   # old turns forgotten, no dangling "two"
    assert "tools" not in client.calls[0]


def test_config_defaults_and_api_key(tmp_path):
    p = tmp_path / "config.toml"
    cfg = load_config(p)                       # missing file -> defaults
    assert cfg.get("wake", "phrase") == "hey claude"
    from vox.config import ensure_config
    ensure_config(p)
    set_api_key("sk-ant-test123", p)
    cfg = load_config(p)
    assert cfg.api_key == "sk-ant-test123"
    assert oct(p.stat().st_mode)[-3:] == "600"
    tomllib.loads(p.read_text())               # still valid TOML
    p.write_text(p.read_text().replace('model = "small"', 'model = "medium"'))
    assert load_config(p).get("stt", "model") == "medium"
    assert load_config(p).get("tts", "voices")["pl"] == "pl_PL-gosia-medium"


def test_ipc_roundtrip(tmp_path):
    got = []
    sock = tmp_path / "vox.sock"
    assert not ipc.is_running(sock)
    server = ipc.Server(lambda c, a: got.append((c, a)) or "ok", sock)
    assert ipc.is_running(sock)
    assert ipc.send("ask jaka pogoda", sock) == "ok"
    assert got == [("ask", "jaka pogoda")]
    server.close()
    assert not ipc.is_running(sock)


def test_tray_builds_offscreen(tmp_path):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from vox import tray as T

    class Eng:
        muted = False
        def __init__(self, ev): self.ev = ev
        def listen_now(self): pass
        def stop(self): pass
        def set_muted(self, m): pass
        def new_conversation(self): pass
        def ask(self, t): pass
        def shutdown(self): pass
        def start(self): pass

    app = QApplication.instance() or QApplication([])
    t = T.TrayApp(app, Eng, popup="bubble")
    t.events.state("listening", "Listening…")
    t.events.user("Jaka jest jutro pogoda?")
    t.events.state("speaking", "Speaking…")
    t.events.reply("Jutro będzie słonecznie.", True)
    app.processEvents()
    assert t.bubble.isVisible() and "słonecznie" in t.bubble.answer.text()
    assert t.tray.toolTip().startswith("Vox – Speaking")
    t.events.state("muted", "Microphone off")
    app.processEvents()
    assert t.mute_action.isChecked()
    # save a preview of the icons + bubble for the README
    t.bubble.grab().save(str(tmp_path / "bubble.png"))
