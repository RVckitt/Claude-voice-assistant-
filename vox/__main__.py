"""Command line entry point.

  vox                  start in the system tray
  vox --no-tray        run in the terminal
  vox --setup          enter API key + download speech models
  vox --say "tekst"    test the voice output
  vox --list-devices   show microphones / speakers
  vox ctl listen       control a running vox (listen|stop|mute|unmute|toggle-mute|reset|ask <text>|quit)
"""
from __future__ import annotations

import argparse
import getpass
import logging
import logging.handlers
import signal
import sys
import time

from . import __version__
from .config import CONFIG_PATH, LOG_PATH, ensure_config, load_config, set_api_key


def setup_logging(verbose: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    sh.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=[fh, sh])
    for noisy in ("httpx", "httpcore", "anthropic", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_ctl(args) -> int:
    from . import ipc
    if not args.ctl:
        print("usage: vox ctl <listen|stop|mute|unmute|toggle-mute|reset|ask TEXT|quit>")
        return 2
    try:
        print(ipc.send(" ".join(args.ctl)))
        return 0
    except OSError:
        print("vox is not running.")
        return 1


def cmd_setup() -> int:
    ensure_config()
    cfg = load_config()
    print(f"Vox {__version__} setup — config: {CONFIG_PATH}\n")

    if cfg.get("claude", "api_key"):
        print("✓ API key already in config")
    else:
        print("Paste your Anthropic API key (https://platform.claude.com/settings/keys).")
        print("Leave empty to use the ANTHROPIC_API_KEY environment variable instead.")
        try:
            key = getpass.getpass("API key (hidden): ").strip()
        except (EOFError, KeyboardInterrupt):
            key = ""
            print("\n(skipped – run `vox --setup` again later)")
        if key:
            set_api_key(key)
            cfg = load_config()
            print("✓ saved (file is readable only by you)")

    if cfg.api_key:
        try:
            import anthropic
            r = anthropic.Anthropic(api_key=cfg.api_key).messages.create(
                model=cfg.get("claude", "model"), max_tokens=20,
                messages=[{"role": "user", "content": "Say 'ready' in Polish, one word."}])
            print(f"✓ Claude answered: {r.content[0].text.strip()}")
        except Exception as e:  # noqa: BLE001
            print(f"✗ Claude test failed: {e}")

    ok = True
    from .stt import load_whisper
    for name in dict.fromkeys([cfg.get("wake", "whisper_model"), cfg.get("stt", "model")]):
        if cfg.get("wake", "engine") != "whisper" and name == cfg.get("wake", "whisper_model") \
                and name != cfg.get("stt", "model"):
            continue
        print(f"… Whisper '{name}' (downloads once)")
        try:
            load_whisper(name, cfg.get("stt", "device"), cfg.get("stt", "compute_type"))
            print(f"✓ Whisper '{name}' ready")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"✗ Whisper '{name}': {e}")

    from .tts import ensure_voice
    for lang, voice in cfg.section("tts").get("voices", {}).items():
        print(f"… Piper voice {voice} ({lang})")
        try:
            ensure_voice(voice)
            print(f"✓ voice {voice} ready")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"✗ voice {voice}: {e}")

    try:
        from .audio import list_devices
        print("\nAudio devices (> = default input, < = default output):")
        print(list_devices())
    except Exception as e:  # noqa: BLE001
        print(f"✗ audio: {e}")
    if ok:
        print("\nDone. Start with:  vox")
    else:
        print("\nSome downloads failed (see above). Check your internet connection and run  vox --setup  again.")
    return 0


def cmd_say(text: str) -> int:
    cfg = load_config()
    from .audio import Player
    from .textutil import guess_lang
    from .tts import Speaker
    sp = Speaker(cfg.section("tts").get("voices", {}), cfg.get("tts", "speed", 1.0))
    wav, sr = sp.synthesize(text, guess_lang(text, (cfg.get("stt", "languages") or ["pl"])[0]))
    Player(cfg.get("audio", "output_device")).play(wav, sr)
    return 0


def run(no_tray: bool) -> int:
    from . import ipc
    if ipc.is_running():
        print("vox is already running (use `vox ctl quit` to stop it).")
        return 1
    cfg = load_config()
    from .engine import Engine, PrintEvents

    if no_tray:
        engine = Engine(cfg, PrintEvents())
        server = ipc.Server(lambda c, a: ipc.dispatch(engine, c, a))
        signal.signal(signal.SIGINT, lambda *_: engine.shutdown())
        signal.signal(signal.SIGTERM, lambda *_: engine.shutdown())
        engine.start()
        print("Vox running – Ctrl+C to quit. Type a question + Enter to ask by text.")
        try:
            import select
            stdin_open = sys.stdin is not None and sys.stdin.isatty()
            while engine.running:
                if not stdin_open:          # started from autostart / a script – just keep running
                    time.sleep(0.5)
                    continue
                r, _, _ = select.select([sys.stdin], [], [], 0.5)
                if r:
                    line = sys.stdin.readline()
                    if not line:
                        stdin_open = False
                    elif line.strip():
                        engine.ask(line.strip())
        except KeyboardInterrupt:
            pass
        engine.shutdown()
        server.close()
        return 0

    from PySide6.QtWidgets import QApplication, QSystemTrayIcon

    from .tray import TrayApp
    app = QApplication(sys.argv[:1])
    app.setApplicationName("vox")
    app.setDesktopFileName("vox")
    app.setQuitOnLastWindowClosed(False)
    if not QSystemTrayIcon.isSystemTrayAvailable():
        logging.warning("No system tray found – falling back to terminal mode")
        return run(no_tray=True)

    tray = TrayApp(app, lambda events: Engine(cfg, events), cfg.get("ui", "popup", "auto"))
    server = ipc.Server(lambda c, a: ipc.dispatch(tray.engine, c, a, on_quit=lambda: _later(tray.quit)))
    signal.signal(signal.SIGINT, lambda *_: tray.quit())
    signal.signal(signal.SIGTERM, lambda *_: tray.quit())
    # let Python handle signals while Qt's loop runs
    from PySide6.QtCore import QTimer
    t = QTimer()
    t.start(300)
    t.timeout.connect(lambda: None)
    tray.engine.start()
    code = app.exec()
    server.close()
    return code


def _later(fn):
    from PySide6.QtCore import QTimer
    QTimer.singleShot(0, fn)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vox", description="Hands-free voice assistant powered by Claude.")
    p.add_argument("--no-tray", action="store_true", help="run in the terminal instead of the system tray")
    p.add_argument("--setup", action="store_true", help="enter API key and download speech models")
    p.add_argument("--say", metavar="TEXT", help="speak TEXT to test the voice")
    p.add_argument("--list-devices", action="store_true", help="list audio devices")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"vox {__version__}")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["ctl"]:
        ns = argparse.Namespace(ctl=argv[1:])
        return cmd_ctl(ns)
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    if args.setup:
        return cmd_setup()
    if args.list_devices:
        from .audio import list_devices
        print(list_devices())
        return 0
    if args.say:
        return cmd_say(args.say)
    return run(args.no_tray)


if __name__ == "__main__":
    sys.exit(main())
