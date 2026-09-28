"""The assistant loop: wake word -> listen -> transcribe -> Claude -> speak -> (follow-up)."""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Protocol

from . import audio as A
from .config import Config
from .textutil import SentenceSplitter, clean_for_speech, guess_lang, is_noise, split_wake

log = logging.getLogger(__name__)

LOADING, IDLE, LISTENING, THINKING, SPEAKING, MUTED, ERROR = (
    "loading", "idle", "listening", "thinking", "speaking", "muted", "error")

ERROR_LINES = {
    "pl": "Przepraszam, nie mogę teraz połączyć się z Claude.",
    "en": "Sorry, I can't reach Claude right now.",
}


class Events(Protocol):
    def state(self, state: str, detail: str = "") -> None: ...
    def user(self, text: str) -> None: ...
    def reply(self, text: str, done: bool) -> None: ...
    def error(self, message: str) -> None: ...


class PrintEvents:
    """Terminal output (used with --no-tray)."""

    def state(self, state, detail=""):
        print(f"[{state}] {detail}".rstrip(), flush=True)

    def user(self, text):
        print(f"\n  you> {text}", flush=True)

    def reply(self, text, done):
        if done:
            print(f"claude> {text}\n", flush=True)

    def error(self, message):
        print(f"  !! {message}", flush=True)


class Engine:
    def __init__(self, cfg: Config, events: Events, **parts):
        self.cfg = cfg
        self.events = events
        self.parts = parts            # lets tests inject fakes: mic, player, stt, wake, brain, tts, segmenter
        self.commands: queue.Queue = queue.Queue()
        self.stop_speaking = threading.Event()
        self.muted = False
        self.running = False
        self.thread: threading.Thread | None = None
        self.state = LOADING

    # ------------------------------------------------------------------ public (any thread)
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._main, name="vox-engine", daemon=True)
        self.thread.start()

    def shutdown(self):
        self.running = False
        self.stop_speaking.set()
        self.commands.put(("quit", None))

    def listen_now(self):
        self.commands.put(("listen", None))

    def ask(self, text: str):
        self.commands.put(("ask", text))

    def stop(self):
        self.stop_speaking.set()

    def set_muted(self, muted: bool):
        self.commands.put(("mute", muted))
        if muted:
            self.stop_speaking.set()

    def new_conversation(self):
        self.commands.put(("reset", None))

    # ------------------------------------------------------------------ internals
    def _set(self, state: str, detail: str = ""):
        self.state = state
        self.events.state(state, detail)

    def _load(self):
        c = self.cfg
        self._set(LOADING, "Loading speech models…")
        p = self.parts
        self.mic = p.get("mic") or A.Microphone(c.get("audio", "input_device"))
        self.player = p.get("player") or A.Player(c.get("audio", "output_device"))
        if "segmenter" in p:
            self.seg = p["segmenter"]
        else:
            from .vad import Segmenter
            self.seg = Segmenter(c.get("audio", "vad_aggressiveness"), c.get("audio", "silence_ms"),
                                 c.get("audio", "max_utterance_s"), min_rms=c.get("audio", "min_volume", 60))
        self.languages = c.get("stt", "languages") or ["pl", "en"]
        if "stt" in p:
            self.stt = p["stt"]
        else:
            from .stt import Transcriber
            self._set(LOADING, f"Loading Whisper '{c.get('stt', 'model')}' (first run downloads it)…")
            self.stt = Transcriber(c.get("stt", "model"), c.get("stt", "device"), c.get("stt", "compute_type"),
                                   self.languages)
        self.wake_phrases = [c.get("wake", "phrase"), *c.get("wake", "aliases", [])]
        self.oww = None
        if "wake" in p:
            self.wake = p["wake"]
        elif c.get("wake", "engine") == "openwakeword":
            from .wake import OpenWakeWordWake
            self.oww = OpenWakeWordWake(c.get("wake", "openwakeword_model"), c.get("wake", "openwakeword_threshold"))
            self.wake = None
        else:
            from .wake import WhisperWake
            self._set(LOADING, "Loading wake-word model…")
            same = c.get("wake", "whisper_model") == c.get("stt", "model")
            self.wake = WhisperWake(c.get("wake", "phrase"), c.get("wake", "aliases", []),
                                    c.get("wake", "whisper_model"), c.get("wake", "sensitivity"),
                                    c.get("stt", "device"), _model=self.stt.model if same else None)
        if "tts" in p:
            self.tts = p["tts"]
        else:
            from .tts import Speaker
            self._set(LOADING, "Loading voices…")
            self.tts = Speaker(c.section("tts").get("voices", {}), c.get("tts", "speed", 1.0))
            self.tts.preload()
        if "brain" in p:
            self.brain = p["brain"]
        else:
            from .brain import Brain
            cl = c.section("claude")
            self.brain = Brain(c.api_key, cl["model"], cl["max_tokens"], cl["web_search"], cl["memory_minutes"],
                               cl["user_name"], cl["location"], cl["timezone"], cl["extra_instructions"])
            if not c.api_key:
                self.events.error("No Anthropic API key – set it in the config or run `vox --setup`.")
        self.mic.start()

    def _main(self):
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            log.exception("startup failed")
            self.running = False
            self._set(ERROR, str(e))
            self.events.error(f"Startup failed: {e}")
            return
        self._set(IDLE, self._idle_hint())
        while self.running:
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 – never let one bad turn kill the assistant
                log.exception("error in main loop")
                self.events.error(str(e))
                self._beep(A.ERROR_SOUND)
                self._set(MUTED if self.muted else IDLE, self._idle_hint())
        try:
            self.mic.stop()
        except Exception:  # noqa: BLE001
            pass

    def _idle_hint(self) -> str:
        return f'Say "{self.cfg.get("wake", "phrase")}"'

    def _pending(self) -> bool:
        return not self.running or not self.commands.empty()

    def _handle_commands(self) -> bool:
        """Runs queued commands. Returns True if something was handled."""
        handled = False
        while True:
            try:
                cmd, arg = self.commands.get_nowait()
            except queue.Empty:
                return handled
            handled = True
            if cmd == "quit":
                self.running = False
                return True
            if cmd == "mute":
                if bool(arg) != self.muted:
                    # really close the microphone while muted (the system mic indicator turns off)
                    try:
                        self.mic.stop() if arg else self.mic.start()
                    except Exception as e:  # noqa: BLE001
                        log.warning("mic %s failed: %s", "stop" if arg else "start", e)
                self.muted = bool(arg)
                self._set(MUTED if self.muted else IDLE, "Microphone off" if self.muted else self._idle_hint())
            elif cmd == "reset":
                self.brain.reset()
                self.events.reply("", True)
            elif cmd == "ask" and arg:
                self._conversation(arg, guess_lang(arg, self.languages[0]), allow_follow_up=not self.muted)
            elif cmd == "listen":
                self._listen_for_command()

    def _tick(self):
        if self._handle_commands() or not self.running:
            self._set(MUTED if self.muted else IDLE, "Microphone off" if self.muted else self._idle_hint())
            return
        if self.muted:
            self.mic.flush()
            time.sleep(0.1)
            return

        if self.oww is not None:
            frame = self.mic.read()
            if frame is not None and self.oww.feed(frame):
                self._listen_for_command()
                self._set(IDLE, self._idle_hint())
            return

        audio = self.seg.next_utterance(self.mic.read, abort=self._pending)
        if audio is None:
            return
        res = self.wake.check(audio)
        if not res.matched:
            return
        if res.has_more:
            # "hey claude, what's the weather" in one breath – transcribe it all with the better model
            self._set(THINKING, "Transcribing…")
            t = self.stt.transcribe(audio, prompt=self.cfg.get("wake", "phrase").capitalize() + ",")
            matched, command = split_wake(t.text, self.wake_phrases, self.cfg.get("wake", "sensitivity"))
            command = command if matched else t.text
            if command and not is_noise(command):
                self._beep(A.WAKE_SOUND)
                self._conversation(command, t.language)
                self._set(IDLE, self._idle_hint())
                return
        self._listen_for_command()
        self._set(IDLE, self._idle_hint())

    def _beep(self, sound):
        if self.cfg.get("audio", "beep", True):
            try:
                self.player.play(sound, 22050)
            except Exception as e:  # noqa: BLE001
                log.debug("beep failed: %s", e)

    def _record(self, timeout: float):
        self.mic.flush()
        return self.seg.next_utterance(self.mic.read, abort=lambda: not self.running or self.stop_speaking.is_set(),
                                       start_timeout=timeout)

    def _listen_for_command(self):
        self.stop_speaking.clear()
        self._beep(A.WAKE_SOUND)
        self._set(LISTENING, "Listening…")
        audio = self._record(self.cfg.get("audio", "command_timeout_s"))
        if audio is None:
            self._beep(A.DONE_SOUND)
            return
        self._set(THINKING, "Transcribing…")
        t = self.stt.transcribe(audio)
        if is_noise(t.text):
            self._beep(A.DONE_SOUND)
            return
        matched, rest = split_wake(t.text, self.wake_phrases, self.cfg.get("wake", "sensitivity"))
        self._conversation(rest if matched and rest else t.text, t.language)

    def _conversation(self, text: str, lang: str, allow_follow_up: bool = True):
        follow_up = float(self.cfg.get("audio", "follow_up_s", 0)) if allow_follow_up else 0
        while self.running:
            finished = self._turn(text, lang)
            if not finished or follow_up <= 0 or self.muted:
                return
            self._set(LISTENING, "Anything else?")
            audio = self._record(follow_up)
            if audio is None:
                return
            self._set(THINKING, "Transcribing…")
            t = self.stt.transcribe(audio)
            if is_noise(t.text):
                return
            matched, rest = split_wake(t.text, self.wake_phrases, self.cfg.get("wake", "sensitivity"))
            text, lang = (rest if matched and rest else t.text), t.language

    def _turn(self, text: str, lang: str) -> bool:
        """One question -> spoken answer. Returns False if interrupted or failed."""
        self.stop_speaking.clear()
        self.events.user(text)
        self._set(THINKING, "Thinking…")
        log.info("user (%s): %s", lang, text)

        sentences: queue.Queue = queue.Queue()
        speak_errors: list[Exception] = []

        def speaker():
            while True:
                s = sentences.get()
                if s is None:
                    return
                if self.stop_speaking.is_set() or speak_errors:
                    continue
                spoken = clean_for_speech(s)
                if not spoken:
                    continue
                try:
                    wav, sr = self.tts.synthesize(spoken, guess_lang(spoken, lang))
                    if self.stop_speaking.is_set():
                        continue
                    self._set(SPEAKING, "Speaking…")
                    self.player.play(wav, sr, self.stop_speaking)
                except Exception as e:  # noqa: BLE001
                    log.exception("TTS/playback failed")
                    speak_errors.append(e)

        th = threading.Thread(target=speaker, name="vox-speaker", daemon=True)
        th.start()
        splitter = SentenceSplitter()
        reply = ""
        ok = True
        try:
            for chunk in self.brain.ask(text, lang, cancelled=self.stop_speaking.is_set):
                reply += chunk
                self.events.reply(reply, False)
                for s in splitter.feed(chunk):
                    sentences.put(s)
            for s in splitter.flush():
                sentences.put(s)
        except Exception as e:  # noqa: BLE001
            ok = False
            log.exception("Claude request failed")
            self.events.error(f"Claude: {e}")
            sentences.put(ERROR_LINES.get(lang, ERROR_LINES["en"]))
        sentences.put(None)
        th.join()
        self.events.reply(reply, True)
        log.info("claude: %s", reply)
        if speak_errors:
            self.events.error(f"Speech output failed: {speak_errors[0]}")
            return False
        return ok and not self.stop_speaking.is_set()
