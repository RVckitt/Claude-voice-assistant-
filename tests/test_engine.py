"""End-to-end engine tests with real VAD on synthetic speech (espeak-ng) and fake STT/Claude/TTS."""
import shutil
import subprocess
import tempfile
import time
import wave

import numpy as np
import pytest

from vox.audio import FRAME_SAMPLES, SAMPLE_RATE
from vox.config import Config, DEFAULT_TOML, tomllib
from vox.engine import Engine
from vox.stt import Transcript
from vox.vad import Segmenter
from vox.wake import WakeResult

pytestmark = pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng needed to synthesise speech")


def speech(text: str) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        subprocess.run(["espeak-ng", "-w", f.name, text], check=True)
        with wave.open(f.name) as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    t_new = np.arange(int(len(a) * SAMPLE_RATE / sr)) / SAMPLE_RATE
    return np.interp(t_new, np.arange(len(a)) / sr, a).astype(np.int16)


def silence(s: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0, 30, int(s * SAMPLE_RATE))).astype(np.int16)  # faint room noise


class FakeMic:
    def __init__(self, *pieces):
        audio = np.concatenate(pieces)
        n = len(audio) // FRAME_SAMPLES
        self.frames = [audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES] for i in range(n)]
        self.pos = 0

    def start(self): pass
    def stop(self): pass
    def flush(self): pass  # real mic drops echo here; the fake keeps the script intact

    def read(self, timeout=0.1):
        if self.pos < len(self.frames):
            self.pos += 1
            return self.frames[self.pos - 1]
        time.sleep(0.01)
        return None


class FakeWake:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def check(self, audio):
        self.calls += 1
        return self.results.pop(0) if self.results else WakeResult(False)


class FakeSTT:
    def __init__(self, texts):
        self.texts = list(texts)
        self.model = None
        self.durations = []

    def transcribe(self, audio, language=None, prompt=None):
        self.durations.append(len(audio) / SAMPLE_RATE)
        text, lang = self.texts.pop(0)
        return Transcript(text, lang, 1.0)


class FakeBrain:
    def __init__(self, replies, fail=False):
        self.replies = list(replies)
        self.asked = []
        self.fail = fail

    def ask(self, text, lang, cancelled=lambda: False):
        self.asked.append((text, lang))
        if self.fail:
            raise ConnectionError("network down")
        for word in self.replies.pop(0).split(" "):
            yield word + " "

    def reset(self): pass


class FakeTTS:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text, lang):
        self.spoken.append((text, lang))
        return np.zeros(100, dtype=np.float32), 22050


class FakePlayer:
    def __init__(self):
        self.plays = 0

    def play(self, audio, sr, stop=None):
        self.plays += 1
        return True


class Recorder:
    def __init__(self):
        self.states, self.users, self.replies, self.errors = [], [], [], []

    def state(self, s, d=""): self.states.append(s)
    def user(self, t): self.users.append(t)
    def reply(self, t, done):
        if done:
            self.replies.append(t)
    def error(self, m): self.errors.append(m)


def cfg(**audio):
    raw = tomllib.loads(DEFAULT_TOML)
    raw["audio"].update({"follow_up_s": 0.3, "command_timeout_s": 0.5, **audio})
    return Config(raw)


def run_engine(c, mic, wake, stt, brain, seconds=4.0):
    ev, tts, player = Recorder(), FakeTTS(), FakePlayer()
    eng = Engine(c, ev, mic=mic, player=player, stt=stt, wake=wake, brain=brain, tts=tts,
                 segmenter=Segmenter(2, 700, 20))
    eng.start()
    deadline = time.time() + seconds
    while time.time() < deadline and (mic.pos < len(mic.frames) or eng.state not in ("idle",)):
        time.sleep(0.05)
    time.sleep(0.3)
    eng.shutdown()
    eng.thread.join(2)
    return ev, tts, player


def test_vad_finds_single_utterance():
    mic = FakeMic(silence(0.5), speech("hey claude, what time is it"), silence(1.2))
    seg = Segmenter(2, 700, 20)
    audio = seg.next_utterance(mic.read, start_timeout=3)
    assert audio is not None
    dur = len(audio) / SAMPLE_RATE
    spoken = len(speech("hey claude, what time is it")) / SAMPLE_RATE
    assert spoken * 0.7 < dur < spoken + 0.9, (dur, spoken)
    assert seg.next_utterance(mic.read, start_timeout=0.3) is None   # only silence left


def test_one_breath_command():
    mic = FakeMic(silence(0.5), speech("hey claude what time is it"), silence(1.5))
    wake = FakeWake([WakeResult(True, has_more=True, heard="Hey Claude what time")])
    stt = FakeSTT([("Hey Claude, what time is it?", "en")])
    brain = FakeBrain(["It is five past three. Anything else?"])
    ev, tts, player = run_engine(cfg(), mic, wake, stt, brain)
    assert brain.asked == [("what time is it?", "en")]
    assert [t for t, _ in tts.spoken] == ["It is five past three.", "Anything else?"]
    assert ev.states[:2] == ["loading", "idle"] and "thinking" in ev.states and "speaking" in ev.states
    assert ev.states[-1] == "idle" and not ev.errors


def test_wake_then_pause_then_command_in_polish():
    mic = FakeMic(silence(0.4), speech("hey claude"), silence(1.0), speech("jaka jest jutro pogoda"), silence(1.2))
    wake = FakeWake([WakeResult(True, has_more=False, heard="Hey Claude")])
    stt = FakeSTT([("Jaka jest jutro pogoda?", "pl")])
    brain = FakeBrain(["Jutro będzie słonecznie i ciepło, około dwudziestu stopni."])
    ev, tts, player = run_engine(cfg(), mic, wake, stt, brain)
    assert brain.asked == [("Jaka jest jutro pogoda?", "pl")]
    assert tts.spoken[0][1] == "pl"
    assert "listening" in ev.states
    assert stt.durations[0] < 3.0          # only the command was transcribed, not the wake phrase


def test_follow_up_without_wake_word():
    mic = FakeMic(silence(0.4), speech("hey claude, tell me a joke"), silence(1.0),
                  speech("another one please"), silence(1.2))
    wake = FakeWake([WakeResult(True, has_more=True)])
    stt = FakeSTT([("Hey Claude, tell me a joke.", "en"), ("Another one, please.", "en")])
    brain = FakeBrain(["Why did the router blush? It saw the packet's headers.", "OK, one more: ping."])
    c = cfg(follow_up_s=2.0)
    ev, tts, player = run_engine(c, mic, wake, stt, brain, seconds=6)
    assert [a for a, _ in brain.asked] == ["tell me a joke.", "Another one, please."]
    assert wake.calls == 1                 # the follow-up did not need the wake word


def test_non_wake_speech_is_ignored():
    mic = FakeMic(silence(0.4), speech("I was talking to someone else"), silence(1.2))
    wake = FakeWake([WakeResult(False, heard="I was talking")])
    brain = FakeBrain([])
    ev, tts, player = run_engine(cfg(), mic, wake, FakeSTT([]), brain)
    assert wake.calls == 1 and brain.asked == [] and tts.spoken == []


def test_claude_failure_is_spoken_and_reported():
    mic = FakeMic(silence(0.4), speech("hey claude, hello"), silence(1.2))
    wake = FakeWake([WakeResult(True, has_more=True)])
    stt = FakeSTT([("Hej Claude, cześć", "pl")])
    ev, tts, player = run_engine(cfg(), mic, wake, stt, FakeBrain([], fail=True))
    assert ev.errors and "network down" in ev.errors[0]
    assert tts.spoken == [("Przepraszam, nie mogę teraz połączyć się z Claude.", "pl")]
    assert ev.states[-1] == "idle"


def test_typed_question_and_mute():
    mic = FakeMic(silence(0.2))
    ev, tts, player = Recorder(), FakeTTS(), FakePlayer()
    brain = FakeBrain(["Stolicą Australii jest Canberra."])
    eng = Engine(cfg(follow_up_s=0), ev, mic=mic, player=player, stt=FakeSTT([]), wake=FakeWake([]), brain=brain,
                 tts=tts, segmenter=Segmenter())
    eng.start()
    time.sleep(0.3)
    eng.ask("Jaka jest stolica Australii?")
    time.sleep(0.8)
    eng.set_muted(True)
    time.sleep(0.4)
    assert brain.asked == [("Jaka jest stolica Australii?", "pl")]
    assert tts.spoken == [("Stolicą Australii jest Canberra.", "pl")]
    assert ev.states[-1] == "muted"
    eng.shutdown()
    eng.thread.join(2)
    assert not eng.thread.is_alive()


def test_stop_interrupts_speaking():
    class SlowPlayer(FakePlayer):
        def play(self, audio, sr, stop=None):
            self.plays += 1
            for _ in range(100):
                if stop is not None and stop.is_set():
                    return False
                time.sleep(0.02)
            return True

    mic = FakeMic(silence(0.2))
    ev, tts, player = Recorder(), FakeTTS(), SlowPlayer()
    brain = FakeBrain(["Zdanie pierwsze jest długie. Zdanie drugie też jest długie. Trzecie zdanie."])
    eng = Engine(cfg(), ev, mic=mic, player=player, stt=FakeSTT([]), wake=FakeWake([]), brain=brain,
                 tts=tts, segmenter=Segmenter())
    eng.start()
    time.sleep(0.2)
    eng.ask("Powiedz coś długiego")
    time.sleep(0.5)
    t = time.time()
    eng.stop()
    while eng.state != "idle" and time.time() - t < 2:
        time.sleep(0.02)
    assert time.time() - t < 0.5 and player.plays == 1
    eng.shutdown()
