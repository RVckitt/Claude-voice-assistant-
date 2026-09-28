"""Wake-word detection.

WhisperWake      – default. Every short burst of speech is transcribed by a small local Whisper
                   model and matched against "hey claude" (fuzzy, so "hej klaud" works too).
                   Nothing leaves the computer; only a command after the wake phrase goes to Claude.
OpenWakeWordWake – optional, much lighter on CPU, but needs a trained model file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .textutil import split_wake

log = logging.getLogger(__name__)

WAKE_WINDOW_S = 3.0  # only the first seconds of an utterance are checked for the phrase


@dataclass
class WakeResult:
    matched: bool
    has_more: bool = False      # did the user keep talking after the phrase ("hey claude, what's…")?
    heard: str = ""


class WhisperWake:
    def __init__(self, phrase: str, aliases: list[str], model: str = "base", sensitivity: float = 0.78,
                 device: str = "auto", _model=None):
        from .stt import load_whisper
        self.phrases = [phrase, *aliases]
        self.threshold = sensitivity
        self.model = _model or load_whisper(model, device, "auto")
        self.prompt = phrase.strip().capitalize() + "."   # biases Whisper towards spelling it right

    def check(self, audio: np.ndarray) -> WakeResult:
        head = audio[: int(WAKE_WINDOW_S * SAMPLE_RATE)]
        # English decoding recognises "hey claude" best, even in a Polish sentence
        segs, _ = self.model.transcribe(head, language="en", beam_size=1, initial_prompt=self.prompt,
                                        condition_on_previous_text=False, without_timestamps=True)
        heard = " ".join(s.text.strip() for s in segs).strip()
        matched, remainder = split_wake(heard, self.phrases, self.threshold)
        if matched:
            log.info("wake phrase heard: %r", heard)
        has_more = bool(remainder) or len(audio) > len(head)
        return WakeResult(matched, has_more and matched, heard)


class OpenWakeWordWake:
    """Streams frames through an openWakeWord model. Use via `feed()` instead of `check()`."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        from pathlib import Path

        from openwakeword.model import Model
        path = str(Path(model_path).expanduser())
        self.model = Model(wakeword_models=[path], inference_framework="onnx" if path.endswith(".onnx") else "tflite")
        self.threshold = threshold
        self._buf = np.zeros(0, dtype=np.int16)

    def feed(self, frame: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, frame])
        hit = False
        while len(self._buf) >= 1280:  # 80 ms chunks
            chunk, self._buf = self._buf[:1280], self._buf[1280:]
            scores = self.model.predict(chunk)
            if any(v >= self.threshold for v in scores.values()):
                hit = True
        if hit:
            self.model.reset()
        return hit
