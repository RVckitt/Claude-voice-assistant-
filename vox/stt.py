"""Local speech-to-text with faster-whisper (Polish + English auto-detect)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str
    confidence: float


def load_whisper(model: str, device: str = "auto", compute_type: str = "auto"):
    """Loads a WhisperModel, falling back to CPU when CUDA libraries are missing."""
    from faster_whisper import WhisperModel

    attempts = []
    if device in ("auto", "cuda"):
        attempts.append(("cuda", "float16" if compute_type == "auto" else compute_type))
    if device in ("auto", "cpu"):
        attempts.append(("cpu", "int8" if compute_type == "auto" else compute_type))
    last = None
    for dev, ct in attempts:
        try:
            m = WhisperModel(model, device=dev, compute_type=ct)
            # warm-up: surfaces missing cuDNN/cuBLAS right away instead of mid-conversation
            list(m.transcribe(np.zeros(16000, dtype=np.float32), language="en", beam_size=1)[0])
            log.info("Whisper %s loaded on %s (%s)", model, dev, ct)
            return m
        except Exception as e:  # noqa: BLE001
            last = e
            log.info("Whisper %s on %s unavailable: %s", model, dev, e)
    raise RuntimeError(f"Could not load Whisper model {model!r}: {last}")


class Transcriber:
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "auto",
                 languages: list[str] | None = None, _model=None):
        self.languages = languages or ["pl", "en"]
        self.model = _model or load_whisper(model, device, compute_type)

    def _pick_language(self, audio: np.ndarray) -> tuple[str, float]:
        if len(self.languages) == 1:
            return self.languages[0], 1.0
        try:
            _, _, probs = self.model.detect_language(audio)
            allowed = [(lang, p) for lang, p in probs if lang in self.languages]
            if allowed:
                lang, p = max(allowed, key=lambda x: x[1])
                return lang, p
        except Exception as e:  # noqa: BLE001
            log.debug("language detection failed: %s", e)
        return self.languages[0], 0.0

    def transcribe(self, audio: np.ndarray, language: str | None = None, prompt: str | None = None) -> Transcript:
        lang, conf = (language, 1.0) if language else self._pick_language(audio)
        segments, _info = self.model.transcribe(
            audio, language=lang, beam_size=5, vad_filter=True,
            initial_prompt=prompt, condition_on_previous_text=False,
        )
        parts = [s.text for s in segments if s.no_speech_prob < 0.7]
        return Transcript(" ".join(p.strip() for p in parts).strip(), lang, conf)
