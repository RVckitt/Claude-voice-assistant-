"""Local text-to-speech with Piper (one voice per language, downloaded on first use)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import VOICES_DIR

log = logging.getLogger(__name__)


def ensure_voice(name: str, directory: Path = VOICES_DIR) -> Path:
    path = directory / f"{name}.onnx"
    if not path.exists() or not (directory / f"{name}.onnx.json").exists():
        from piper.download_voices import download_voice
        directory.mkdir(parents=True, exist_ok=True)
        log.info("Downloading Piper voice %s …", name)
        download_voice(name, directory)
    return path


class Speaker:
    def __init__(self, voices: dict[str, str], speed: float = 1.0, voices_dir: Path = VOICES_DIR):
        self.voice_names = dict(voices)
        self.speed = speed
        self.dir = voices_dir
        self._loaded: dict[str, object] = {}

    def _voice(self, lang: str):
        name = self.voice_names.get(lang) or next(iter(self.voice_names.values()))
        if name not in self._loaded:
            from piper import PiperVoice
            self._loaded[name] = PiperVoice.load(ensure_voice(name, self.dir))
        return self._loaded[name]

    def preload(self):
        for lang in self.voice_names:
            self._voice(lang)

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        from piper import SynthesisConfig
        voice = self._voice(lang)
        cfg = SynthesisConfig(length_scale=1.0 / max(0.3, float(self.speed)))
        chunks = list(voice.synthesize(text, syn_config=cfg))
        if not chunks:
            return np.zeros(0, dtype=np.float32), 22050
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate
