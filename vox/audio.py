"""Microphone input, speaker output and UI sounds."""
from __future__ import annotations

import logging
import queue
import threading

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples – what webrtcvad expects


def _sd():
    import sounddevice as sd  # imported lazily so tests run without PortAudio
    return sd


def resolve_device(spec, kind: str):
    """'' -> default, '3' -> index 3, 'USB' -> first device whose name contains 'USB'."""
    if spec in ("", None):
        return None
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    sd = _sd()
    for i, dev in enumerate(sd.query_devices()):
        channels = dev["max_input_channels"] if kind == "input" else dev["max_output_channels"]
        if channels > 0 and str(spec).lower() in dev["name"].lower():
            return i
    log.warning("Audio %s device %r not found, using default", kind, spec)
    return None


def list_devices() -> str:
    return str(_sd().query_devices())


class Microphone:
    """Continuously captures 16 kHz mono int16 audio into a queue of 30 ms frames."""

    def __init__(self, device=None):
        self.device = resolve_device(device, "input")
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=2000)
        self._stream = None
        self._buf = np.zeros(0, dtype=np.int16)

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("mic status: %s", status)
        data = np.concatenate([self._buf, indata[:, 0].copy()])
        n = len(data) // FRAME_SAMPLES
        for i in range(n):
            try:
                self.frames.put_nowait(data[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
            except queue.Full:  # consumer is busy (e.g. speaking) – drop oldest
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
        self._buf = data[n * FRAME_SAMPLES:]

    def start(self):
        sd = _sd()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=FRAME_SAMPLES, device=self.device, callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def flush(self):
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout: float = 0.1) -> np.ndarray | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None


class Player:
    """Plays float32 audio; can be interrupted from another thread."""

    def __init__(self, device=None):
        self.device = resolve_device(device, "output")
        self._lock = threading.Lock()

    def play(self, audio: np.ndarray, sample_rate: int, stop: threading.Event | None = None) -> bool:
        """Blocks until finished. Returns False if interrupted."""
        sd = _sd()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
        block = 2048
        with self._lock, sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                         device=self.device) as out:
            for i in range(0, len(audio), block):
                if stop is not None and stop.is_set():
                    return False
                out.write(audio[i:i + block])
        return True


def tone(freqs: list[float], ms: int = 90, sr: int = 22050, volume: float = 0.25) -> np.ndarray:
    """Short, soft sequence of sine tones (used for wake / cancel sounds)."""
    parts = []
    for f in freqs:
        t = np.arange(int(sr * ms / 1000)) / sr
        env = np.minimum(1, np.minimum(t, t[::-1]) * 200)  # 5 ms fade in/out, no clicks
        parts.append(np.sin(2 * np.pi * f * t) * env * volume)
    return np.concatenate(parts).astype(np.float32)


WAKE_SOUND = tone([660, 990])
DONE_SOUND = tone([990, 660])
ERROR_SOUND = tone([330, 220], ms=140)
