"""Cuts the microphone stream into utterances using WebRTC voice activity detection."""
from __future__ import annotations

import collections
import time
from typing import Callable

import numpy as np

from .audio import FRAME_MS, SAMPLE_RATE


class Segmenter:
    def __init__(self, aggressiveness: int = 2, silence_ms: int = 900, max_utterance_s: float = 25,
                 min_speech_ms: int = 240, preroll_ms: int = 300, min_rms: float = 60):
        import webrtcvad
        self.vad = webrtcvad.Vad(int(aggressiveness))
        self.silence_frames = max(1, silence_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.start_window = 8          # frames looked at to decide speech started
        self.start_ratio = 0.6         # …of which this share must be voiced
        self.preroll = preroll_ms // FRAME_MS
        self.min_rms = min_rms          # energy gate: WebRTC VAD alone calls steady hiss "speech"

    def is_speech(self, frame: np.ndarray) -> bool:
        if np.sqrt(np.mean(frame.astype(np.float32) ** 2)) < self.min_rms:
            return False
        return self.vad.is_speech(frame.tobytes(), SAMPLE_RATE)

    def next_utterance(self, read: Callable[[], np.ndarray | None],
                       abort: Callable[[], bool] = lambda: False,
                       start_timeout: float | None = None) -> np.ndarray | None:
        """Blocks until one utterance was spoken. Returns float32 audio in [-1, 1].

        read()        – returns the next 30 ms int16 frame, or None on timeout
        abort()       – polled regularly; return True to give up (returns None)
        start_timeout – give up if no speech *starts* within this many seconds
        """
        ring: collections.deque = collections.deque(maxlen=max(self.start_window, self.preroll))
        voiced: collections.deque = collections.deque(maxlen=self.start_window)
        t0 = time.monotonic()
        frames: list[np.ndarray] = []
        triggered = False
        silence = 0
        speech_frames = 0

        while True:
            if abort():
                return None
            frame = read()
            if frame is None:
                if not triggered and start_timeout is not None and time.monotonic() - t0 > start_timeout:
                    return None
                continue
            speech = self.is_speech(frame)

            if not triggered:
                ring.append(frame)
                voiced.append(speech)
                if len(voiced) == voiced.maxlen and sum(voiced) >= self.start_ratio * voiced.maxlen:
                    triggered = True
                    frames = list(ring)[-max(self.preroll, self.start_window):]
                    speech_frames = sum(voiced)
                    silence = 0
                elif start_timeout is not None and time.monotonic() - t0 > start_timeout:
                    return None
                continue

            frames.append(frame)
            if speech:
                speech_frames += 1
                silence = 0
            else:
                silence += 1
            if silence >= self.silence_frames or len(frames) >= self.max_frames:
                if speech_frames < self.min_speech_frames:   # a click or cough – start over
                    triggered, frames, silence, speech_frames = False, [], 0, 0
                    ring.clear()
                    voiced.clear()
                    continue
                trim = silence if silence < len(frames) else 0
                audio = np.concatenate(frames[:len(frames) - trim + 3] if trim else frames)
                return audio.astype(np.float32) / 32768.0
