from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .base import LoaderBase


@dataclass
class AudioRms:
    times: np.ndarray
    values: np.ndarray


@dataclass
class AudioSpectrogram:
    times: np.ndarray
    frequencies: np.ndarray
    magnitude: np.ndarray


@dataclass
class AudioData:
    sample_rate: int
    samples: np.ndarray
    rms: AudioRms
    spectrogram: AudioSpectrogram
    duration: float


class AudioLoader(LoaderBase):
    def __init__(
        self,
        wav_path: Path,
        rms_window: float = 0.02,
        spectrogram_window: float = 0.02,
        spectrogram_step: float = 0.01,
    ):
        super().__init__(wav_path)
        self.rms_window = rms_window
        self.spectrogram_window = spectrogram_window
        self.spectrogram_step = spectrogram_step

    def load(self) -> AudioData:
        samples, sample_rate = self._read_wav()
        duration = len(samples) / sample_rate if sample_rate else 0.0
        rms = self._compute_rms(samples, sample_rate)
        spectrogram = self._compute_spectrogram(samples, sample_rate)
        return AudioData(
            sample_rate=sample_rate,
            samples=samples,
            rms=rms,
            spectrogram=spectrogram,
            duration=duration,
        )

    def _read_wav(self) -> Tuple[np.ndarray, int]:
        if not self.source.exists():
            return np.array([]), 0

        with wave.open(str(self.source), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            samp_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if samp_width == 3:
            raw_array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            padded = np.hstack(
                (raw_array, np.zeros((raw_array.shape[0], 1), dtype=np.uint8))
            )
            data32 = padded.view(np.int32).flatten()
            audio = data32.astype(np.float32)
        else:
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(samp_width)
            if dtype is None:
                raise ValueError(f"Unsupported sample width: {samp_width}")
            audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return audio, sample_rate

    def _compute_rms(self, samples: np.ndarray, sample_rate: int) -> AudioRms:
        if len(samples) == 0 or sample_rate == 0:
            return AudioRms(np.array([]), np.array([]))

        window_samples = max(1, int(self.rms_window * sample_rate))
        n_windows = len(samples) // window_samples
        rms_values = []
        times = []
        for idx in range(n_windows):
            start = idx * window_samples
            segment = samples[start : start + window_samples]
            if len(segment) == 0:
                break
            rms_values.append(np.sqrt(np.mean(np.square(segment))))
            times.append((start + len(segment) / 2) / sample_rate)
        return AudioRms(times=np.array(times), values=np.array(rms_values))

    def _compute_spectrogram(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> AudioSpectrogram:
        if len(samples) == 0 or sample_rate == 0:
            return AudioSpectrogram(np.array([]), np.array([]), np.array([[]]))

        window_size = max(1, int(self.spectrogram_window * sample_rate))
        step_size = max(1, int(self.spectrogram_step * sample_rate))
        window = np.hanning(window_size)

        segments = []
        times = []
        for start in range(0, len(samples) - window_size + 1, step_size):
            slice_ = samples[start : start + window_size]
            windowed = slice_ * window
            spectrum = np.fft.rfft(windowed)
            segments.append(np.abs(spectrum))
            times.append((start + window_size / 2) / sample_rate)

        magnitude = np.stack(segments, axis=1) if segments else np.array([[]])
        freqs = np.fft.rfftfreq(window_size, d=1.0 / sample_rate)

        return AudioSpectrogram(
            times=np.array(times),
            frequencies=freqs,
            magnitude=magnitude,
        )
