"""Audio loading and resampling helpers."""

import numpy as np
import librosa

# The ByteDance AMT model operates at 16 kHz mono.
TARGET_SR = 16000


def load_audio(path, target_sr=TARGET_SR):
    """Load an audio file as mono float32, resampled to ``target_sr``.

    Returns ``(samples, sample_rate)``.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), sr


def resample(audio, orig_sr, target_sr=TARGET_SR):
    """Resample a mono signal to ``target_sr``."""
    if orig_sr == target_sr:
        return np.asarray(audio, dtype=np.float32)
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
