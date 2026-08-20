"""Tempo, beat-grid phase and meter estimation."""

import numpy as np
import librosa


def estimate_tempo(audio, sr, onsets=None):
    """Estimate BPM: ``librosa.feature.tempo`` prior, refined against onsets."""
    prior = float(np.asarray(librosa.feature.tempo(y=audio, sr=sr)).ravel()[0])
    if onsets is not None and len(onsets) >= 4:
        return refine_tempo(onsets, prior)
    return prior


def refine_tempo(onsets, prior, tol=0.15):
    """Refine a tempo prior to the beat period that best aligns onsets to an
    eighth-note grid. Returns BPM."""
    onsets = np.asarray(onsets, dtype=float)
    prior_period = 60.0 / prior
    best_period, best_err = None, np.inf
    for period in np.arange(prior_period * (1 - tol),
                            prior_period * (1 + tol),
                            prior_period * 0.005):
        grid = period / 2.0
        err = float(np.mean((onsets - np.round(onsets / grid) * grid) ** 2))
        if err < best_err:
            best_err, best_period = err, period
    return 60.0 / best_period if best_period else prior


def detect_meter(onsets, bpm):
    """Detect (numerator, denominator) from note-onset accents.

    Handles simple meters (2/4, 3/4, 4/4, 5/4, ...) and compound (6/8, 9/8,
    12/8). Falls back to (4, 4).
    """
    onsets = np.asarray(onsets, dtype=float)
    if len(onsets) < 16:
        return 4, 4
    beat_len = 60.0 / bpm
    beat_idx = np.round((onsets - onsets[0]) / beat_len).astype(int)
    n_beats = int(beat_idx.max()) + 1
    accent = np.zeros(n_beats)
    for b in beat_idx:
        if 0 <= b < n_beats:
            accent[b] += 1
    x = accent - accent.mean()
    scores = {}
    for lag in range(2, 9):
        if lag > n_beats // 2:
            break
        a, b = x[:-lag], x[lag:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        scores[lag] = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    best_lag = max(scores, key=scores.get) if scores else 4
    # octave-error disambiguation: 8/4 is usually 4/4 doubled, 6/4 -> 3/4, etc.
    while (best_lag % 2 == 0 and (best_lag // 2) in scores
           and scores[best_lag // 2] > 0.8 * scores[best_lag]):
        best_lag //= 2
    # 复合拍（/8）不自动检测：曾按 onset 三连音占比判复合拍，对 tempo 敏感
    # （八分音符 0.25/0.75 会被误判为三连音 1/3/2/3），已弃用，复合拍靠 --time-sig 手动指定。
    return best_lag, 4


def detect_downbeats(audio, sr, bpm):
    """Detect downbeat times [s], beats-per-bar and tempo from mono audio.

    Uses ``madmom_infer`` (pure-numpy reimplementation of madmom). The beat
    tracker runs *free* (no tempo constraint) so it finds the true tempo --
    this also corrects AMT's occasional double/half-tempo errors (e.g. a piece
    at 79 bpm mis-detected as 158). Returns ``(downbeats, beats_per_bar,
    tempo)``; on failure returns ``(empty, None, None)`` and the caller falls
    back to its own tempo/meter estimates.

    NOTE: ``audio`` must be 44.1kHz -- madmom-infer's ``RNNDownBeatProcessor``
    hard-codes ``sample_rate=44100`` and does not resample.
    """
    try:
        from madmom_infer.features.downbeats import (
            DBNDownBeatTrackingProcessor, RNNDownBeatProcessor)
    except Exception:
        return np.empty(0), None, None
    try:
        act = np.asarray(RNNDownBeatProcessor()(audio))
        dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4, 6], fps=100)
        beats = np.asarray(dbn(act))
    except Exception:
        return np.empty(0), None, None
    if beats.size == 0:
        return np.empty(0), None, None
    downbeats = beats[beats[:, 1] == 1, 0]
    beats_per_bar = int(beats[:, 1].max())
    ibi = np.diff(beats[:, 0])
    ibi = ibi[ibi > 0]
    tempo = 60.0 / float(np.median(ibi)) if len(ibi) else None
    return downbeats, beats_per_bar, tempo
