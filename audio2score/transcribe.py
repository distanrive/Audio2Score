"""Piano audio -> MIDI transcription (ByteDance high-resolution AMT).

Unlike the upstream package, this writes a *clean* MIDI:
  * pads leading silence so the first note is not dropped,
  * embeds the real detected tempo (not a hard-coded 120 bpm),
  * clips note offsets to the actual audio length (no runaway long notes).
"""

from pathlib import Path

import numpy as np
import pretty_midi
import torch

from piano_transcription_inference import PianoTranscription

from . import audio as audio_utils
from . import tempo as tempo_utils

# The path the package (and pianotrans) expect for the checkpoint.
DEFAULT_CHECKPOINT = (
    Path.home() / "piano_transcription_inference_data" / "note_F1=0.9677_pedal_F1=0.9186.pth"
)
CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)

LEADING_PAD_SEC = 1.0   # silence prepended to avoid first-note drop
MIN_NOTE_SEC = 0.02


def auto_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PianoAMT:
    """Thin wrapper around ByteDance's piano transcription model."""

    def __init__(self, device=None, checkpoint_path=None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self._check_checkpoint()
        self.device = torch.device(device) if device is not None else auto_device()
        self._model = PianoTranscription(
            device=self.device, checkpoint_path=str(self.checkpoint_path)
        )

    def _check_checkpoint(self):
        if not self.checkpoint_path.exists() or self.checkpoint_path.stat().st_size < 1.6e8:
            raise FileNotFoundError(
                "Model checkpoint missing or incomplete:\n"
                f"  {self.checkpoint_path}\n"
                f"Download it (~172 MB) from:\n  {CHECKPOINT_URL}\n"
                "and save it to the path above (see README, section 4.1)."
            )

    def transcribe(self, audio, sr=16000, midi_path=None, tempo=None):
        """Transcribe mono audio samples (any sample rate).

        Returns a dict: ``{'note_events', 'pedal_events', 'tempo'}``. Writes a
        clean MIDI to ``midi_path`` if given.
        """
        y = audio_utils.resample(audio, sr)
        dur_sec = len(y) / 16000.0

        # 1) prepend silence -> gives the model context at t=0 (fixes first-note drop)
        pad = int(LEADING_PAD_SEC * 16000)
        y_pad = np.concatenate([np.zeros(pad, dtype=np.float32), y])

        # 2) run the model (do NOT let it write the MIDI; we write it ourselves)
        result = self._model.transcribe(y_pad, midi_path=None)
        note_events = result.get("est_note_events") or []
        pedal_events = result.get("est_pedal_events") or []

        # 3) shift times back by the padding and clip to the real audio length
        note_events = _shift_clip_notes(note_events, LEADING_PAD_SEC, dur_sec)
        pedal_events = _shift_clip_pedals(pedal_events, LEADING_PAD_SEC, dur_sec)

        # 4) detect tempo (from the un-padded audio + final onsets)
        onsets = [e["onset_time"] for e in note_events]
        bpm = tempo if tempo is not None else tempo_utils.estimate_tempo(y, 16000, onsets)

        if midi_path is not None:
            _write_midi(note_events, pedal_events, midi_path, bpm)

        return {"note_events": note_events, "pedal_events": pedal_events, "tempo": bpm}

    def transcribe_file(self, audio_path, midi_path, tempo=None):
        y, sr = audio_utils.load_audio(audio_path)
        return self.transcribe(y, sr=sr, midi_path=midi_path, tempo=tempo)


def _shift_clip_notes(note_events, pad_sec, dur_sec):
    out = []
    for e in note_events:
        s = e["onset_time"] - pad_sec
        o = e["offset_time"] - pad_sec
        if o <= 0.0:            # entirely inside the padding
            continue
        s = max(0.0, s)
        o = min(o, dur_sec)     # clip offset to real audio length
        if o - s < MIN_NOTE_SEC:
            o = min(s + MIN_NOTE_SEC, dur_sec)
        out.append({"onset_time": s, "offset_time": o,
                    "midi_note": int(e["midi_note"]), "velocity": int(e["velocity"])})
    return out


def _shift_clip_pedals(pedal_events, pad_sec, dur_sec):
    out = []
    for e in pedal_events:
        s = e["onset_time"] - pad_sec
        o = e["offset_time"] - pad_sec
        if o <= 0.0:
            continue
        s = max(0.0, s)
        o = min(o, dur_sec)
        out.append({"onset_time": s, "offset_time": o})
    return out


def _write_midi(note_events, pedal_events, midi_path, bpm):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    inst = pretty_midi.Instrument(program=0, name="Piano")
    for e in note_events:
        inst.notes.append(pretty_midi.Note(
            velocity=e["velocity"], pitch=e["midi_note"],
            start=e["onset_time"], end=e["offset_time"],
        ))
    for e in pedal_events:
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=e["onset_time"]))
        inst.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=e["offset_time"]))
    pm.instruments.append(inst)
    pm.write(midi_path)
