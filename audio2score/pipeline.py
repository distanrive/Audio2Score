"""End-to-end pipeline: audio -> MIDI -> MusicXML."""

from pathlib import Path

from . import audio as audio_utils
from . import tempo as tempo_utils
from .transcribe import PianoAMT
from .notation import midi_to_musicxml


def transcribe_to_score(audio_path, out_dir, *, midi_only=False, musicxml_only=False,
                        device=None, hand_split=60, tempo=None, time_sig=None, key_spec=None):
    """Transcribe ``audio_path`` to MIDI and/or MusicXML.

    By default both files are kept. ``midi_only=True`` skips MusicXML;
    ``musicxml_only=True`` keeps only MusicXML (the intermediate MIDI is
    deleted). Returns a dict of generated paths plus ``tempo``.
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    midi_path = out_dir / f"{stem}.mid"

    amt = PianoAMT(device=device)
    res = amt.transcribe_file(str(audio_path), str(midi_path), tempo=tempo)
    bpm = res["tempo"]

    result = {"tempo": bpm}
    if not musicxml_only:
        result["midi"] = str(midi_path)
    if not midi_only:
        y, sr = audio_utils.load_audio(str(audio_path))
        # downbeat detection needs 44.1kHz (madmom-infer hard-codes it)
        y44, _ = audio_utils.load_audio(str(audio_path), target_sr=44100)
        downbeats, beats_per_bar, madmom_tempo = tempo_utils.detect_downbeats(y44, 44100, bpm)
        # madmom's beat tracker is more reliable than AMT/librosa; use its tempo
        # to correct double/half-tempo mis-detections (e.g. 79 bpm -> 158).
        if madmom_tempo is not None:
            bpm = float(madmom_tempo)
            result["tempo"] = bpm
        xml_path = out_dir / f"{stem}.musicxml"
        midi_to_musicxml(str(midi_path), str(xml_path), audio=y, sr=sr,
                         hand_split=hand_split, tempo=bpm, time_sig=time_sig,
                         key_spec=key_spec, downbeats=downbeats,
                         beats_per_bar=beats_per_bar)
        result["musicxml"] = str(xml_path)
        if musicxml_only:
            midi_path.unlink(missing_ok=True)
    return result
