"""Command-line interface."""

import argparse
from pathlib import Path

from .pipeline import transcribe_to_score


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="audio2score",
        description="Transcribe piano audio to MIDI and MusicXML.",
    )
    p.add_argument("input", help="input audio file (wav/flac/mp3/...)")
    p.add_argument("-o", "--out-dir", default=None,
                   help="output directory (default: next to the input file)")
    p.add_argument("--midi-only", action="store_true", help="only write MIDI, skip MusicXML")
    p.add_argument("--cpu", action="store_true", help="force CPU inference")
    p.add_argument("--hand-split", type=int, default=60,
                   help="MIDI pitch that splits LH/RH (default 60 = C4)")
    p.add_argument("--tempo", type=float, default=None, help="force BPM (else auto-detect)")
    p.add_argument("--time-sig", type=str, default=None,
                   help='time signature, e.g. "3/4" or "6/8" (else auto-detect simple meters)')
    p.add_argument("--key", dest="key_spec", type=str, default=None,
                   help='key: int sharps/flats (e.g. 2) or a name (e.g. "D", "b")')
    args = p.parse_args(argv)

    inp = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else inp.parent
    device = "cpu" if args.cpu else None

    try:
        res = transcribe_to_score(
            str(inp), out_dir, midi_only=args.midi_only, device=device,
            hand_split=args.hand_split, tempo=args.tempo, time_sig=args.time_sig,
            key_spec=args.key_spec,
        )
    except FileNotFoundError as e:
        print(e)
        return 1
    for kind, path in res.items():
        if kind == "tempo":
            print(f"[tempo] {path:.1f} bpm")
        else:
            print(f"[{kind}] {path}")
    return 0
