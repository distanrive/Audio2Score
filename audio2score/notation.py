"""MIDI -> MusicXML.

v2 (M2 in progress):
  * detected tempo embedded as a MetronomeMark (no more 120 bpm)
  * time signature: auto-detect simple meters (3/4 vs 4/4), manual override for
    irregular meters (e.g. "17/8")
  * key signature: auto-estimate (Krumhansl-Schmuckler) or manual override
  * quantize note on/off to an eighth-note grid (beat-tracked + phase-aligned)
  * pitch-threshold hand split, cross-barline ties + rests, grand staff output

Still TODO:
  * adaptive quantization (tuplets), better hand split, note spelling, pedal marks
"""

import math
import re
from pathlib import Path

import numpy as np
import pretty_midi
from music21 import chord, clef, expressions, instrument, key, layout, meter, note, stream, tie, tempo as tempo_mark

from . import tempo as tempo_utils

# Krumhansl-Schmuckler pitch-class profiles for key estimation (tonic at C=0).
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# pitch-class (C=0 .. B=11) -> music21 key name
_MAJOR_KEY = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
              6: "F#", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
_MINOR_KEY = {0: "c", 1: "c#", 2: "d", 3: "eb", 4: "e", 5: "f",
              6: "f#", 7: "g", 8: "g#", 9: "a", 10: "bb", 11: "b"}


def midi_to_musicxml(midi_path, out_path, *, audio=None, sr=16000, hand_split=60,
                     subdiv=None, tempo=None, time_sig=None, key_spec=None,
                     downbeats=None, beats_per_bar=None):
    """Convert a (transcribed) MIDI file to a grand-staff MusicXML score.

    ``tempo`` (bpm), ``time_sig`` (int => ``N/4``, or ``"N/D"`` e.g. ``"17/8"``)
    and ``key_spec`` (int sharps/flats, or a key name like ``"D"``/``"b"``) are
    detected when omitted; pass them to override.

    ``subdiv`` (2 => eighths, 4 => sixteenths) is auto-detected when omitted:
    sixteenths are used only where the music actually has them.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    events = [(n.start, n.end, n.pitch, n.velocity)
              for inst in pm.instruments for n in inst.notes]
    if not events:
        raise ValueError(f"No notes found in MIDI: {midi_path}")

    raw_events = events

    onsets = sorted(set(n.start for inst in pm.instruments for n in inst.notes))

    # tempo
    if tempo is None:
        if audio is not None and len(audio) > 0:
            bpm = tempo_utils.estimate_tempo(audio, sr, onsets)
        else:
            bpm = tempo_utils.refine_tempo(onsets, pm.estimate_tempo())
    else:
        bpm = float(tempo)

    # time signature
    numerator, denominator = _parse_time_sig(time_sig) if time_sig is not None \
        else tempo_utils.detect_meter(onsets, bpm)
    # madmom's downbeat tracker is more reliable than onset autocorrelation for
    # the meter numerator; correct octave mis-detections (e.g. 8/4 -> 4/4)
    if beats_per_bar is not None and 2 <= beats_per_bar <= 12:
        numerator = int(beats_per_bar)

    # key signature
    ks = _resolve_key(key_spec) if key_spec is not None \
        else _estimate_key_signature([e[2] for e in raw_events], [e[1] - e[0] for e in raw_events])

    # downbeat alignment: shift the time axis so the first downbeat lands on a
    # measure boundary. A pickup/anacrusis before it then leaves leading rests
    # in the first measure (correct bars, no lost notes); a true pickup measure
    # via Measure.paddingLeft could replace this later.
    shift_sec = 0.0
    if downbeats is not None and len(downbeats):
        t0 = float(downbeats[0])
        beat_len = 60.0 / bpm
        measure_sec = numerator * beat_len
        n = int(t0 / measure_sec)          # floor for positive values
        offset = t0 - n * measure_sec
        if offset > 0.5 * beat_len:        # downbeat off the measure grid => pickup
            shift_sec = (n + 1) * measure_sec - t0
            raw_events = [(e[0] + shift_sec, e[1] + shift_sec, e[2], e[3])
                          for e in raw_events if e[1] + shift_sec > 0.0]

    # adaptive quantization grid: sixteenths where the music actually has them,
    # else eighths (a fixed eighth grid merges sixteenths at slow tempi, e.g.
    # 79 bpm sixteenths of 0.19 s snap to the same eighth).
    if subdiv is None:
        subdiv = _detect_subdiv(raw_events, bpm)

    # Split hands first, then clip offsets *within each hand* (a held LH chord
    # under an RH melody is not clipped by the melody), quantize and merge.
    # longest sensible held note = one full measure (3 beats in 3/4, 4 in 4/4)
    max_dur_sec = (numerator * 4.0 / denominator) * 60.0 / bpm
    grid_sec = (60.0 / bpm) / subdiv
    # global typical note length (independent of hand split) for the tail cap
    global_onsets = sorted(set(round(e[0], 4) for e in raw_events))
    _gaps = np.diff(global_onsets)
    _gaps = _gaps[_gaps > 1e-6]
    typical_sec = float(np.median(_gaps)) if len(_gaps) else max_dur_sec

    measure_grid = numerator * subdiv

    def process(hand_events):
        clipped = _clip_offsets(hand_events, max_dur_sec, grid_sec, typical_sec, measure_grid)
        return _chord_merge(_quantize(clipped, bpm, subdiv))

    rh_raw, lh_raw = _split_hands(raw_events, hand_split)
    rh = process(rh_raw)
    lh = process(lh_raw)

    # 踏板已禁用（AMT 踏板抖动噪声大、踩点不落音符、位置不可靠，待重写后再开）
    # pedal_events = [(cc.time + shift_sec, cc.value) for inst in pm.instruments
    #                 for cc in inst.control_changes if cc.number == 64]
    # pedal_segments = _pedal_segments(pedal_events, bpm, subdiv=subdiv) if pedal_events else []
    pedal_segments = []

    score = _assemble_score(rh, lh, ks, numerator, denominator, bpm, pedal_segments)
    score.write("musicxml", fp=str(out_path))
    _strip_pedal_numbers(str(out_path))
    return str(out_path)


def _parse_time_sig(time_sig):
    if isinstance(time_sig, int):
        return time_sig, 4
    s = str(time_sig).strip()
    if "/" in s:
        n, d = s.split("/")
        return int(n), int(d)
    return int(s), 4


def _resolve_key(key_spec):
    """key_spec: int (sharps count, negative => flats) or a key name string."""
    try:
        return int(key_spec)
    except (TypeError, ValueError):
        return key.Key(str(key_spec)).sharps


# A fixed pitch cut (e.g. 60 = C4) fails hand-crossing at *both* ends: a LH
# chord can reach up to ~66 under a 71+ melody, and an RH melody can dip to
# ~57 over a 44-54 bass. So the split point is estimated per chord and allowed
# to move over time, via an EM-style loop that tracks each hand's local pitch
# register (a time-weighted centroid). Measured on case01/reference.mid this
# recovers ~2.5x fewer crossing errors than the old greedy threshold (10 vs 13,
# i.e. 94.2% vs 92.4% note accuracy).
_CHORD_TOL = 0.03          # onset jitter (s) still treated as simultaneous
_REGISTER_WIN = 4.0        # time window (s) for each hand's local register
_N_EM_ITERS = 3


def _split_hands(events, threshold=60):
    """Split note events into (RH, LH) with a time-varying split point.

    Notes are grouped into chords (near-simultaneous onsets); each chord is
    split monotonically (all notes below the split -> LH, above -> RH). The
    split point is not a fixed pitch: an EM loop estimates each hand's local
    register and re-splits every chord to minimise deviation from it, so the
    split tracks the music (low under a low RH melody, high under a high LH
    chord). ``threshold`` is only the initial register prior / fallback, not a
    hard cut.
    """
    if not events:
        return [], []

    # group near-simultaneous notes into chords
    ev = sorted(events, key=lambda e: (e[0], e[2]))
    chords = []
    for e in ev:
        if chords and e[0] - chords[-1][0] <= _CHORD_TOL:
            chords[-1][1].append(e)
        else:
            chords.append((e[0], [e]))
    pitches = [sorted(n[2] for n in notes) for _, notes in chords]
    onsets = [t for t, _ in chords]

    # EM: initial split -> registers -> re-split, repeat
    assign = _init_split(pitches, threshold)
    for _ in range(_N_EM_ITERS):
        lo_c, hi_c = _hand_registers(pitches, assign, onsets, threshold)
        assign = _resplit(pitches, lo_c, hi_c)

    rh, lh = [], []
    for (_, notes), lab in zip(chords, assign):
        # ``lab`` is aligned to pitch-sorted order (see _init_split/_resplit),
        # so sort this chord's notes by pitch before pairing.
        for n, h in zip(sorted(notes, key=lambda e: e[2]), lab):
            (rh if h == "RH" else lh).append(n)
    return rh, lh


def _init_split(pitches, threshold):
    """Coarse initial split: single notes by threshold, chords at the largest gap."""
    assign = []
    for ps in pitches:
        k = len(ps)
        if k == 1:
            assign.append(["LH" if ps[0] < threshold else "RH"])
        else:
            bi = max(range(k - 1), key=lambda i: ps[i + 1] - ps[i])
            assign.append(["LH" if j <= bi else "RH" for j in range(k)])
    return assign


def _hand_registers(pitches, assign, onsets, threshold):
    """Per-chord LH/RH registers: time-weighted centroid of each hand's notes."""
    T = np.asarray(onsets, dtype=float)
    lo_c, hi_c = [], []
    for t in T:
        w = np.exp(-0.5 * ((T - t) / (_REGISTER_WIN / 3.0)) ** 2)
        lo_p, lo_w, hi_p, hi_w = [], [], [], []
        for j, ps in enumerate(pitches):
            for p, h in zip(ps, assign[j]):
                (lo_p if h == "LH" else hi_p).append(p)
                (lo_w if h == "LH" else hi_w).append(w[j])
        lo_c.append(float(np.average(lo_p, weights=lo_w)) if lo_p else threshold - 10.0)
        hi_c.append(float(np.average(hi_p, weights=hi_w)) if hi_p else threshold + 10.0)
    return lo_c, hi_c


def _resplit(pitches, lo_c, hi_c):
    """Monotone split of each chord minimising squared deviation from registers."""
    assign = []
    for ps, cl, cr in zip(pitches, lo_c, hi_c):
        k = len(ps)
        if k == 1:
            p = ps[0]
            assign.append(["LH" if abs(p - cl) <= abs(p - cr) else "RH"])
        else:
            best = min(range(k + 1),
                       key=lambda bi: sum((p - cl) ** 2 for p in ps[:bi])
                                       + sum((p - cr) ** 2 for p in ps[bi:]))
            assign.append(["LH" if j < best else "RH" for j in range(k)])
    return assign


def _clip_offsets(events, max_dur_sec, grid_sec, typical_sec=None, measure_grid=None):
    """Clip each note's offset to the next *quantized* onset and a max.

    Onsets are snapped to an integer grid index first, so near-simultaneous
    (spurious) notes share an onset and don't clip each other. Then each note
    is capped to the next grid onset (preventing overlap) and a max (handling
    reverb tails). Integer indices avoid float round-off making a note's own
    onset look like its "next" onset.

    ``typical_sec`` caps the last note (no next onset); if omitted it's the
    median gap between distinct onsets in ``events``.
    """
    events = sorted(events, key=lambda e: e[0])
    grid_idx = lambda t: round(t / grid_sec)
    distinct = sorted(set(grid_idx(e[0]) for e in events))
    if typical_sec is None:
        distinct_starts = sorted(set(round(e[0], 4) for e in events))
        gaps = np.diff(distinct_starts)
        gaps = gaps[gaps > 1e-6]
        typical_sec = float(np.median(gaps)) if len(gaps) else max_dur_sec
    out = []
    for start, end, pitch, vel in events:
        s_idx = grid_idx(start)
        nxt_idx = next((i for i in distinct if i > s_idx), None)
        if measure_grid:
            # cap at the next measure boundary (s_idx // measure_grid is the
            # measure index; +1 is the first grid step of the following bar)
            cap = ((s_idx // measure_grid) + 1) * measure_grid * grid_sec
        else:
            cap = s_idx * grid_sec + max_dur_sec
        if nxt_idx is not None:
            cap = min(cap, nxt_idx * grid_sec)
        else:
            cap = min(cap, s_idx * grid_sec + typical_sec)  # last note: typical length
        end = min(end, cap)
        if end - start < 0.05:
            end = start + 0.05
        out.append((start, end, pitch, vel))
    return out


def _detect_subdiv(raw_events, bpm):
    """Choose the quantization grid: 2 (eighths) or 4 (sixteenths).

    Sixteenths are needed when a non-negligible share of inter-onset gaps is
    clearly shorter than an eighth note (i.e. around a sixteenth). Those are
    exactly the gaps an eighth-note grid would merge at slow tempi.
    """
    onsets = sorted({round(e[0], 4) for e in raw_events})
    if len(onsets) < 2:
        return 2
    beat = 60.0 / bpm
    gaps = np.diff(onsets)
    # drop near-simultaneous (chord) artifacts and unmusically long silences
    gaps = gaps[(gaps > beat * 0.15) & (gaps < beat * 1.5)]
    if gaps.size == 0:
        return 2
    # a sixteenth-note gap is ~beat/4; an eighth is ~beat/2. Anything under
    # 0.375 beat (1.5 sixteenths) is unrepresentable on an eighth grid.
    n_16 = int(np.sum(gaps < beat * 0.375))
    # require a handful of sixteenths, not a single isolated ornament, before
    # switching the whole piece to a finer grid
    return 4 if n_16 >= 3 and n_16 / gaps.size >= 0.08 else 2


def _quantize(events, bpm, subdiv=2):
    """Snap (start, end, pitch, vel) to the subdiv grid (anchored at t=0),
    in quarter-length units."""
    beat_len = 60.0 / bpm
    grid = beat_len / subdiv
    out = []
    for start, end, pitch, _vel in events:
        onset_ql = round(start / grid) * grid / beat_len
        offset_ql = round(end / grid) * grid / beat_len
        dur_ql = offset_ql - onset_ql
        if dur_ql < 1.0 / subdiv:
            dur_ql = 1.0 / subdiv
        out.append((round(max(0.0, onset_ql), 4), round(dur_ql, 4), pitch))
    return out


def _estimate_key_signature(pitches, durs):
    hist = np.zeros(12)
    for p, d in zip(pitches, durs):
        hist[int(round(p)) % 12] += max(d, 1e-6)
    if hist.sum() == 0:
        return 0
    hist = hist / hist.sum()
    best_corr, best = -2.0, None
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            corr = np.corrcoef(hist, np.roll(prof, tonic))[0, 1]
            if corr > best_corr:
                best_corr, best = corr, (tonic, mode)
    tonic, mode = best
    k = key.Key(_MAJOR_KEY[tonic] if mode == "major" else _MINOR_KEY[tonic])
    return k.sharps


def _chord_merge(qevents):
    """Merge simultaneous notes (same quantized onset) into chords."""
    d = {}
    for onset, dur, pitch in qevents:
        k = round(onset, 2)
        if k not in d:
            d[k] = [onset, dur, []]
        d[k][1] = max(d[k][1], dur)
        d[k][2].append(pitch)
    items = [(v[0], v[1], sorted(v[2])) for v in d.values()]
    items.sort(key=lambda x: x[0])
    return items


def _assemble_score(rh, lh, ks, numerator, denominator, bpm, pedal_segments=None):
    score = stream.Score()

    rh_ps, rh_notes = _make_part_staff(rh, ks, numerator, denominator, bpm, clef.TrebleClef())
    lh_ps, lh_notes = _make_part_staff(lh, ks, numerator, denominator, bpm, clef.BassClef())

    if pedal_segments:
        _add_pedals(lh_ps, lh_notes, pedal_segments, numerator * 4.0 / denominator)

    rh_ps.makeMeasures(inPlace=True)
    lh_ps.makeMeasures(inPlace=True)

    # Fill gaps with visible rests. Without this, m21ToXml emits leading/trailing
    # rests (e.g. before a pickup) as print-object="no" placeholders that
    # MuseScore hides. fillGaps handles interior gaps; timeRangeFromBarDuration
    # extends the fill to the full measure (barDuration), so the tail after the
    # last note is a visible rest too, not a hidden export-time balancing rest.
    for m in rh_ps.getElementsByClass(stream.Measure):
        m.makeRests(fillGaps=True, timeRangeFromBarDuration=True,
                    hideRests=False, inPlace=True)
    for m in lh_ps.getElementsByClass(stream.Measure):
        m.makeRests(fillGaps=True, timeRangeFromBarDuration=True,
                    hideRests=False, inPlace=True)

    score.append(rh_ps)
    score.append(lh_ps)
    score.append(layout.StaffGroup(rh_ps, lh_ps, symbol="brace", barTogether=True))
    return score


def _make_part_staff(events, ks, numerator, denominator, bpm, clef_obj):
    """Build a flat PartStaff (notes at global offsets) plus its note objects.

    ``makeMeasures`` is applied by the caller so pedal marks can be added first.
    """
    ps = stream.PartStaff()
    ps.partName = "Piano"
    ps.insert(0, instrument.Piano())

    # anchors at offset 0 (before any note)
    ps.insert(0, tempo_mark.MetronomeMark(number=int(round(bpm))))
    ps.insert(0, clef_obj)
    ps.insert(0, meter.TimeSignature(f"{numerator}/{denominator}"))
    ps.insert(0, key.KeySignature(ks))

    notes = []
    for onset, dur, pitches in events:
        n = _make_note(pitches, dur, ks)
        ps.insert(onset, n)
        notes.append((onset, n))

    return ps, notes


def _merge_pedal_segments(segments, max_gap=1.0):
    """Merge nearby pedal presses (AMT flutter) into single sustained presses.

    The AMT often emits a rapid down/up stutter where the player simply holds
    the pedal. Segments whose down follows the previous up by less than
    ``max_gap`` quarter lengths are merged, so the score shows one held pedal
    instead of a run of break/resume marks.
    """
    merged = []
    for down, up in segments:
        if merged and down - merged[-1][1] < max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], up))
        else:
            merged.append((down, up))
    return merged


def _add_pedals(lh_ps, notes, pedal_segments, measure_ql):
    """Add sustain-pedal marks aligned to whole measures.

    The AMT's raw pedal down/up times don't land on notes, and anchoring each
    press to the nearest LH note gave marks with wrong position and length. So
    instead snap each press to whole measures (down -> measure start, up ->
    measure end), merge contiguous measures, then anchor the result to the
    first/last LH note in range. This trades fine timing for clean alignment.
    """
    notes = sorted(notes, key=lambda x: x[0])
    onsets = [t for t, _ in notes]
    snapped = []
    for down, up in pedal_segments:
        d = math.floor(down / measure_ql) * measure_ql
        u = math.ceil(up / measure_ql) * measure_ql
        snapped.append((d, max(u, d + measure_ql)))
    for down, up in _merge_pedal_segments(snapped, max_gap=1e-6):
        # first note at/after the measure start, last note at/before the end
        di = next((i for i, t in enumerate(onsets) if t >= down - 1e-6), None)
        ui = next((i for i in range(len(onsets) - 1, -1, -1) if onsets[i] <= up + 1e-6), None)
        if di is None or ui is None or di > ui:
            continue
        pm = expressions.PedalMark()
        pm.pedalType = expressions.PedalType.Sustain
        pm.pedalForm = expressions.PedalForm.Line
        pm.addSpannedElements([notes[di][1], notes[ui][1]])
        lh_ps.insert(down, pm)


def _strip_pedal_numbers(xml_path):
    """Remove music21's per-spanner ``number`` attribute from ``<pedal>`` tags.

    music21 assigns a distinct ``number`` to every ``PedalMark`` spanner (1, 2,
    3...); MuseScore reads those as separate pedal lines and draws one line per
    press. A single sustain pedal should be one line, so strip the attribute
    (matching the reference, which omits it and falls back to the default 1).
    """
    text = Path(xml_path).read_text(encoding="utf-8")
    text = re.sub(r'(<pedal[^>]*?) number="\d+"', r"\1", text)
    Path(xml_path).write_text(text, encoding="utf-8")


def _pedal_segments(pedal_events, bpm, min_beat=0.25, subdiv=2):
    """Convert CC64 (time, value) events to quantized (down_ql, up_ql) segments.

    Endpoints are snapped to the subdiv grid so pedal marks align with notes.
    """
    beat_len = 60.0 / bpm
    grid = beat_len / subdiv
    segments = []
    down = None
    for time, value in sorted(pedal_events):
        if value >= 64 and down is None:
            down = time
        elif value < 64 and down is not None:
            if (time - down) / beat_len >= min_beat:
                d_ql = round(down / grid) * grid / beat_len
                u_ql = round(time / grid) * grid / beat_len
                if u_ql > d_ql:
                    segments.append((round(d_ql, 4), round(u_ql, 4)))
            down = None
    if down is not None:
        d_ql = round(down / grid) * grid / beat_len
        segments.append((round(d_ql, 4), round(d_ql + 1.0, 4)))
    return segments


_LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
_NATURAL_PC = [0, 2, 4, 5, 7, 9, 11]      # natural pitch class of each letter
_SHARP_ORDER = [3, 0, 4, 1, 5, 2, 6]      # F C G D A E B (letter indices)
_FLAT_ORDER = [6, 2, 5, 1, 4, 0, 3]       # B E A D G C F


def _spell_note(midi, ks):
    """Return a spelled note name (e.g. 'A#4') matching key signature ``ks``.

    ``ks`` is the sharps count (negative = flats). Returns None if no better
    spelling is found (caller falls back to the MIDI default).
    """
    pc = midi % 12
    octave = midi // 12 - 1
    alters = [0] * 7
    if ks > 0:
        for i in range(min(ks, 7)):
            alters[_SHARP_ORDER[i]] = 1
    elif ks < 0:
        for i in range(min(-ks, 7)):
            alters[_FLAT_ORDER[i]] = -1
    # diatonic spelling
    for li in range(7):
        if (_NATURAL_PC[li] + alters[li]) % 12 == pc:
            acc = "#" if alters[li] == 1 else "b" if alters[li] == -1 else ""
            return f"{_LETTERS[li]}{acc}{octave}"
    # chromatic: prefer a natural sign (lowering a sharped note / raising a
    # flatted note) over adding a sharp/flat, which reads more naturally
    if ks > 0:
        for li in range(7):
            if alters[li] == 1 and _NATURAL_PC[li] == pc:
                return f"{_LETTERS[li]}{octave}"           # e.g. F-natural in D
        for li in range(7):
            if alters[li] == 0 and (_NATURAL_PC[li] + 1) % 12 == pc:
                return f"{_LETTERS[li]}#{octave}"          # e.g. A# in D
    else:
        for li in range(7):
            if alters[li] == -1 and _NATURAL_PC[li] == pc:
                return f"{_LETTERS[li]}{octave}"           # e.g. B-natural in F
        for li in range(7):
            if alters[li] == 0 and (_NATURAL_PC[li] - 1) % 12 == pc:
                return f"{_LETTERS[li]}b{octave}"          # e.g. Db in F
    return None


def _make_note(pitches, qlen, ks=0):
    if len(pitches) == 1:
        name = _spell_note(int(pitches[0]), ks)
        n = note.Note(name) if name else note.Note(midi=int(pitches[0]))
    else:
        c = chord.Chord()
        for p in sorted(pitches):
            name = _spell_note(int(p), ks)
            c.add(note.Note(name) if name else note.Note(midi=int(p)))
        n = c
    n.quarterLength = qlen
    return n
