#!/usr/bin/env python3
# blade_runner_pack_config.py
# Config-driven Blade Runner-style pack:
#  1) Chord/Voicing PDF
#  2) Mono practice WAV (pads + bass + arp)
#  3) Mono rolling percussion WAV (brush ghosts + tom swells + shaker)

import math, wave, os, re
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
try:
    from midiutil import MIDIFile
    HAVE_MIDI = True
except ImportError:
    HAVE_MIDI = False

# =========================
# ======== CONFIG =========
# =========================
CONFIG = {
    # Global song settings
    "key": "D",                # tonic (e.g., "D", "Bb", "F#")
    "mode": "Dorian",          # Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian
    "bpm": 88,                 # tempo
    "sr": 44100,               # sample rate
    "bars": 11,                # total bars (~30s at 88bpm)

    # Theme progressions as Roman numerals (triads default)
    # supported modifiers: maj7, m7, 7, sus2, sus4, add9
    "theme_A": ["i(add9)", "VII", "IIImaj7", "IVsus2"],  # e.g., D Dorian -> Dm(add9) C Bbmaj7 Gsus2
    "theme_B": ["bVIImaj7", "IV", "IVsus2", "i(add9)"],  # e.g., Bbmaj7 F Gsus2 Dm(add9)

    # Optional: PDF voicing octave and practice-bed choices
    "root_oct_pdf": 3,     # where to center stacks for PDF display
    "root_oct_audio": 3,   # where to center stacks for audio pad/arp
    "bass_oct": 2,         # octave for bass pulses

    # Output filenames
    "pdf_path": "blade_runner_chords_voicings.pdf",
    "practice_wav": "blade_runner_practice.wav",
    "perc_wav": "blade_runner_percussion.wav",

    "export_midi": True,  # turn on/off MIDI export
    "perc_midi_path": "blade_runner_percussion.mid",
    "arp_midi_path": "blade_runner_arp.mid",
}

# =========================
# ====== MUSIC CORE =======
# =========================

NOTE_TO_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
              "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}

MODES = {
    "ionian":     [0,2,4,5,7,9,11],  # major
    "dorian":     [0,2,3,5,7,9,10],
    "phrygian":   [0,1,3,5,7,8,10],
    "lydian":     [0,2,4,6,7,9,11],
    "mixolydian": [0,2,4,5,7,9,10],
    "aeolian":    [0,2,3,5,7,8,10],  # natural minor
    "locrian":    [0,1,3,5,6,8,10]
}

ROMAN_RE = re.compile(
    r'^([b#]?)([ivIV]+)(?:\(([^)]*)\))?(?:\s*(maj7|m7|7|sus2|sus4|add9))?$'
)
def parse_key_mode(tonic: str, mode_name: str):
    if tonic not in NOTE_TO_PC:
        raise ValueError(f"Bad tonic '{tonic}'")
    mode_l = mode_name.strip().lower()
    if mode_l not in MODES:
        raise ValueError(f"Unsupported mode '{mode_name}'")
    return NOTE_TO_PC[tonic], mode_l

def build_scale(tonic_pc: int, mode_l: str):
    return [(tonic_pc + step) % 12 for step in MODES[mode_l]]

def name_to_midi(nm: str, octave: int) -> int:
    pc = NOTE_TO_PC[nm]
    return 12 * (octave + 1) + pc

def pc_to_name(pc: int):
    # For display only; pick flats for black keys to match cinematic vibe
    names = {0:"C",1:"Db",2:"D",3:"Eb",4:"E",5:"F",6:"Gb",7:"G",8:"Ab",9:"A",10:"Bb",11:"B"}
    return names[pc % 12]


def roman_to_degree(roman: str):
    m = ROMAN_RE.match(roman.strip())
    if not m:
        raise ValueError(f"Bad Roman '{roman}'")
    accidental, core, paren_mods, trailing = m.groups()
    valmap = {"I":0,"II":1,"III":2,"IV":3,"V":4,"VI":5,"VII":6}
    deg = valmap[core.upper()]
    is_upper = core.isupper()
    if accidental == "b":
        deg = (deg - 1) % 7
    elif accidental == "#":
        deg = (deg + 1) % 7

    mods = []
    if paren_mods:
        for mtxt in paren_mods.split(","):
            mtxt = mtxt.strip().lower()
            if mtxt:
                mods.append(mtxt)
    if trailing:
        mods.append(trailing.strip().lower())

    return deg, is_upper, mods, accidental


def build_chord_from_roman(roman: str, scale_pcs: list[int], add_seventh_by_modifier=True):
    """
    Triads by default; supports maj7, m7, 7, sus2, sus4, add9.
    Case of Roman chooses maj/min triad relative to scale degree quality (modal).
    We keep triads modal (take 1-3-5 degrees), then apply sus/add extensions.
    """
    deg, is_upper, mods, accidental = roman_to_degree(roman)
    # core triad from modal scale degrees: 1-3-5 relative
    d1 = scale_pcs[deg]
    d3 = scale_pcs[(deg + 2) % 7]
    d5 = scale_pcs[(deg + 4) % 7]
    chord = [d1, d3, d5]  # triad

    # sus mods override third
    if "sus2" in mods:
        chord[1] = scale_pcs[(deg + 1) % 7]
    if "sus4" in mods:
        chord[1] = scale_pcs[(deg + 3) % 7]

    # sevenths
    if "maj7" in mods:
        chord.append(scale_pcs[(deg + 6) % 7])
    elif "m7" in mods:
        # modal seventh often works; but ensure minor7 interval from root
        chord.append((d1 + 10) % 12)
    elif "7" in mods:
        # dom7 flavor
        chord.append((d1 + 10) % 12)

    # add9
    if "add9" in mods:
        chord.append(scale_pcs[(deg + 1) % 7])

    # label for PDF
    # Try to label with accidental if present and case for maj/min
    root_name = pc_to_name(d1)
    label = roman  # keep as given for clarity
    return chord, label, root_name

def stack_to_midi(chord_pcs: list[int], root_pc: int, root_oct: int) -> list[int]:
    """Spread chord tones into a playable ascending stack near root_oct."""
    out = []
    base = 12 * (root_oct + 1) + root_pc  # root around C3–C4 region
    last = base - 12
    for pc in chord_pcs:
        # map pc near base
        m = 12 * (root_oct + 1) + pc
        # lift until ascending-ish
        while m <= last:
            m += 12
        out.append(m)
        last = m
    return out

def roman_progression_to_stacks(config):
    tonic_pc, mode_l = parse_key_mode(CONFIG["key"], CONFIG["mode"])
    scale = build_scale(tonic_pc, mode_l)

    def to_stacks(roman_list):
        stacks = []   # list of (label, pcs_list, root_pc)
        for r in roman_list:
            pcs, label, root_name = build_chord_from_roman(r, scale)
            stacks.append((label, pcs, NOTE_TO_PC[root_name]))
        return stacks
    return to_stacks(CONFIG["theme_A"]), to_stacks(CONFIG["theme_B"])

# =========================
# ====== AUDIO CORE =======
# =========================

def write_wav_mono(path, audio, sr=44100):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) or 1.0
    audio = (audio / peak) * 0.95
    pcm = (audio * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"[WAV] {path}")

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def adsr_env(n, sr, a=0.02, d=0.12, s=0.75, r=0.18):
    """
    ADSR where a,d,r are FRACTIONS of the total duration.
    Guarantees len(env) == n by renormalizing and final clamp/pad.
    """
    # guard
    n = max(1, int(n))

    # raw fractional lengths
    a_n = int(n * max(0.0, a))
    d_n = int(n * max(0.0, d))
    r_n = int(n * max(0.0, r))

    total_adr = a_n + d_n + r_n
    if total_adr > n:  # renormalize to fit exactly
        scale = n / total_adr
        a_n = int(a_n * scale)
        d_n = int(d_n * scale)
        r_n = max(0, n - (a_n + d_n))  # remainder to r

    s_n = max(0, n - (a_n + d_n + r_n))

    # build segments
    att = np.linspace(0.0, 1.0, max(1, a_n), endpoint=True)
    dec = np.linspace(1.0, max(0.0, s), max(1, d_n), endpoint=True)
    sus = np.full(max(1, s_n), max(0.0, s), dtype=np.float32)
    rel = np.linspace(max(0.0, s), 0.0, max(1, r_n), endpoint=True)

    env = np.concatenate([att, dec, sus, rel]).astype(np.float32)

    # FINAL SAFETY: enforce exact n
    if len(env) > n:
        env = env[:n]
    elif len(env) < n:
        env = np.pad(env, (0, n - len(env)), mode="edge")

    return env


def synth_tone(freq, dur_s, sr=44100, kind="pad", detune_cents=0.0):
    """Duration-robust synth; guarantees tone/env lengths match."""
    # guard duration → samples
    n = max(1, int(sr * max(1e-6, dur_s)))
    t = np.linspace(0.0, dur_s, n, endpoint=False)
    f = float(freq) * (2.0 ** (detune_cents / 1200.0))

    if kind == "pad":
        base = np.sin(2*np.pi*f*t, dtype=np.float64)
        # band-limited-ish saw (simple), still length n
        saw  = 2.0 * (t*f - np.floor(0.5 + t*f))
        tone = 0.45*base + 0.55*saw
        env  = adsr_env(len(tone), sr, a=0.12, d=0.60, s=0.75, r=1.80)

    elif kind == "arp":
        sq   = np.sign(np.sin(2*np.pi*f*t))
        tone = 0.6 * sq
        env  = adsr_env(len(tone), sr, a=0.01, d=0.08, s=0.60, r=0.12)

    elif kind == "bass":
        s1 = np.sin(2*np.pi*f*t)
        s2 = 0.2 * np.sin(2*np.pi*(2*f)*t)
        tone = 0.85*s1 + 0.15*s2
        env  = adsr_env(len(tone), sr, a=0.01, d=0.08, s=0.80, r=0.20)

    elif kind == "tom":
        f_inst = f * (1.0 - 0.15 * (t / max(1e-6, dur_s)))
        tone = np.sin(2*np.pi*f_inst*t)
        env  = adsr_env(len(tone), sr, a=0.05, d=0.20, s=0.60, r=0.40)

    elif kind == "noise":
        tone = np.random.normal(0.0, 1.0, n).astype(np.float32)
        env  = adsr_env(len(tone), sr, a=0.005, d=0.05, s=0.40, r=0.08)

    else:  # default sine
        tone = np.sin(2*np.pi*f*t)
        env  = adsr_env(len(tone), sr)

    # normalize & FINAL CLAMP
    tone = tone.astype(np.float32)
    peak = np.max(np.abs(tone)) or 1.0
    tone = tone / peak

    # hard enforce equal lengths (paranoid safeguard)
    nmin = min(len(tone), len(env))
    if len(tone) != len(env):
        tone = tone[:nmin]
        env  = env[:nmin]

    return (tone * env).astype(np.float32)



def iir_lowpass(x, alpha=0.02):
    y = np.zeros_like(x, dtype=np.float32)
    acc = 0.0
    for i in range(len(x)):
        acc = alpha * x[i] + (1 - alpha) * acc
        y[i] = acc
    return y

def iir_highpass(x, alpha=0.02):
    return (x - iir_lowpass(x, alpha)).astype(np.float32)

# =========================
# ======== PDF OUT ========
# =========================

def make_pdf(path, themeA_stacks, themeB_stacks):
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Blade Runner Piece — Chords & Voicings ({CONFIG['key']} {CONFIG['mode']})", styles["Heading1"]))
    story.append(Spacer(1, 12))

    def section(label, stacks):
        data = [["Chord (Roman)", "Stack (PC names)", "Suggested Piano Voicing (LH / RH)"]]
        for roman, pcs, root_pc in stacks:
            # Pretty-print stack names
            stack_names = " – ".join(pc_to_name(pc) for pc in pcs)
            # LH: root + fifth, RH: upper tones
            # derive display names from pcs list order (already 1-3-5-..)
            names = [pc_to_name(pc) for pc in pcs]
            lh = f"{names[0]} / {names[1] if len(names)>1 else names[0]}"
            rh = " – ".join(names[2:]) if len(names) > 2 else "(—)"
            data.append([roman, stack_names, f"LH: {lh} | RH: {rh}"])
        tbl = Table(data, colWidths=[120, 200, 220])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))
        story.append(tbl)
        story.append(Spacer(1, 16))

    section("Theme A", themeA_stacks)
    section("Theme B", themeB_stacks)

    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(story)
    print(f"[PDF] {path}")

# =========================
# ======== AUDIO OUT ======
# =========================
def build_arp_events(themeA_stacks, themeB_stacks):
    """Return list of (onset_sec, dur_sec, midi_note, velocity) for the arp line."""
    bpm = CONFIG["bpm"]; bars = CONFIG["bars"]
    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat
    prog = themeA_stacks + themeB_stacks
    plen = len(prog)

    events = []
    for bar in range(bars):
        roman, pcs, root_pc = prog[bar % plen]
        mids = stack_to_midi(pcs, root_pc, CONFIG["root_oct_audio"])
        arp_notes = mids[-3:] if len(mids) >= 3 else mids
        if not arp_notes:
            continue
        # 1/16th pattern (same as in practice bed)
        step = sec_per_beat / 4.0
        pattern = [0,1,2,1,  2,1,0,1]
        for s, idx in enumerate(pattern):
            onset = bar * bar_dur + s * step
            dur = step * 0.9
            note = arp_notes[idx % len(arp_notes)]
            vel = 75
            events.append((onset, dur, note, vel))
    return events

def write_arp_midi(path, events):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skipping arp MIDI.")
        return
    bpm = CONFIG["bpm"]
    mf = MIDIFile(1)
    track = 0
    mf.addTempo(track, 0, bpm)
    ch = 0  # normal instrument channel
    for onset_sec, dur_sec, note, vel in events:
        onset_beats = onset_sec * (bpm / 60.0)
        dur_beats = max(0.05, dur_sec * (bpm / 60.0))
        mf.addNote(track, ch, note, onset_beats, dur_beats, vel)
    with open(path, "wb") as f:
        mf.writeFile(f)
    print(f"[MIDI] {path}")

def render_practice_wav(path, themeA_stacks, themeB_stacks):
    bpm = CONFIG["bpm"]; sr = CONFIG["sr"]; bars = CONFIG["bars"]
    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat
    total = np.zeros(int(sr * bar_dur * bars), dtype=np.float32)
    prog = themeA_stacks + themeB_stacks
    plen = len(prog)
    rng = np.random.default_rng(42)

    for bar in range(bars):
        roman, pcs, root_pc = prog[bar % plen]
        bar_sig = np.zeros(int(sr * bar_dur), dtype=np.float32)

        # Pad stack (detuned)
        mids = stack_to_midi(pcs, root_pc, CONFIG["root_oct_audio"])
        for i, m in enumerate(mids):
            f = midi_to_hz(m)
            det = (-8 + 16 * rng.random())  # ±8 cents
            tone = synth_tone(f, bar_dur, sr, kind="pad", detune_cents=det)
            bar_sig[:len(tone)] += tone * (0.22 if i == 0 else 0.18)

        # Bass pulse on beats 1 & 3 (use first tone as "root")
        root_m = 12 * (CONFIG["bass_oct"] + 1) + root_pc
        f_b = midi_to_hz(root_m)
        for beat in (0, 2):
            start = int(beat * sec_per_beat * sr)
            dur = sec_per_beat * 0.9
            tone = synth_tone(f_b, dur, sr, kind="bass")
            end = min(len(bar_sig), start + len(tone))
            bar_sig[start:end] += tone[:end - start] * 0.35

        # Soft arp (16ths over top three tones)
        step = sec_per_beat / 4.0
        arp_notes = mids[-3:] if len(mids) >= 3 else mids
        pattern = [0,1,2,1, 2,1,0,1]
        for s, idx in enumerate(pattern):
            start = int(s * step * sr)
            dur = step * 0.9
            if not arp_notes: break
            m = arp_notes[idx % len(arp_notes)]
            tone = synth_tone(midi_to_hz(m), dur, sr, kind="arp")
            tone = iir_lowpass(tone, alpha=0.04)
            end = min(len(bar_sig), start + len(tone))
            bar_sig[start:end] += tone[:end - start] * 0.18

        # Place into total
        s0 = int(bar * bar_dur * sr)
        s1 = s0 + len(bar_sig)
        total[s0:s1] += bar_sig

    write_wav_mono(path, total, sr)

def build_percussion_events():
    """Return list of drum events [(onset_sec, dur_sec, gm_note, velocity)] fully matching the WAV generator."""
    bpm = CONFIG["bpm"]; sr = CONFIG["sr"]; bars = CONFIG["bars"]
    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat
    rng = np.random.default_rng(99)

    events = []
    for bar in range(bars):
        t_bar0 = bar * bar_dur

        # A) Brush ghosts: many tiny filtered taps with jitter  (Closed HH = 42)
        for slot in range(8):  # 8ths
            if rng.random() < 0.7:
                start_t = slot * (bar_dur/8.0) + rng.uniform(-0.015, 0.015)
                start_t = max(0.0, min(bar_dur - 1.0/CONFIG["sr"], start_t))
                dur = rng.uniform(0.04, 0.08)
                vel = int(rng.uniform(10, 25))
                events.append((t_bar0 + start_t, dur, 42, vel))

        # B) Shaker (Maracas = 70)
        for slot in range(8):
            if rng.random() < 0.55:
                start_t = slot * (bar_dur/8.0) + rng.uniform(-0.01, 0.01)
                start_t = max(0.0, min(bar_dur - 1.0/CONFIG["sr"], start_t))
                dur = rng.uniform(0.05, 0.09)
                vel = int(rng.uniform(35, 60))
                events.append((t_bar0 + start_t, dur, 70, vel))

        # C) Mallet tom swell every 4 bars (Low Floor Tom = 41)
        if bar % 4 == 0:
            start_t = 2.0 * sec_per_beat  # beats 3–4 swell
            dur = 1.6 * sec_per_beat
            vel = 50
            events.append((t_bar0 + start_t, dur, 41, vel))

    return events


def write_percussion_midi(path, events):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skipping percussion MIDI.")
        return
    bpm = CONFIG["bpm"]
    mf = MIDIFile(1)
    track = 0
    mf.addTempo(track, 0, bpm)
    drum_channel = 9  # GM channel 10 (0-indexed)
    for onset_sec, dur_sec, gm_note, vel in events:
        onset_beats = onset_sec * (bpm / 60.0)
        dur_beats = max(0.05, dur_sec * (bpm / 60.0))
        mf.addNote(track, drum_channel, gm_note, onset_beats, dur_beats, vel)
    with open(path, "wb") as f:
        mf.writeFile(f)
    print(f"[MIDI] {path}")


def render_percussion_wav(path):
    sr = CONFIG["sr"]; bpm = CONFIG["bpm"]
    sec_per_beat = 60.0 / bpm
    bars = CONFIG["bars"]; bar_dur = 4 * sec_per_beat

    total = np.zeros(int(sr * bar_dur * bars), dtype=np.float32)
    events = build_percussion_events()

    for onset_sec, dur_sec, gm_note, vel in events:
        si = int(onset_sec * sr)
        n = int(dur_sec * sr)
        if n <= 0 or si >= len(total):
            continue
        # synth a sound per instrument class
        if gm_note == 42:  # brush ghost -> airy noise tick
            tone = synth_tone(0, dur_sec, sr, kind="noise")
            tone = iir_lowpass(tone, alpha=0.06)
            tone = iir_highpass(tone, alpha=0.008)
            amp = vel / 127.0 * 0.12
        elif gm_note == 70:  # maracas
            tone = synth_tone(0, dur_sec, sr, kind="noise")
            tone = iir_lowpass(tone, alpha=0.03)
            amp = vel / 127.0 * 0.15
        elif gm_note == 41:  # low tom swell
            tone = synth_tone(70.0, dur_sec, sr, kind="tom")
            tone = iir_lowpass(tone, alpha=0.02)
            amp = vel / 127.0 * 0.6
        else:
            tone = synth_tone(0, dur_sec, sr, kind="noise")
            amp = vel / 127.0 * 0.1

        tone = (tone * amp).astype(np.float32)
        ei = min(len(total), si + len(tone))
        if ei > si:
            total[si:ei] += tone[:ei - si]

    write_wav_mono(path, total, sr)


import yaml, json

def export_batch_job_yaml(path, config, themeA, themeB):
    """
    Export a job config usable by make_tracks_v5.py
    """
    job = {
        "outdir": "renders",
        "defaults": {
            "bpm": config["bpm"],
            "loops": config["bars"],
            "sr": config["sr"],
            "style": "pad",
            "thunder": False
        },
        "jobs": [
            {
                "name": "blade_runner_piece",
                "key": f"{config['key']} {config['mode']}",
                "progression": config["theme_A"] + config["theme_B"],  # same roman strings
                "style": "pad",
                "bpm": config["bpm"],
                "loops": config["bars"],
                "color": "warm"
            }
        ]
    }
    with open(path, "w") as f:
        yaml.safe_dump(job, f)
    print(f"[YAML] {path}")

def export_batch_job_json(path, config):
    job = {
        "outdir": "renders",
        "defaults": {
            "bpm": config["bpm"],
            "loops": config["bars"],
            "sr": config["sr"],
            "style": "pad",
            "thunder": False
        },
        "jobs": [
            {
                "name": "blade_runner_piece",
                "key": f"{config['key']} {config['mode']}",
                "progression": config["theme_A"] + config["theme_B"],
                "style": "pad",
                "bpm": config["bpm"],
                "loops": config["bars"],
                "color": "warm"
            }
        ]
    }
    with open(path, "w") as f:
        json.dump(job, f, indent=2)
    print(f"[JSON] {path}")


# =========================
# ========= MAIN ==========
# =========================

def main():
    # Build stacks from config
    themeA, themeB = roman_progression_to_stacks(CONFIG)

    # Make PDF (shows Roman, stack names, and LH/RH suggestion)
    make_pdf(CONFIG["pdf_path"], themeA, themeB)

    # Practice WAV
    render_practice_wav(CONFIG["practice_wav"], themeA, themeB)

    # Percussion WAV
    render_percussion_wav(CONFIG["perc_wav"])

    export_batch_job_yaml("blade_runner_job.yaml", CONFIG, themeA, themeB)
    export_batch_job_json("blade_runner_job.json", CONFIG)

    if CONFIG.get("export_midi", False):
        # Percussion MIDI
        perc_events = build_percussion_events()
        write_percussion_midi(CONFIG["perc_midi_path"], perc_events)
        # Arp MIDI
        arp_events = build_arp_events(themeA, themeB)
        write_arp_midi(CONFIG["arp_midi_path"], arp_events)


if __name__ == "__main__":
    main()



