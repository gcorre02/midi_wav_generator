#!/usr/bin/env python3
# make_tracks_v5.py
# Generate WAV + MIDI tracks with modal keys + Roman numeral chord progressions

import argparse, os, math, wave, csv, re
import numpy as np
import yaml

# ----------- MIDI optional ----------
try:
    from midiutil import MIDIFile

    HAVE_MIDI = True
except ImportError:
    HAVE_MIDI = False

# ----------- Modal Scales -----------
MODES = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],  # major
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],  # natural minor
    "locrian": [0, 1, 3, 5, 6, 8, 10]
}

NOTE_TO_MIDI = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
                "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def parse_key(s: str):
    parts = s.strip().split()
    tonic = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "ionian"
    if tonic not in NOTE_TO_MIDI: raise ValueError(f"Bad tonic {tonic}")
    if mode not in MODES: raise ValueError(f"Unsupported mode {mode}")
    root_pc = NOTE_TO_MIDI[tonic]
    return root_pc, mode


def build_scale(root_pc, mode):
    return [(root_pc + step) % 12 for step in MODES[mode]]


# ----------- Roman Parser -----------
ROMAN_RE = re.compile(r'([b#]?)([ivIV]+)(.*)')


def roman_to_degree(r: str):
    match = ROMAN_RE.match(r)
    if not match: raise ValueError(f"Bad numeral {r}")
    accidental, roman, rest = match.groups()
    valmap = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5, "VII": 6}
    deg = valmap[roman.upper()]
    if accidental == "b": deg -= 1
    if accidental == "#": deg += 1
    deg %= 7
    return deg, roman.isupper(), rest


def chord_from_roman(r: str, scale):
    deg, is_upper, rest = roman_to_degree(r)
    root_pc = scale[deg]

    # triad default
    third = scale[(deg + 2) % 7]
    fifth = scale[(deg + 4) % 7]

    notes = [root_pc, third, fifth]

    # quality adjust from case
    if is_upper:
        pass  # major implied
    else:
        pass  # minor implied (we use scale as given)

    # parse modifiers
    if rest:
        if "maj7" in rest:
            notes.append(scale[(deg + 6) % 7])  # major 7
        elif "m7" in rest or "7" in rest:
            notes.append((root_pc + 10) % 12)  # minor7 dom7 fudge
        if "sus2" in rest:
            notes[1] = scale[(deg + 1) % 7]
        if "sus4" in rest:
            notes[1] = scale[(deg + 3) % 7]
        if "add9" in rest:
            notes.append(scale[(deg + 1) % 7])

    # Convert to MIDI around C3
    root_midi = 48 + root_pc
    midi_notes = [root_midi + (n - root_pc) for n in notes]
    return sorted(midi_notes)


def progression_to_chords(prog_list, key):
    root_pc, mode = parse_key(key)
    scale = build_scale(root_pc, mode)
    return [chord_from_roman(r, scale) for r in prog_list]


# ----------- Synth helpers ----------
def midi_to_freq(m): return 440.0 * (2.0 ** ((m - 69) / 12.0))


def adsr(length, sr, a=0.02, d=0.12, s=0.75, r=0.18):
    a_len = int(length * a);
    d_len = int(length * d);
    r_len = int(length * r)
    s_len = max(0, length - (a_len + d_len + r_len))
    attack = np.linspace(0, 1, max(1, a_len), endpoint=True)
    decay = np.linspace(1, s, max(1, d_len), endpoint=True)
    sustain = np.full(max(1, s_len), s, dtype=np.float32)
    release = np.linspace(s, 0, max(1, r_len), endpoint=True)
    env = np.concatenate([attack, decay, sustain, release]).astype(np.float32)
    if len(env) < length: env = np.pad(env, (0, length - len(env)), "edge")
    return env


def synth_tone(freq, dur_s, sr, bright=False, color="neutral"):
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    base = np.sin(2 * math.pi * freq * t)
    h2 = (0.35 if bright else 0.25) * np.sin(2 * math.pi * 2 * freq * t)
    h3 = (0.20 if bright else 0.15) * np.sin(2 * math.pi * 3 * freq * t)
    tone = (base + h2 + h3)
    if color == "bright": tone += 0.2 * np.sin(2 * math.pi * 4 * freq * t)
    if color == "warm": tone *= 0.9
    tone /= np.max(np.abs(tone))
    env = adsr(len(tone), sr, a=0.01 if bright else 0.02, d=0.08, s=0.7, r=0.18)
    return (tone * env).astype(np.float32)


def synth_thunder(dur_s, sr, base_freq=55.0):
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    freq = base_freq * (1 - 0.12 * (t / max(1e-6, dur_s)))
    sine = np.sin(2 * math.pi * freq * t)
    noise = np.random.normal(0, 1, len(t))
    lp = np.zeros_like(noise);
    alpha = 0.02
    for i in range(1, len(noise)): lp[i] = alpha * noise[i] + (1 - alpha) * lp[i - 1]
    body = 0.85 * sine + 0.35 * lp
    env = adsr(len(body), sr, a=0.08, d=0.18, s=0.6, r=0.35)
    return np.tanh(body * env * 0.9).astype(np.float32)


def write_wav(path, audio, sr):
    audio = audio / (np.max(np.abs(audio)) or 1.0) * 0.95
    int16 = (audio * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1);
        wf.setsampwidth(2);
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    print("[WAV]", path)


# ----------- Renderers -----------
def render_audio(bpm, loops, sr, style, prog, color,
                 rain_start, rain_end, jitter_ms,
                 thunder, thunder_every, thunder_base,
                 crescendo_csv=None, name=None):
    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat
    total_bars = loops * len(prog)
    rng = np.random.default_rng(42)

    if style == "pad":
        blocks = []
        for bar in range(total_bars):
            notes = prog[bar % len(prog)]
            parts = [synth_tone(midi_to_freq(m), bar_dur, sr, color=color) for m in notes]
            block = np.sum(np.stack(parts), axis=0)
            if thunder and bar % max(1, thunder_every) == 0:
                th = synth_thunder(min(bar_dur * 0.9, 2.5), sr, base_freq=thunder_base)
                L = min(len(block), len(th))
                block[:L] += th[:L] * 0.9
            blocks.append(np.tanh(block * 0.8).astype(np.float32))
        return np.concatenate(blocks), None

    elif style == "arp":
        step = 0.5
        step_dur = step * sec_per_beat
        pattern = [0, 1, 2, 1, 3, 2, 1, 0]
        blocks = []
        for bar in range(total_bars):
            notes = prog[bar % len(prog)]
            barbuf = np.zeros(int(sr * bar_dur), dtype=np.float32)
            for s, idx in enumerate(pattern):
                si = int(s * step_dur * sr)
                tone = synth_tone(midi_to_freq(notes[idx % len(notes)]), step_dur, sr,
                                  bright=True, color=color)
                ei = min(len(barbuf), si + len(tone))
                barbuf[si:ei] += tone[:ei - si]
            if thunder and bar % max(1, thunder_every) == 0:
                th = synth_thunder(min(bar_dur * 0.9, 2.4), sr, base_freq=thunder_base)
                L = min(len(barbuf), len(th))
                barbuf[:L] += th[:L] * 0.9
            blocks.append(np.tanh(barbuf * 1.1))
        return np.concatenate(blocks), None

    elif style == "rain":
        # RAIN: capture stochastic events for MIDI
        steps = 8
        base_step = bar_dur / steps
        jitter_s = jitter_ms / 1000.0
        blocks = []
        rows = []
        events = []  # (onset_sec, dur_sec, midi_note, velocity)

        for bar in range(total_bars):
            alpha = bar / max(1, total_bars - 1)
            density = (1 - alpha) * rain_start + alpha * rain_end
            swell = 0.925 + 0.075 * math.sin(2 * math.pi * bar / max(1, total_bars))
            notes = prog[bar % len(prog)]
            order = [0, 1, 2, 1, 3, 2, 1, 0]
            barbuf = np.zeros(int(sr * bar_dur), dtype=np.float32)

            for s in range(steps):
                if rng.random() <= density:
                    idx = order[s % len(order)]
                    dur = base_step * rng.uniform(0.35, 0.6)
                    start = s * base_step + rng.uniform(-jitter_s, jitter_s)

                    # clamp start safely inside the bar
                    start = max(0.0, min(bar_dur - (1.0 / sr), start))

                    tone = synth_tone(midi_to_freq(notes[idx % len(notes)]), dur, sr,
                                      bright=True, color=color) * swell

                    si = int(start * sr)
                    ei = min(len(barbuf), si + len(tone))
                    if si >= len(barbuf) or ei <= si:
                        continue

                    barbuf[si:ei] += tone[:ei - si]

                    # MIDI velocity ~ 75 * swell * rand
                    vel = int(75 * swell * rng.uniform(0.85, 1.1))
                    vel = max(30, min(115, vel))
                    # absolute onset in seconds:
                    abs_onset = bar * bar_dur + start
                    events.append((abs_onset, dur, notes[idx % len(notes)], vel))

                    # optional echo
                    if rng.random() < 0.12:
                        echo_start = start + base_step * rng.uniform(0.12, 0.25)
                        echo_start = min(bar_dur - (1.0 / sr), max(0.0, echo_start))
                        esi = int(echo_start * sr)
                        eei = min(len(barbuf), esi + len(tone))
                        if esi < len(barbuf) and eei > esi:
                            barbuf[esi:eei] += tone[:eei - esi] * rng.uniform(0.5, 0.7)
                            # log echo as a slightly softer MIDI note
                            echo_vel = int(vel * rng.uniform(0.5, 0.7))
                            echo_abs_onset = bar * bar_dur + echo_start
                            events.append((echo_abs_onset, dur * 0.7, notes[idx % len(notes)], echo_vel))

            if thunder and (bar % max(1, thunder_every) == 0):
                th = synth_thunder(min(bar_dur * 0.9, 2.4), sr, base_freq=thunder_base)
                L = min(len(barbuf), len(th))
                barbuf[:L] += th[:L] * 0.9

            blocks.append(np.tanh(barbuf * 1.15))
            rows.append({"bar": bar + 1, "density": density, "swell": swell})

        if crescendo_csv and name:
            csvpath = f"{name}_crescendo.csv"
            import csv
            with open(csvpath, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["bar", "density", "swell"])
                w.writeheader()
                w.writerows(rows)
            print("[CSV]", csvpath)

        return np.concatenate(blocks), events

    else:
        raise ValueError("style must be pad|arp|rain")


# ----------- Main -----------
def main():
    import yaml, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)) if args.config.endswith((".yaml", ".yml")) else json.load(open(args.config))
    outdir = cfg.get("outdir", ".");
    defaults = cfg.get("defaults", {});
    jobs = cfg["jobs"]
    for job in jobs:
        j = {**defaults, **job}
        name = j["name"];
        key = j["key"];
        prog = j["progression"]
        chords = progression_to_chords(prog, key)
        bpm = int(j.get("bpm", 72));
        loops = int(j.get("loops", 8));
        sr = int(j.get("sr", 44100))
        style = j.get("style", "pad");
        color = j.get("color", "neutral")
        thunder = j.get("thunder", False);
        thunder_every = j.get("thunder_every", 2);
        thunder_base = j.get("thunder_base", 55.0)
        rain_start = float(j.get("rain_start", 0.8));
        rain_end = float(j.get("rain_end", 0.8));
        jitter_ms = float(j.get("jitter_ms", 15))
        crescendo_csv = j.get("crescendo_csv", False)

        audio, events = render_audio(
            bpm, loops, sr, style, chords, color,
            rain_start, rain_end, jitter_ms,
            thunder, thunder_every, thunder_base,
            crescendo_csv, name
        )
        wav_path = os.path.join(outdir, f"{name}_{style.upper()}.wav")
        write_wav(wav_path, audio, sr)

        if HAVE_MIDI:
            mid_path = os.path.join(outdir, f"{name}_{style.upper()}.mid")
            mf = MIDIFile(1)
            track = 0
            mf.addTempo(track, 0, bpm)

            if style == "rain" and events:
                # mirror stochastic events exactly
                for onset_sec, dur_sec, midi_note, vel in events:
                    onset_beats = onset_sec * (bpm / 60.0)
                    dur_beats = max(0.05, dur_sec * (bpm / 60.0))
                    mf.addNote(track, 0, midi_note, onset_beats, dur_beats, vel)
            else:
                # pad/arp fallback: block chords per bar
                time_beats = 0.0
                for _ in range(loops):
                    for chord in chords:
                        for n in chord:
                            mf.addNote(track, 0, n, time_beats, 4, 90)
                        time_beats += 4

            with open(mid_path, "wb") as f:
                mf.writeFile(f)
            print("[MIDI]", mid_path)


if __name__ == "__main__":
    main()
