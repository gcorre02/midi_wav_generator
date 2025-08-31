# make_three_creatures_tracks_v4.py
# Generates MIDI + WAV practice tracks for "Three Creatures" in B♭ minor.
# Styles:
#   - pad: sustained chords
#   - arp: gentle, even arpeggios
#   - rain: humanized, irregular "rain" arpeggios with density ramp + jitter
#
# New in v4:
#   - Thunder bus: low tom/swell layered into WAV (+ optional MIDI hits)
#   - Rain density ramp: smoothly varies event density across whole track
#
# Install: pip install numpy midiutil

import argparse
import math
import wave
import numpy as np

# ---------- Shared helpers ----------
def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def write_wav(path, audio, sr):
    audio = audio / (np.max(np.abs(audio)) or 1.0) * 0.95
    int16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    print(f"[WAV] Wrote {path}")

def adsr(length, sr, a=0.02, d=0.12, s=0.75, r=0.18):
    a_len = int(length * a)
    d_len = int(length * d)
    r_len = int(length * r)
    s_len = max(0, length - (a_len + d_len + r_len))
    attack = np.linspace(0, 1, max(1, a_len), endpoint=True)
    decay  = np.linspace(1, s, max(1, d_len), endpoint=True)
    sustain= np.full(max(1, s_len), s, dtype=np.float32)
    release= np.linspace(s, 0, max(1, r_len), endpoint=True)
    env = np.concatenate([attack, decay, sustain, release]).astype(np.float32)
    if len(env) < length:
        env = np.pad(env, (0, length - len(env)), mode="edge")
    return env

def synth_tone(freq, dur_s, sr, bright=False):
    t = np.linspace(0, dur_s, int(sr*dur_s), endpoint=False)
    base = np.sin(2*np.pi*freq*t)
    h2 = (0.35 if bright else 0.25) * np.sin(2*np.pi*2*freq*t)
    h3 = (0.20 if bright else 0.15) * np.sin(2*np.pi*3*freq*t)
    tone = (base + h2 + h3) / (1 + (0.35 if bright else 0.25) + (0.20 if bright else 0.15))
    env = adsr(len(tone), sr, a=0.01 if bright else 0.02,
                         d=0.08 if bright else 0.12,
                         s=0.70, r=0.18)
    return (tone * env).astype(np.float32)

# ---------- Chords & progression (B♭ minor) ----------
CHORDS_MIDI = {
    "Bb_m_add9": [46, 49, 53, 60],  # Bb2, Db3, F3, C4
    "Ab_maj7":  [44, 48, 51, 55],  # Ab2, C3, Eb3, G3
    "Eb_add9":  [51, 53, 55, 58],  # Eb3, F3, G3, Bb3
    "F_sus2":   [53, 55, 60, 65],  # F3, G3, C4, F4
}
PROGRESSION = ["Bb_m_add9", "Ab_maj7", "Eb_add9", "F_sus2"]

# ---------- MIDI ----------
def write_midi(outfile, bpm, loops, style, rain_start, rain_end, jitter_ms,
               thunder, thunder_every):
    try:
        from midiutil import MIDIFile
    except ImportError:
        print("[MIDI] midiutil not installed; skipping MIDI file. (pip install midiutil)")
        return

    track = 0
    ch = 0
    vol = 90
    bpb = 4
    sec_per_beat = 60.0 / bpm

    midi = MIDIFile(1)
    midi.addTrackName(track, 0, f"Three Creatures {style.upper()}")
    midi.addTempo(track, 0, bpm)

    rng = np.random.default_rng(42)
    time_beats = 0.0
    total_bars = loops * len(PROGRESSION)

    if style == "pad":
        for bar in range(total_bars):
            name = PROGRESSION[bar % len(PROGRESSION)]
            for n in CHORDS_MIDI[name]:
                midi.addNote(track, ch, n, time_beats, bpb, vol)
            # thunder MIDI hit?
            if thunder and (bar % max(1, thunder_every) == 0):
                # super-low hit (Bb1 ~ 58 MIDI) or F1 (53) — choose one
                midi.addNote(track, ch, 53, time_beats, 0.8, 110)
            time_beats += bpb

    elif style == "arp":
        step = 0.5
        pattern = [0,1,2,1, 3,2,1,0]
        for bar in range(total_bars):
            name = PROGRESSION[bar % len(PROGRESSION)]
            notes = CHORDS_MIDI[name]
            t = time_beats
            for idx in pattern:
                midi.addNote(track, ch, notes[idx], t, step, vol)
                t += step
            if thunder and (bar % max(1, thunder_every) == 0):
                midi.addNote(track, ch, 53, time_beats, 0.8, 110)
            time_beats += bpb

    elif style == "rain":
        steps_per_bar = 8
        base_step = bpb / steps_per_bar
        jitter_beats = (jitter_ms / 1000.0) / sec_per_beat

        for bar in range(total_bars):
            # density ramp 0..1 across bars
            alpha = bar / max(1, total_bars-1)
            density = (1-alpha) * rain_start + alpha * rain_end

            name = PROGRESSION[bar % len(PROGRESSION)]
            notes = CHORDS_MIDI[name]
            order = [0,1,2,1, 3,2,1,0]

            # subtle bar-level swell
            swell = 0.925 + 0.075 * math.sin(2*math.pi * bar / max(1,total_bars))

            t = time_beats
            for step_i in range(steps_per_bar):
                if rng.random() <= density:
                    idx = order[step_i % len(order)]
                    if rng.random() < 0.2:
                        idx = rng.integers(0, len(notes))
                    onset = t + rng.uniform(-jitter_beats, jitter_beats)
                    dur = base_step * rng.uniform(0.35, 0.6)
                    velocity = int(75 * swell * rng.uniform(0.85, 1.1))
                    velocity = max(30, min(115, velocity))
                    midi.addNote(track, ch, notes[idx], onset, dur, velocity)
                    # echo
                    if rng.random() < 0.12:
                        echo_onset = onset + base_step * rng.uniform(0.12, 0.25)
                        echo_vel = int(velocity * rng.uniform(0.5, 0.7))
                        midi.addNote(track, ch, notes[idx], echo_onset, dur*0.7, echo_vel)
                t += base_step

            if thunder and (bar % max(1, thunder_every) == 0):
                midi.addNote(track, ch, 53, time_beats, 0.8, 110)

            time_beats += bpb

    else:
        raise ValueError("style must be 'pad', 'arp', or 'rain'")

    with open(outfile, "wb") as f:
        midi.writeFile(f)
    print(f"[MIDI] Wrote {outfile}")

# ---------- Thunder synthesis (WAV) ----------
def synth_thunder(dur_s, sr, base_freq=55.0):
    """
    Low, rumbly swell: sine + low noise, long ADSR, gentle pitch drop.
    """
    t = np.linspace(0, dur_s, int(sr*dur_s), endpoint=False)
    # slight down-glide in pitch
    freq = base_freq * (1 - 0.12 * (t / max(1e-6, dur_s)))
    sine = np.sin(2*np.pi*freq*t)
    # lowpassed noise
    noise = np.random.normal(0, 1, len(t)).astype(np.float32)
    # simple 1-pole low-pass
    lp = np.zeros_like(noise)
    alpha = 0.02
    for i in range(1, len(noise)):
        lp[i] = alpha*noise[i] + (1-alpha)*lp[i-1]
    body = 0.85*sine + 0.35*lp
    env = adsr(len(body), sr, a=0.08, d=0.18, s=0.6, r=0.35)
    thunder = (body * env).astype(np.float32)
    return np.tanh(thunder * 0.9).astype(np.float32)

# ---------- Audio builders ----------
def build_audio(bpm, loops, sr, style, rain_start, rain_end, jitter_ms,
                thunder, thunder_every):
    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat
    total_bars = loops * len(PROGRESSION)
    rng = np.random.default_rng(42)

    if style == "pad":
        # Pre-render sustained chords per bar
        cache = {}
        for name, notes in CHORDS_MIDI.items():
            parts = [synth_tone(midi_to_freq(m), bar_dur, sr, bright=False) for m in notes]
            chord = np.sum(np.stack(parts), axis=0)
            cache[name] = np.tanh(chord * 0.8).astype(np.float32)

        bars = []
        for bar in range(total_bars):
            name = PROGRESSION[bar % len(PROGRESSION)]
            bar_audio = cache[name].copy()
            # thunder?
            if thunder and (bar % max(1, thunder_every) == 0):
                th = synth_thunder(dur_s=min(0.9*bar_dur, 2.8), sr=sr, base_freq=55.0)
                # place at start, mix in
                L = min(len(bar_audio), len(th))
                bar_audio[:L] += th[:L] * 0.9
            bars.append(np.tanh(bar_audio * 1.0).astype(np.float32))

        full = np.concatenate(bars)
        # gentle fade
        f = int(sr*0.04)
        if f>0:
            ramp = np.linspace(0,1,f,endpoint=True).astype(np.float32)
            full[:f] *= ramp
            full[-f:] *= ramp[::-1]
        return full

    elif style == "arp":
        step_beats = 0.5
        step_dur = step_beats * sec_per_beat
        pattern_idx = [0,1,2,1, 3,2,1,0]
        note_cache = {}
        def tone_for(m):
            if m not in note_cache:
                note_cache[m] = synth_tone(midi_to_freq(m), step_dur, sr, bright=True)
            return note_cache[m]

        blocks = []
        for bar in range(total_bars):
            name = PROGRESSION[bar % len(PROGRESSION)]
            notes = CHORDS_MIDI[name]
            bar_audio = np.zeros(int(sr*bar_dur), dtype=np.float32)
            for s, idx in enumerate(pattern_idx):
                start = int(s * step_dur * sr)
                tone = tone_for(notes[idx])
                end = min(len(bar_audio), start + len(tone))
                bar_audio[start:end] += tone[:end-start]
            # thunder?
            if thunder and (bar % max(1, thunder_every) == 0):
                th = synth_thunder(dur_s=min(0.9*bar_dur, 2.4), sr=sr, base_freq=55.0)
                L = min(len(bar_audio), len(th))
                bar_audio[:L] += th[:L] * 0.9
            bar_audio = np.tanh(bar_audio * 1.1).astype(np.float32)
            blocks.append(bar_audio)

        full = np.concatenate(blocks)
        f = int(sr*0.02)
        if f>0:
            ramp = np.linspace(0,1,f,endpoint=True).astype(np.float32)
            full[:f] *= ramp
            full[-f:] *= ramp[::-1]
        return full

    elif style == "rain":
        steps_per_bar = 8
        base_step = bar_dur / steps_per_bar
        jitter_s = jitter_ms / 1000.0
        note_cache = {}
        def tone_for(m, dur):
            key = (m, round(dur*1000))
            if key not in note_cache:
                note_cache[key] = synth_tone(midi_to_freq(m), dur, sr, bright=True)
            return note_cache[key]

        blocks = []
        for bar in range(total_bars):
            # density ramp across bars
            alpha = bar / max(1, total_bars-1)
            density = (1-alpha)*rain_start + alpha*rain_end
            name = PROGRESSION[bar % len(PROGRESSION)]
            notes = CHORDS_MIDI[name]
            order = [0,1,2,1, 3,2,1,0]
            bar_audio = np.zeros(int(sr*bar_dur), dtype=np.float32)

            # subtle per-bar swell
            swell = 0.925 + 0.075 * math.sin(2*math.pi * bar / max(1,total_bars))

            for s in range(steps_per_bar):
                if rng.random() <= density:
                    idx = order[s % len(order)]
                    if rng.random() < 0.2:
                        idx = rng.integers(0, len(notes))
                    dur = base_step * rng.uniform(0.35, 0.6)
                    start = s * base_step + rng.uniform(-jitter_s, jitter_s)
                    start = max(0.0, min(bar_dur - 0.01, start))
                    tone = tone_for(notes[idx], dur) * swell
                    start_i = int(start * sr)
                    end_i = min(len(bar_audio), start_i + len(tone))
                    if end_i > start_i:
                        bar_audio[start_i:end_i] += tone[:end_i-start_i]
                    # occasional echo
                    if rng.random() < 0.12:
                        echo_start = start + base_step * rng.uniform(0.12, 0.25)
                        es_i = int(min(bar_dur - 0.01, echo_start) * sr)
                        ee_i = min(len(bar_audio), es_i + len(tone))
                        if ee_i > es_i:
                            bar_audio[es_i:ee_i] += (tone[:ee_i-es_i] * rng.uniform(0.5, 0.7))

            # thunder?
            if thunder and (bar % max(1, thunder_every) == 0):
                th = synth_thunder(dur_s=min(0.9*bar_dur, 2.4), sr=sr, base_freq=55.0)
                L = min(len(bar_audio), len(th))
                bar_audio[:L] += th[:L] * 0.9

            bar_audio = np.tanh(bar_audio * 1.15).astype(np.float32)
            blocks.append(bar_audio)

        full = np.concatenate(blocks)
        f = int(sr*0.02)
        if f>0:
            ramp = np.linspace(0,1,f,endpoint=True).astype(np.float32)
            full[:f] *= ramp
            full[-f:] *= ramp[::-1]
        return full

    else:
        raise ValueError("style must be 'pad', 'arp', or 'rain'")

def main():
    p = argparse.ArgumentParser(description="Generate MIDI/WAV pad, arp, or rain loop for 'Three Creatures' (B♭ minor).")
    p.add_argument("--bpm", type=int, default=72)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--sr", type=int, default=44100)
    p.add_argument("--style", choices=["pad","arp","rain"], default="pad")
    p.add_argument("--rain-start", type=float, default=0.8, help="start density (rain style)")
    p.add_argument("--rain-end", type=float, default=0.8, help="end density (rain style)")
    p.add_argument("--jitter-ms", type=float, default=15, help="max onset jitter in ms (rain style)")
    p.add_argument("--thunder", action="store_true", help="add thunder swells")
    p.add_argument("--thunder-every", type=int, default=2, help="add thunder every N bars")
    args = p.parse_args()

    # Build audio
    audio = build_audio(
        bpm=args.bpm, loops=args.loops, sr=args.sr, style=args.style,
        rain_start=args.rain_start, rain_end=args.rain_end, jitter_ms=args.jitter_ms,
        thunder=args.thunder, thunder_every=args.thunder_every
    )

    tag = args.style.upper()
    wav_name = f"Three_Creatures_Practice_{tag}.wav"
    write_wav(wav_name, audio, args.sr)

    # Write MIDI
    mid_name = f"Three_Creatures_Practice_{tag}.mid"
    write_midi(
        outfile=mid_name, bpm=args.bpm, loops=args.loops, style=args.style,
        rain_start=args.rain_start, rain_end=args.rain_end, jitter_ms=args.jitter_ms,
        thunder=args.thunder, thunder_every=args.thunder_every
    )

    print("Done ✅")

if __name__ == "__main__":
    main()