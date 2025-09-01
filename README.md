# Music Fabricator v1

**Music Fabricator** is a config-driven tool for generating chord sheets, MIDI, and WAV stems from modal chord progressions written in Roman numerals.

It combines:
- Chord theory: modal scales (Ionian → Locrian), Roman numerals with modifiers (`maj7`, `m7`, `7`, `sus2`, `sus4`, `add9`).
- PDF exports: chord voicings (LH/RH) and timeline views.
- Audio exports: mono practice WAVs (pad+bass+arp), rolling percussion stems, optional style renders (`pad`, `arp`, `rain`).
- MIDI exports: arp lines, block pads, bass pulses, percussion (mapped to GM or per-job maps).

---

## Features

- **Roman numeral parsing**
  - Triads by default, with optional `maj7`, `m7`, `7`, `sus2`, `sus4`, `add9`.
  - Accepts `IIImaj7` or `III(maj7)`.

- **Modes & keys**
  - All 7 modes supported: Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian.
  - Key changes per job (`sections:`).

- **Section repeats**
  - Play Theme A/B multiple times, e.g. `repeats: { A: 2, B: 1 }`.

- **Instrument mapping**
  - Override MIDI channels/programs for arp/pad/bass.
  - Custom percussion note pools, velocity ranges, density, jitter, swell settings.
  - Audio synth config per layer (noise, tom, filters, gains).

- **Outputs**
  - PDFs:
    - Timeline PDF (by sections, repeats, key changes).
    - Voicing PDF (LH/RH piano voicings).
  - WAVs:
    - Practice bed (pad+bass+arp).
    - Rolling percussion stem.
    - Style render (pad/arp/rain).
  - MIDI:
    - Arp 16ths, Pad block chords, Bass pulses, Percussion hits (GM or custom).
    - Rain style MIDI matches stochastic drops, with crescendo CSV if requested.

---

## Installation

Clone the repo and install dependencies:

```bash
git clone <your-repo>
cd <your-repo>
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- [numpy](https://numpy.org/)
- [reportlab](https://pypi.org/project/reportlab/) (PDF generation)
- [midiutil](https://pypi.org/project/MIDIUtil/) (MIDI export, optional)
- [PyYAML](https://pypi.org/project/PyYAML/) (YAML configs)

Example:

```bash
pip install numpy reportlab midiutil pyyaml
```

---

## Usage

Run the tool with a YAML config:

```bash
python music_fabricator_v1.py --config songs.yaml
```

Outputs are saved in the `outdir` defined in the config (default: `renders/`).

---

## Configuration

All jobs live in `songs.yaml`.

### Top-level keys

```yaml
outdir: renders

defaults:
  sr: 44100          # sample rate
  bpm: 88            # tempo
  bars: 11           # total bars
  color: neutral     # pad timbre: neutral|warm|bright
  export_pdf: true   # export PDFs by default
  export_wav: true   # export WAVs by default
  export_midi: true  # export MIDI by default
```

### Job types

#### 1. Section-based jobs (with repeats, key changes)

```yaml
- name: blade_runner_piece
  sections:
    - key: D
      mode: Dorian
      progression_A: [i(add9), VII, IIImaj7, IVsus2]
      progression_B: [bVIImaj7, IV, IVsus2, i(add9)]
      repeats: { A: 2, B: 1 }
    - key: F
      mode: Dorian
      progression_A: [i(add9), VII, III, IVsus2]
      repeats: { A: 1, B: 0 }
  bpm: 88
  bars: 11
  exports:
    pdf_timeline: blade_runner_chords_timeline.pdf
    pdf_voicings: blade_runner_chords_voicings.pdf
    practice_wav: blade_runner_practice.wav
    percussion_wav: blade_runner_percussion.wav
    midi_arp: blade_runner_arp.mid
    midi_pad: blade_runner_pad.mid
    midi_bass: blade_runner_bass.mid
    midi_perc: blade_runner_percussion.mid
```

#### 2. Style jobs (pad / arp / rain)

```yaml
- name: three_creatures_rain
  sections:
    - key: Bb
      mode: Aeolian
      progression_A: [i(add9), VII(maj7), v(add9), VIIsus2]
      repeats: { A: 3, B: 0 }
  bpm: 72
  bars: 12
  style: rain
  rain:
    start: 0.6
    end: 0.95
    jitter_ms: 18
    thunder: true
    thunder_every: 2
  exports:
    style_wav: three_creatures_RAIN.wav
    style_mid: three_creatures_RAIN.mid
    crescendo_csv: three_creatures_crescendo.csv
```

---

## Instrument Map

Optional per-job overrides:

```yaml
instrument_map:
  midi:
    arp:  { channel: 0, program: 81 }   # Lead 2 (sawtooth)
    pad:  { channel: 1, program: 91 }   # Pad 3 (polysynth)
    bass: { channel: 2, program: 38 }   # Synth bass 1
    perc:
      channel: 9   # GM channel 10
      layers:
        brush:  { notes: [42,44], vel_min: 12, vel_max: 28, density: 0.70, jitter_ms: 15 }
        shaker: { notes: [70],    vel_min: 40, vel_max: 65, density: 0.55, jitter_ms: 10 }
        tom:    { notes: [43,41], vel_min: 48, vel_max: 64, every_bars: 4, swell_beats: 2.0 }

  audio:
    brush:  { synth: noise, lp_alpha: 0.06, hp_alpha: 0.008, gain: 0.12 }
    shaker: { synth: noise, lp_alpha: 0.03, gain: 0.15 }
    tom:    { synth: tom, freq_hz: 70.0, lp_alpha: 0.02, gain: 0.6 }
```

---

## Outputs

For each job you can get:

- `*_timeline.pdf` → chord progression per section (timeline view).
- `*_voicings.pdf` → LH/RH voicing suggestions.
- `*_practice.wav` → pad+bass+arp mono practice loop.
- `*_percussion.wav` → rolling brush/shaker/tom stem.
- `*_arp.mid` → 16th-note arp line.
- `*_pad.mid` → block chords (1 chord per bar).
- `*_bass.mid` → pulses on beats 1 & 3.
- `*_percussion.mid` → GM/kit-mapped percussion hits.
- `*_RAIN.wav` + `*_RAIN.mid` + `*_crescendo.csv` → rain style with stochastic density.

---

## Tips

- Use timeline PDF when composing (sections, key changes).
- Use voicing PDF when practicing at piano.
- For “rain” style, start with `rain_start: 0.6`, `rain_end: 0.95` for a sparse-to-dense drizzle.
- Percussion MIDI defaults to GM: 42 (closed HH/brush), 70 (maracas), 41 (floor tom). Override with `instrument_map` as needed.
- All WAVs are mono to give you freedom in DAW panning. Add your own stereo spread/reverb.

---

## License

This project is licensed under the [MIT License](LICENSE).

