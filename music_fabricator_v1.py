#!/usr/bin/env python3
# music_fabricator_v1.py
# One-stop: modal Roman progressions -> WAV stems + MIDI (arp/pad/bass/perc) + optional style jobs (pad/arp/rain)

import os, math, re, wave, argparse, csv
import numpy as np
import yaml
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---- optional MIDI ----
try:
    from midiutil import MIDIFile
    HAVE_MIDI = True
except ImportError:
    HAVE_MIDI = False


def _require(obj, key, typ, where):
    if key not in obj:
        raise ValueError(f"Missing required key '{key}' in {where}")
    val = obj[key]
    if typ is not None and not isinstance(val, typ):
        raise ValueError(f"Key '{key}' in {where} must be {typ.__name__}, got {type(val).__name__}")
    return val

def _ensure_positive_int(val, name, where):
    if not isinstance(val, int) or val <= 0:
        raise ValueError(f"'{name}' in {where} must be a positive integer, got {val}")

def _validate_progression_list(lst, where):
    if not isinstance(lst, list):
        raise ValueError(f"'progression_A'/'progression_B' in {where} must be a list")
    for i, token in enumerate(lst, 1):
        if not isinstance(token, str):
            raise ValueError(f"Progression token #{i} in {where} must be string, got {type(token).__name__}")

def outpath(outdir, filename):
    return filename if os.path.isabs(filename) else os.path.join(outdir, filename)

def validate_job_schema(job, job_ix):
    where = f"job[{job_ix}] ('{job.get('name','unnamed')}')"

    # Required timing / audio basics
    bpm  = _require(job, "bpm",  (int, float), where);  # allow float BPM if you like
    sr   = _require(job, "sr",   int,          where)
    bars = _require(job, "bars", int,          where)
    _ensure_positive_int(int(bars), "bars", where)
    _ensure_positive_int(int(sr),   "sr",   where)

    # Sections OR legacy key/mode + progressions
    if "sections" in job and job["sections"]:
        if not isinstance(job["sections"], list):
            raise ValueError(f"'sections' in {where} must be a list")
        for s_ix, sec in enumerate(job["sections"], 1):
            sw = f"{where}.sections[{s_ix}]"
            key  = _require(sec, "key",  str, sw)
            mode = _require(sec, "mode", str, sw)
            # at least one A or B present
            if "progression_A" not in sec and "progression_B" not in sec:
                raise ValueError(f"{sw} must have 'progression_A' or 'progression_B'")
            if "progression_A" in sec: _validate_progression_list(sec["progression_A"], sw)
            if "progression_B" in sec: _validate_progression_list(sec["progression_B"], sw)
            # repeats is optional but if present must have ints
            reps = sec.get("repeats", {})
            if not isinstance(reps, dict):
                raise ValueError(f"{sw}.repeats must be a dict if present")
            for k in ("A","B"):
                if k in reps and (not isinstance(reps[k], int) or reps[k] < 0):
                    raise ValueError(f"{sw}.repeats.{k} must be a non-negative integer")
    else:
        # legacy path
        key  = _require(job, "key",  str, where)
        mode = _require(job, "mode", str, where)
        if "progression_A" not in job and "progression_B" not in job:
            raise ValueError(f"{where} must define 'progression_A' or 'progression_B'")
        if "progression_A" in job: _validate_progression_list(job["progression_A"], where)
        if "progression_B" in job: _validate_progression_list(job["progression_B"], where)

    # Exports / instrument_map are optional; do a light check if present
    ex = job.get("exports", {})
    if not isinstance(ex, dict):
        raise ValueError(f"'exports' in {where} must be a dict if present")

    im = job.get("instrument_map", {})
    if im and not isinstance(im, dict):
        raise ValueError(f"'instrument_map' in {where} must be a dict")
    # if perc layers present, check shapes
    layers = im.get("midi", {}).get("perc", {}).get("layers", {})
    if layers:
        if not isinstance(layers, dict):
            raise ValueError(f"{where}.instrument_map.midi.perc.layers must be a dict")
        for lname, lcfg in layers.items():
            if not isinstance(lcfg, dict):
                raise ValueError(f"{where}.instrument_map.midi.perc.layers.{lname} must be a dict")
            notes = lcfg.get("notes")
            if notes is not None and not (isinstance(notes, int) or (isinstance(notes, list) and all(isinstance(x,int) for x in notes))):
                raise ValueError(f"{where}.instrument_map.midi.perc.layers.{lname}.notes must be int or list[int]")


# =========================
# ====== Music theory =====
# =========================

NOTE_TO_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
              "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}

PC_TO_NAME_FLAT = {0:"C",1:"Db",2:"D",3:"Eb",4:"E",5:"F",6:"Gb",7:"G",8:"Ab",9:"A",10:"Bb",11:"B"}

MODES = {
    "ionian":     [0,2,4,5,7,9,11],
    "dorian":     [0,2,3,5,7,9,10],
    "phrygian":   [0,1,3,5,7,8,10],
    "lydian":     [0,2,4,6,7,9,11],
    "mixolydian": [0,2,4,5,7,9,10],
    "aeolian":    [0,2,3,5,7,8,10],
    "locrian":    [0,1,3,5,6,8,10]
}

# Accept "III(maj7)" OR "IIImaj7", also sus/add
ROMAN_RE = re.compile(
    r'^([b#]?)([ivIV]+)(?:\(([^)]*)\))?(?:\s*(maj7|m7|7|sus2|sus4|add9))?$'
)

def imap(job):
    return job.get("instrument_map", {})

def imidi(job):
    return imap(job).get("midi", {})

def iaudio(job):
    return imap(job).get("audio", {})

def perc_layer_cfg(job, layer):
    return imidi(job).get("perc", {}).get("layers", {}).get(layer, {})

def perc_audio_cfg(job, layer):
    return iaudio(job).get(layer, {})

def midi_ch(job, part, default):
    return int(imidi(job).get(part, {}).get("channel", default))

def midi_prog(job, part):
    return imidi(job).get(part, {}).get("program", None)

def send_program(mf, track, channel, program):
    # midiutil: addProgramChange(track, channel, time, program)
    if program is not None:
        mf.addProgramChange(track, channel, 0, int(program))


def pc_name(pc:int) -> str: return PC_TO_NAME_FLAT[pc % 12]

def parse_key_mode(tonic:str, mode_name:str):
    if tonic not in NOTE_TO_PC:
        raise ValueError(f"Bad tonic '{tonic}'")
    ml = mode_name.strip().lower()
    if ml not in MODES:
        raise ValueError(f"Unsupported mode '{mode_name}'")
    return NOTE_TO_PC[tonic], ml

def build_scale(tonic_pc:int, mode_l:str):
    return [(tonic_pc + step) % 12 for step in MODES[mode_l]]

def roman_to_degree(roman:str):
    m = ROMAN_RE.match(roman.strip())
    if not m: raise ValueError(f"Bad Roman '{roman}'")
    accidental, core, paren_mods, trailing = m.groups()
    valmap = {"I":0,"II":1,"III":2,"IV":3,"V":4,"VI":5,"VII":6}
    deg = valmap[core.upper()]
    is_upper = core.isupper()
    if accidental == "b": deg = (deg - 1) % 7
    elif accidental == "#": deg = (deg + 1) % 7
    mods = []
    if paren_mods:
        for token in paren_mods.split(","):
            token = token.strip().lower()
            if token: mods.append(token)
    if trailing:
        mods.append(trailing.strip().lower())
    return deg, is_upper, mods, accidental

def chord_from_roman(roman:str, scale_pcs:list[int]):
    deg, is_upper, mods, accidental = roman_to_degree(roman)
    # modal triad: degrees 1-3-5 on this mode
    d1 = scale_pcs[deg]
    d3 = scale_pcs[(deg+2)%7]
    d5 = scale_pcs[(deg+4)%7]
    chord = [d1, d3, d5]  # triad default

    # sus overrides 3rd
    if "sus2" in mods: chord[1] = scale_pcs[(deg+1)%7]
    if "sus4" in mods: chord[1] = scale_pcs[(deg+3)%7]

    # sevenths
    if "maj7" in mods:
        chord.append(scale_pcs[(deg+6)%7])
    elif "m7" in mods or "7" in mods:
        chord.append((d1 + 10) % 12)  # minor7/dominant7 color

    # add9
    if "add9" in mods:
        chord.append(scale_pcs[(deg+1)%7])

    return chord, roman  # keep label as given

def stack_to_midi(chord_pcs:list[int], root_pc:int, root_oct:int) -> list[int]:
    """Spread chord tones ascending near chosen octave."""
    out = []
    base = 12*(root_oct+1) + root_pc
    last = base - 12
    for pc in chord_pcs:
        m = 12*(root_oct+1) + pc
        while m <= last: m += 12
        out.append(m); last = m
    return out

# =========================
# ====== Synth Engine =====
# =========================

def write_wav_mono(path, audio, sr):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio)) or 1.0
    audio = (audio / peak) * 0.95
    pcm = (audio * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print("[WAV]", path)

def midi_to_hz(m): return 440.0*(2.0**((m-69)/12.0))

def adsr_env(n, sr, a=0.02, d=0.12, s=0.75, r=0.18):
    n = max(1, int(n))
    a_n = int(n*max(0.0,a)); d_n=int(n*max(0.0,d)); r_n=int(n*max(0.0,r))
    total = a_n+d_n+r_n
    if total > n:
        scale = n/total
        a_n = int(a_n*scale); d_n = int(d_n*scale); r_n = max(0, n-(a_n+d_n))
    s_n = max(0, n-(a_n+d_n+r_n))
    att=np.linspace(0,1,max(1,a_n),endpoint=True)
    dec=np.linspace(1,max(0.0,s),max(1,d_n),endpoint=True)
    sus=np.full(max(1,s_n),max(0.0,s),dtype=np.float32)
    rel=np.linspace(max(0.0,s),0,max(1,r_n),endpoint=True)
    env=np.concatenate([att,dec,sus,rel]).astype(np.float32)
    if len(env)>n: env=env[:n]
    if len(env)<n: env=np.pad(env,(0,n-len(env)),"edge")
    return env

def synth_tone(freq, dur_s, sr=44100, kind="pad", detune_cents=0.0, color="neutral"):
    n = max(1, int(sr*max(1e-6, dur_s)))
    t = np.linspace(0.0, dur_s, n, endpoint=False)
    f = float(freq) * (2.0**(detune_cents/1200.0))
    if kind == "pad":
        base = np.sin(2*np.pi*f*t)
        saw  = 2.0*(t*f - np.floor(0.5 + t*f))
        tone = 0.45*base + 0.55*saw
        env  = adsr_env(len(tone), sr, a=0.12, d=0.60, s=0.75, r=1.80)
    elif kind == "arp":
        sq   = np.sign(np.sin(2*np.pi*f*t))
        tone = 0.6*sq
        env  = adsr_env(len(tone), sr, a=0.01, d=0.08, s=0.60, r=0.12)
    elif kind == "bass":
        s1 = np.sin(2*np.pi*f*t); s2 = 0.2*np.sin(2*np.pi*(2*f)*t)
        tone = 0.85*s1 + 0.15*s2
        env  = adsr_env(len(tone), sr, a=0.01, d=0.08, s=0.80, r=0.20)
    elif kind == "tom":
        f_inst = f*(1.0 - 0.15*(t/max(1e-6,dur_s)))
        tone = np.sin(2*np.pi*f_inst*t)
        env  = adsr_env(len(tone), sr, a=0.05, d=0.20, s=0.60, r=0.40)
    elif kind == "noise":
        tone = np.random.normal(0.0, 1.0, n).astype(np.float32)
        env  = adsr_env(len(tone), sr, a=0.005, d=0.05, s=0.40, r=0.08)
    else:
        tone = np.sin(2*np.pi*f*t); env = adsr_env(len(tone), sr)
    # color tweak
    if color=="bright":
        tone += 0.2*np.sin(2*np.pi*(4*f)*t)
    elif color=="warm":
        tone *= 0.9
    tone = tone.astype(np.float32)
    peak = np.max(np.abs(tone)) or 1.0
    tone = tone/peak
    nmin = min(len(tone), len(env))
    if len(tone)!=len(env): tone=tone[:nmin]; env=env[:nmin]
    return (tone*env).astype(np.float32)

def iir_lowpass(x, alpha=0.02):
    y=np.zeros_like(x,dtype=np.float32); acc=0.0
    for i in range(len(x)):
        acc = alpha*x[i] + (1-alpha)*acc
        y[i]=acc
    return y

def iir_highpass(x, alpha=0.02):
    return (x - iir_lowpass(x,alpha)).astype(np.float32)

# =========================
# ======= PDF output ======
# =========================

def make_timeline_pdf(path, job, timeline):
    """
    Build a timeline PDF that reflects sections (key/mode + A/B repeats).
    If job['sections'] exists we print a table per section; otherwise a single table.
    """
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"{job['name']} — Timeline ({job.get('key','sections')} {job.get('mode','')})", styles["Heading1"]))
    story.append(Spacer(1, 12))

    def section_table(title, stacks):
        data = [["Bar", "Roman", "Chord Tones (PC names)"]]
        for i, (roman, pcs, _) in enumerate(stacks, start=1):
            data.append([str(i), roman, " – ".join(pc_name(pc) for pc in pcs)])
        tbl = Table(data, colWidths=[50, 160, 330])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('GRID',(0,0),(-1,-1),0.5,colors.black),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ]))
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(tbl)
        story.append(Spacer(1, 16))

    # If sections exist, render each section independently
    if "sections" in job and job["sections"]:
        running = 0
        for idx, sec in enumerate(job["sections"], start=1):
            sec_stacks = expand_section(sec)  # same logic used by build_timeline
            title = f"Section {idx} — {sec['key']} {sec['mode']}  |  A×{sec.get('repeats',{}).get('A',1)}  B×{sec.get('repeats',{}).get('B',0)}"
            section_table(title, sec_stacks)
            running += len(sec_stacks)
    else:
        # single-table fallback using the flattened timeline
        section_table(f"All bars — {job['key']} {job['mode']}", timeline)

    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(story)
    print("[PDF]", path)


def make_pdf(path, title, timeline, bars_per_page=None):
    """
    Timeline-based voicing PDF.
    Expects `timeline` as a flat list of (roman, pcs, root_pc) for the whole piece.
    Keeps the LH/RH voicing column, and numbers bars.
    Optionally splits across multiple tables/pages if bars_per_page is set (e.g., 32).
    """
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 12))

    def build_rows(tl_slice, start_bar_index):
        rows = [["Bar", "Roman", "Stack (PC)", "Suggested Piano Voicing (LH / RH)"]]
        bar_no = start_bar_index
        for roman, pcs, _root_pc in tl_slice:
            names = [pc_name(pc) for pc in pcs]  # pc_name already in your file
            stack_txt = " – ".join(names)
            lh = f"{names[0]} / {names[1] if len(names) > 1 else names[0]}"
            rh = " – ".join(names[2:]) if len(names) > 2 else "—"
            rows.append([str(bar_no), roman, stack_txt, f"LH: {lh} | RH: {rh}"])
            bar_no += 1
        return rows

    def add_table(rows, caption=None):
        if caption:
            story.append(Paragraph(f"<b>{caption}</b>", styles["Heading2"]))
        tbl = Table(rows, colWidths=[40, 110, 200, 250])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.whitesmoke),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.black),
            ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 16))

    # Single table or chunked pagination
    if bars_per_page and bars_per_page > 0:
        start = 0
        bar_index = 1
        page_idx = 1
        while start < len(timeline):
            end = min(start + bars_per_page, len(timeline))
            rows = build_rows(timeline[start:end], bar_index)
            add_table(rows, caption=f"Bars {bar_index}–{bar_index + (end-start) - 1} (Page {page_idx})")
            bar_index += (end - start)
            start = end
            page_idx += 1
    else:
        rows = build_rows(timeline, 1)
        add_table(rows)

    doc = SimpleDocTemplate(path, pagesize=A4)
    doc.build(story)
    print("[PDF]", path)


# =========================
# ====== MIDI writers =====
# =========================

def write_arp_midi(path, job, events):
    if not HAVE_MIDI: print("[MIDI] midiutil not installed; skip arp"); return
    bpm=job["bpm"]; ch=midi_ch(job,"arp",0); prog=midi_prog(job,"arp")
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm)
    send_program(mf,tr,ch,prog)
    for onset_sec, dur_sec, note, vel in events:
        mf.addNote(tr, ch, note, onset_sec*(bpm/60.0), max(0.05,dur_sec*(bpm/60.0)), vel)
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)

def write_pad_midi(path, job, chords):
    if not HAVE_MIDI: print("[MIDI] midiutil not installed; skip pad"); return
    bpm=job["bpm"]; bars=job["bars"]; ch=midi_ch(job,"pad",0); prog=midi_prog(job,"pad")
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm); send_program(mf,tr,ch,prog)
    time=0.0
    for bar in range(bars):
        for n in chords[bar % len(chords)]:
            mf.addNote(tr,ch,n, time, 4, 80)
        time+=4
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)

def write_bass_midi(path, job, root_notes):
    if not HAVE_MIDI: print("[MIDI] midiutil not installed; skip bass"); return
    bpm=job["bpm"]; bars=job["bars"]; ch=midi_ch(job,"bass",0); prog=midi_prog(job,"bass")
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm); send_program(mf,tr,ch,prog)
    time=0.0
    for bar in range(bars):
        root = root_notes[bar % len(root_notes)]
        mf.addNote(tr,ch, root, time+0, 2, 90)
        mf.addNote(tr,ch, root, time+2, 2, 90)
        time+=4
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)


def pick_note_from_pool(pool):
    # pool can be [note,...] or single int
    if isinstance(pool, list) and pool:
        return pool[np.random.randint(0, len(pool))]
    if isinstance(pool, int):
        return pool
    return 42  # fallback to closed HH

def write_perc_midi(path, job, events):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skip perc"); return
    bpm=job["bpm"]
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm)
    ch = int(imidi(job).get("perc",{}).get("channel", 9))  # default GM drums
    # no program change on GM drums channel usually; we skip

    for layer, onset_sec, dur_sec, vel in events:
        notes_cfg = perc_layer_cfg(job, layer).get("notes", None)
        # sensible GM defaults:
        if notes_cfg is None:
            notes_cfg = 42 if layer=="brush" else (70 if layer=="shaker" else 41)
        gm_note = pick_note_from_pool(notes_cfg)
        onset_beats = onset_sec * (bpm/60.0)
        dur_beats = max(0.05, dur_sec * (bpm/60.0))
        mf.addNote(tr, ch, gm_note, onset_beats, dur_beats, max(1,min(127,vel)))

    with open(path,"wb") as f:
        mf.writeFile(f)
    print("[MIDI]", path)


# =========================
# = Practice/Arp/Pad/Bass =
# =========================
def parse_side_with_scale(roman_list, scale_pcs):
    out=[]
    for r in roman_list:
        pcs, label = chord_from_roman(r, scale_pcs)
        root_pc = pcs[0]
        out.append((label, pcs, root_pc))
    return out

def expand_section(sec):
    """Return stacks for this section, after repeats of A/B."""
    tonic_pc, mode_l = parse_key_mode(sec["key"], sec["mode"])
    scale = build_scale(tonic_pc, mode_l)
    A = parse_side_with_scale(sec.get("progression_A", []), scale)
    B = parse_side_with_scale(sec.get("progression_B", []), scale)
    rA = int(sec.get("repeats", {}).get("A", 1))
    rB = int(sec.get("repeats", {}).get("B", 0))
    stacks=[]
    for _ in range(max(0, rA)): stacks += A
    for _ in range(max(0, rB)): stacks += B
    return stacks

def build_timeline(job):
    """
    If job['sections'] present, flatten them into a single stacks timeline;
    otherwise fallback to old top-level key/mode/progressions.
    """
    stacks=[]
    if "sections" in job and job["sections"]:
        for sec in job["sections"]:
            stacks += expand_section(sec)
    else:
        tonic_pc, mode_l = parse_key_mode(job["key"], job["mode"])
        scale = build_scale(tonic_pc, mode_l)
        A = parse_side_with_scale(job.get("progression_A", []), scale)
        B = parse_side_with_scale(job.get("progression_B", []), scale)
        stacks = A + B
    return stacks

def stacks_from_config(job):
    tonic_pc, mode_l = parse_key_mode(job["key"], job["mode"])
    scale = build_scale(tonic_pc, mode_l)
    def parse_side(roman_list):
        out=[]
        for r in roman_list:
            pcs, label = chord_from_roman(r, scale)
            root_pc = pcs[0]
            out.append((label, pcs, root_pc))
        return out
    stacksA = parse_side(job.get("progression_A",[]))
    stacksB = parse_side(job.get("progression_B",[]))
    return stacksA, stacksB

def chord_stacks_to_audio_midi_refs(stacks, root_oct_audio, color):
    """Returns (pad_midi_chords, root_midi_notes_for_bass, pad_audio_builder)"""
    pad_midi_chords=[]; roots=[]
    for _, pcs, root_pc in stacks:
        mids = stack_to_midi(pcs, root_pc, root_oct_audio)
        pad_midi_chords.append(mids)
        roots.append(12*(2+1) + root_pc)  # default bass octave 2
    return pad_midi_chords, roots
def timeline_to_pad_bass_refs(job, timeline):
    pad_chords=[]; bass_roots=[]
    for _, pcs, root_pc in timeline:
        pad_chords.append(stack_to_midi(pcs, root_pc, job.get("root_oct_audio",3)))
        bass_roots.append(12*(job.get("bass_oct",2)+1) + pcs[0])
    return pad_chords, bass_roots

def build_arp_events_from_timeline(job, timeline):
    bpm=job["bpm"]; bars=job["bars"]
    sec_per_beat=60.0/bpm; bar_dur=4*sec_per_beat; step=sec_per_beat/4.0
    events=[]
    for bar in range(bars):
        _, pcs, root_pc = timeline[bar % len(timeline)]
        mids = stack_to_midi(pcs, root_pc, job.get("root_oct_audio",3))
        arp = mids[-3:] if len(mids)>=3 else mids
        pat=[0,1,2,1, 2,1,0,1]
        for s,idx in enumerate(pat):
            if not arp: break
            onset = bar*bar_dur + s*step
            events.append((onset, step*0.9, arp[idx%len(arp)], 75))
    return events


def render_practice_wav(path, job, timeline):
    sr   = job["sr"]
    bpm  = job["bpm"]
    bars = job["bars"]
    color = job.get("color","neutral")

    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat
    total = np.zeros(int(sr * bar_dur * bars), dtype=np.float32)
    rng = np.random.default_rng(42)

    for bar in range(bars):
        _, pcs, root_pc = timeline[bar % len(timeline)]
        bar_sig = np.zeros(int(sr * bar_dur), dtype=np.float32)

        # pad
        mids = stack_to_midi(pcs, root_pc, job.get("root_oct_audio", 3))
        for i, m in enumerate(mids):
            det  = (-8 + 16 * rng.random())
            tone = synth_tone(midi_to_hz(m), bar_dur, sr, kind="pad", detune_cents=det, color=color)
            bar_sig[:len(tone)] += tone * (0.22 if i == 0 else 0.18)

        # bass pulses
        root_m = 12 * (job.get("bass_oct", 2) + 1) + pcs[0]
        for beat in (0, 2):
            start = int(beat * sec_per_beat * sr)
            dur   = sec_per_beat * 0.9
            tb    = synth_tone(midi_to_hz(root_m), dur, sr, kind="bass", color=color)
            end   = min(len(bar_sig), start + len(tb))
            bar_sig[start:end] += tb[:end - start] * 0.35

        # arp
        step = sec_per_beat / 4.0
        arp_notes = mids[-3:] if len(mids) >= 3 else mids
        pattern = [0,1,2,1, 2,1,0,1]
        for s, idx in enumerate(pattern):
            if not arp_notes: break
            start = int(s * step * sr)
            dur   = step * 0.9
            ta    = synth_tone(midi_to_hz(arp_notes[idx % len(arp_notes)]), dur, sr, kind="arp", color=color)
            ta    = iir_lowpass(ta, 0.04)
            end   = min(len(bar_sig), start + len(ta))
            bar_sig[start:end] += ta[:end - start] * 0.18

        s0 = int(bar * bar_dur * sr)
        total[s0:s0 + len(bar_sig)] += bar_sig

    write_wav_mono(path, total, sr)



# =========================
# ====== Percussion =======
# =========================

def build_perc_events(job):
    bpm=job["bpm"]; sr=job["sr"]; bars=job["bars"]
    sec_per_beat=60.0/bpm; bar_dur=4*sec_per_beat
    rng=np.random.default_rng(99); ev=[]

    # Defaults
    def layer_density(layer, default):
        return float(perc_layer_cfg(job, layer).get("density", default))
    def layer_jitter(layer, default_ms):
        return float(perc_layer_cfg(job, layer).get("jitter_ms", default_ms))/1000.0
    def vel_minmax(layer, dflt=(10,25)):
        cfg=perc_layer_cfg(job, layer)
        return int(cfg.get("vel_min", dflt[0])), int(cfg.get("vel_max", dflt[1]))

    # BRUSH + SHAKER: 8 slots per bar
    for bar in range(bars):
        t0 = bar*bar_dur

        # brush ghosts
        dens = layer_density("brush", 0.70)
        jitter = layer_jitter("brush", 15)
        vmin,vmax = vel_minmax("brush",(12,28))
        for slot in range(8):
            if rng.random() <= dens:
                start = slot*(bar_dur/8.0)+rng.uniform(-jitter, jitter)
                start = max(0.0, min(bar_dur - 1.0/sr, start))
                dur = rng.uniform(0.04, 0.08)
                vel = int(rng.uniform(vmin, vmax))
                ev.append(("brush", t0+start, dur, vel))

        # shaker
        dens = layer_density("shaker", 0.55)
        jitter = layer_jitter("shaker", 10)
        vmin,vmax = vel_minmax("shaker",(40,65))
        for slot in range(8):
            if rng.random() <= dens:
                start = slot*(bar_dur/8.0)+rng.uniform(-jitter, jitter)
                start = max(0.0, min(bar_dur - 1.0/sr, start))
                dur = rng.uniform(0.05, 0.09)
                vel = int(rng.uniform(vmin, vmax))
                ev.append(("shaker", t0+start, dur, vel))

        # tom swell, every N bars
        every = int(perc_layer_cfg(job, "tom").get("every_bars", 4))
        if every>0 and (bar % every == 0):
            swell_beats = float(perc_layer_cfg(job,"tom").get("swell_beats", 2.0))
            start = (4 - swell_beats) * sec_per_beat
            dur = swell_beats * sec_per_beat * 0.8
            vmin,vmax = vel_minmax("tom",(48,64))
            vel = int((vmin+vmax)//2)
            ev.append(("tom", t0+start, dur, vel))
    return ev


def render_perc_wav(path, job, events):
    sr=job["sr"]; bpm=job["bpm"]; bars=job["bars"]
    sec_per_beat=60.0/bpm; bar_dur=4*sec_per_beat
    total=np.zeros(int(sr*bar_dur*bars),dtype=np.float32)

    for layer, onset_sec, dur_sec, vel in events:
        si=int(onset_sec*sr)
        if si>=len(total) or dur_sec<=0: continue
        cfg = perc_audio_cfg(job, layer)
        synth = cfg.get("synth", "noise")
        lp_alpha = float(cfg.get("lp_alpha", 0.03 if layer=="shaker" else 0.06))
        hp_alpha = float(cfg.get("hp_alpha", 0.0 if layer!="brush" else 0.008))
        gain = float(cfg.get("gain", 0.15 if layer=="shaker" else (0.12 if layer=="brush" else 0.6)))
        if layer=="tom":
            freq = float(cfg.get("freq_hz", 70.0))
            tone = synth_tone(freq, dur_sec, sr, kind="tom")
            tone = iir_lowpass(tone, lp_alpha)
        else:
            tone = synth_tone(0, dur_sec, sr, kind="noise")
            tone = iir_lowpass(tone, lp_alpha)
            if hp_alpha>0: tone = iir_highpass(tone, hp_alpha)
        amp = gain * (vel/127.0)
        tone = (tone * amp).astype(np.float32)
        ei = min(len(total), si + len(tone))
        if ei>si: total[si:ei] += tone[:ei-si]

    write_wav_mono(path, total, sr)


# =========================
# ===== Style engines =====
# =========================

def render_style(job):
    """
    Optional make_tracks_v5-like style render.
    Returns (audio, events_for_midi_or_None, crescendo_rows_or_None)
    """
    style = job.get("style")
    if not style:
        return None, None, None

    sr  = job["sr"]
    bpm = job["bpm"]
    bars = job["bars"]
    color = job.get("color", "neutral")
    sec_per_beat = 60.0 / bpm
    bar_dur = 4 * sec_per_beat

    timeline = build_timeline(job)
    plen = len(timeline)

    # MIDI chords list (per bar)
    midi_chords = []
    for _, pcs, root_pc in timeline:
        midi_chords.append(stack_to_midi(pcs, root_pc, job.get("root_oct_audio", 3)))

    if style == "pad":
        blocks = []
        for bar in range(bars):
            notes = midi_chords[bar % plen]
            barbuf = np.zeros(int(sr * bar_dur), dtype=np.float32)
            for i, n in enumerate(notes):
                barbuf += synth_tone(midi_to_hz(n), bar_dur, sr, kind="pad", color=color) * (0.22 if i == 0 else 0.18)
            blocks.append(np.tanh(barbuf * 0.8))
        return np.concatenate(blocks), None, None

    if style == "arp":
        blocks = []
        step = sec_per_beat / 4.0
        pattern = [0, 1, 2, 1, 2, 1, 0, 1]
        for bar in range(bars):
            notes = midi_chords[bar % plen]
            arp = notes[-3:] if len(notes) >= 3 else notes
            barbuf = np.zeros(int(sr * bar_dur), dtype=np.float32)
            for s, idx in enumerate(pattern):
                start = int(s * step * sr)
                dur = step * 0.9
                ta = synth_tone(midi_to_hz(arp[idx % len(arp)]), dur, sr, kind="arp", color=color)
                ta = iir_lowpass(ta, 0.04)
                end = min(len(barbuf), start + len(ta))
                barbuf[start:end] += ta[:end - start] * 0.18
            blocks.append(np.tanh(barbuf * 1.1))
        return np.concatenate(blocks), None, None

    if style == "rain":
        rain = job.get("rain", {})
        start = float(rain.get("start", 0.8))
        end   = float(rain.get("end",   0.8))
        jitter_ms = float(rain.get("jitter_ms", 15))
        thunder   = bool(rain.get("thunder", False))
        thunder_every = int(rain.get("thunder_every", 2))

        rng = np.random.default_rng(42)
        steps = 8
        base_step = bar_dur / steps
        jitter_s = jitter_ms / 1000.0

        blocks = []
        rows = []
        events = []
        for bar in range(bars):
            alpha = bar / max(1, bars - 1)
            density = (1 - alpha) * start + alpha * end
            swell = 0.925 + 0.075 * math.sin(2 * math.pi * bar / max(1, bars))
            notes = midi_chords[bar % plen]
            order = [0, 1, 2, 1, 3, 2, 1, 0]
            barbuf = np.zeros(int(sr * bar_dur), dtype=np.float32)

            for s in range(steps):
                if rng.random() <= density:
                    idx = order[s % len(order)]
                    dur = base_step * rng.uniform(0.35, 0.60)
                    start_t = s * base_step + rng.uniform(-jitter_s, jitter_s)
                    start_t = max(0.0, min(bar_dur - (1.0 / sr), start_t))
                    tone = synth_tone(midi_to_hz(notes[idx % len(notes)]), dur, sr, kind="arp", color=color) * swell
                    si = int(start_t * sr)
                    ei = min(len(barbuf), si + len(tone))
                    if ei > si:
                        barbuf[si:ei] += tone[:ei - si]
                        vel = int(75 * swell * np.random.uniform(0.85, 1.1))
                        vel = max(30, min(115, vel))
                        events.append((bar * bar_dur + start_t, dur, notes[idx % len(notes)], vel))

            if thunder and bar % max(1, thunder_every) == 0:
                th = synth_tone(55.0, min(bar_dur * 0.9, 2.4), sr, kind="tom", color=color)
                L = min(len(barbuf), len(th))
                barbuf[:L] += th[:L] * 0.9

            blocks.append(np.tanh(barbuf * 1.15))
            rows.append({"bar": bar + 1, "density": density, "swell": swell})

        return np.concatenate(blocks), events, rows

    return None, None, None


# =========================
# ========= MAIN ==========
# =========================

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()

    cfg=yaml.safe_load(open(args.config, "r"))
    outdir=cfg.get("outdir",".")
    defaults=cfg.get("defaults",{})
    jobs=cfg.get("jobs",[])
    if not jobs: raise SystemExit("No jobs found in config")


    for j_ix, job_raw in enumerate(jobs, 1):
        job = {**defaults, **job_raw}
        # validate early

        validate_job_schema(job, j_ix)
        os.makedirs(outdir, exist_ok=True)


        # Build stacks (Theme A + Theme B)
        timeline = build_timeline(job)

        # PDF exports (timeline + voicings)
        if job.get("export_pdf", defaults.get("export_pdf", False)) or job.get("exports", {}).get(
                "pdf_timeline") or job.get("exports", {}).get("pdf_voicings"):
            ex = job.get("exports", {})
            pdf_timeline = ex.get("pdf_timeline", f"{job['name']}_chords_timeline.pdf")
            pdf_voicings = ex.get("pdf_voicings", f"{job['name']}_chords_voicings.pdf")

            # timeline (sections, repeats, key changes)
            make_timeline_pdf(outpath(outdir, pdf_timeline), job, timeline)

            # voicings (LH / RH) over the same flattened timeline
            make_pdf(
                outpath(outdir, pdf_voicings),
                f"{job['name']} — Voicings ({job.get('key', 'sections')} {job.get('mode', '')})",
                timeline,
                bars_per_page=32
            )

        # Practice WAV
        if job.get("export_wav", defaults.get("export_wav", False)) and job.get("exports", {}).get("practice_wav"):
            render_practice_wav(os.path.join(outdir, job["exports"]["practice_wav"]), job, timeline)

        # Percussion events + WAV + MIDI
        perc_events = None
        if job.get("export_wav", defaults.get("export_wav", False)) and job.get("exports", {}).get("percussion_wav"):
            perc_events = build_perc_events(job)
            render_perc_wav(os.path.join(outdir, job["exports"]["percussion_wav"]), job, perc_events)

        if job.get("export_midi", defaults.get("export_midi", False)):
            ex = job.get("exports", {})
            if ex.get("midi_arp"):
                arp_events = build_arp_events_from_timeline(job, timeline)
                write_arp_midi(os.path.join(outdir, ex["midi_arp"]), job, arp_events)
            if ex.get("midi_pad"):
                pad_chords, _ = timeline_to_pad_bass_refs(job, timeline)
                write_pad_midi(os.path.join(outdir, ex["midi_pad"]), job, pad_chords)
            if ex.get("midi_bass"):
                _, bass_roots = timeline_to_pad_bass_refs(job, timeline)
                write_bass_midi(os.path.join(outdir, ex["midi_bass"]), job, bass_roots)
            if ex.get("midi_perc"):
                if perc_events is None: perc_events = build_perc_events(job)
                write_perc_midi(os.path.join(outdir, ex["midi_perc"]), job, perc_events)

        # Style render (pad/arp/rain) using the timeline
        if job.get("style"):
            audio, events, rows = render_style(job)
            ex = job.get("exports", {})
            if audio is not None and ex.get("style_wav"):
                write_wav_mono(os.path.join(outdir, ex["style_wav"]), audio, job["sr"])
            if events is not None and ex.get("style_mid") and HAVE_MIDI:
                mf = MIDIFile(1);
                tr = 0
                mf.addTempo(tr, 0, job["bpm"])

                # Prefer explicit "style" channel/program; else fall back to arp, then pad
                style_ch = midi_ch(job, "style", midi_ch(job, "arp", midi_ch(job, "pad", 0)))
                style_prog = midi_prog(job, "style")
                if style_prog is None:
                    style_prog = midi_prog(job, "arp")
                    if style_prog is None:
                        style_prog = midi_prog(job, "pad")

                send_program(mf, tr, style_ch, style_prog)

                for onset, dur, note, vel in events:
                    mf.addNote(tr, style_ch,
                               note,
                               onset * (job["bpm"] / 60.0),
                               max(0.05, dur * (job["bpm"] / 60.0)),
                               vel)

                with open(os.path.join(outdir, ex["style_mid"]), "wb") as f:
                    mf.writeFile(f)
                print("[MIDI]", ex["style_mid"])
            if rows is not None and ex.get("crescendo_csv"):
                with open(os.path.join(outdir, ex["crescendo_csv"]), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["bar","density","swell"])
                    w.writeheader(); w.writerows(rows)
                print("[CSV]", ex["crescendo_csv"])

        # Style render (pad/arp/rain) still works — just use timeline in render_style


if __name__=="__main__":
    main()
