#!/usr/bin/env python3
# music_fabricator_v1.py
# One-stop: modal Roman progressions -> WAV stems + MIDI (arp/pad/bass/perc) + optional style jobs (pad/arp/rain)

import os, math, re, wave, argparse, csv, midiutil
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

def make_pdf(path, title, stacksA, stacksB):
    styles=getSampleStyleSheet(); story=[]
    story.append(Paragraph(title, styles["Heading1"]))
    story.append(Spacer(1, 12))

    def section(label, stacks):
        data=[["Roman", "Stack (PC)", "Suggested Piano Voicing (LH / RH)"]]
        for roman, pcs, root_pc in stacks:
            names = [pc_name(pc) for pc in pcs]
            stack_txt = " – ".join(names)
            lh = f"{names[0]} / {names[1] if len(names)>1 else names[0]}"
            rh = " – ".join(names[2:]) if len(names)>2 else "—"
            data.append([roman, stack_txt, f"LH: {lh} | RH: {rh}"])
        tbl=Table(data, colWidths=[120,200,220])
        tbl.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('GRID',(0,0),(-1,-1),0.5,colors.black)
        ]))
        story.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))
        story.append(tbl); story.append(Spacer(1,16))

    section("Theme A", stacksA); section("Theme B", stacksB)
    doc=SimpleDocTemplate(path, pagesize=A4); doc.build(story)
    print("[PDF]", path)

# =========================
# ====== MIDI writers =====
# =========================

def write_arp_midi(path, events, bpm):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skip arp"); return
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm); ch=0
    for onset_sec, dur_sec, note, vel in events:
        mf.addNote(tr,ch,note, onset_sec*(bpm/60.0), max(0.05,dur_sec*(bpm/60.0)), vel)
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)

def write_pad_midi(path, chords, bpm, bars):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skip pad"); return
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm); ch=0
    time=0.0
    for bar in range(bars):
        notes = chords[bar % len(chords)]
        for n in notes: mf.addNote(tr,ch,n, time, 4, 80)
        time += 4
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)

def write_bass_midi(path, root_notes, bpm, bars):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skip bass"); return
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm); ch=0
    time=0.0
    for bar in range(bars):
        root = root_notes[bar % len(root_notes)]
        # beats 1 & 3 pulses
        mf.addNote(tr,ch, root, time+0, 2, 90)
        mf.addNote(tr,ch, root, time+2, 2, 90)
        time += 4
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)

def write_perc_midi(path, events, bpm):
    if not HAVE_MIDI:
        print("[MIDI] midiutil not installed; skip perc"); return
    mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,bpm); ch=9  # GM ch10
    for onset_sec, dur_sec, gm_note, vel in events:
        mf.addNote(tr,ch, gm_note, onset_sec*(bpm/60.0), max(0.05,dur_sec*(bpm/60.0)), vel)
    with open(path,"wb") as f: mf.writeFile(f); print("[MIDI]", path)

# =========================
# = Practice/Arp/Pad/Bass =
# =========================

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

def build_arp_events(stacksA, stacksB, bpm, bars, root_oct_audio):
    sec_per_beat = 60.0/bpm; step = sec_per_beat/4.0
    prog = stacksA + stacksB; plen=len(prog)
    events=[]
    for bar in range(bars):
        _, pcs, root_pc = prog[bar % plen]
        mids = stack_to_midi(pcs, root_pc, root_oct_audio)
        arp_notes = mids[-3:] if len(mids)>=3 else mids
        pattern=[0,1,2,1, 2,1,0,1]
        for s, idx in enumerate(pattern):
            onset = bar*4*sec_per_beat + s*step
            dur = step*0.9
            if arp_notes:
                events.append((onset, dur, arp_notes[idx%len(arp_notes)], 75))
    return events

def render_practice_wav(path, job, stacksA, stacksB):
    sr=job["sr"]; bpm=job["bpm"]; bars=job["bars"]; color=job.get("color","neutral")
    sec_per_beat=60.0/bpm; bar_dur=4*sec_per_beat
    total=np.zeros(int(sr*bar_dur*bars),dtype=np.float32)
    prog=stacksA+stacksB; plen=len(prog)
    rng=np.random.default_rng(42)
    for bar in range(bars):
        _, pcs, root_pc = prog[bar%plen]
        bar_sig=np.zeros(int(sr*bar_dur),dtype=np.float32)
        # pad
        mids = stack_to_midi(pcs, root_pc, job.get("root_oct_audio",3))
        for i,m in enumerate(mids):
            det=(-8 + 16*rng.random())
            tone = synth_tone(midi_to_hz(m), bar_dur, sr, kind="pad", detune_cents=det, color=color)
            bar_sig[:len(tone)] += tone * (0.22 if i==0 else 0.18)
        # bass (beats 1 & 3)
        root_m = 12*(job.get("bass_oct",2)+1) + pcs[0]
        for beat in (0,2):
            start=int(beat*sec_per_beat*sr); dur=sec_per_beat*0.9
            tb=synth_tone(midi_to_hz(root_m), dur, sr, kind="bass", color=color)
            end=min(len(bar_sig), start+len(tb))
            bar_sig[start:end] += tb[:end-start]*0.35
        # arp
        step=sec_per_beat/4.0; arp_notes=mids[-3:] if len(mids)>=3 else mids
        pat=[0,1,2,1, 2,1,0,1]
        for s,idx in enumerate(pat):
            start=int(s*step*sr); dur=step*0.9
            if arp_notes:
                ta=synth_tone(midi_to_hz(arp_notes[idx%len(arp_notes)]), dur, sr, kind="arp", color=color)
                ta=iir_lowpass(ta, alpha=0.04)
                end=min(len(bar_sig), start+len(ta))
                bar_sig[start:end]+=ta[:end-start]*0.18
        s0=int(bar*bar_dur*sr); s1=s0+len(bar_sig)
        total[s0:s1]+=bar_sig
    write_wav_mono(path,total,sr)

# =========================
# ====== Percussion =======
# =========================

def build_perc_events(job):
    bpm=job["bpm"]; sr=job["sr"]; bars=job["bars"]
    sec_per_beat=60.0/bpm; bar_dur=4*sec_per_beat
    rng=np.random.default_rng(99); events=[]
    for bar in range(bars):
        t0 = bar*bar_dur
        # brush ghosts -> Closed HH (42)
        for slot in range(8):
            if rng.random()<0.7:
                start = slot*(bar_dur/8.0)+rng.uniform(-0.015,0.015)
                start=max(0.0,min(bar_dur-1.0/sr,start))
                events.append((t0+start, rng.uniform(0.04,0.08), 42, int(rng.uniform(10,25))))
        # shaker -> Maracas (70)
        for slot in range(8):
            if rng.random()<0.55:
                start = slot*(bar_dur/8.0)+rng.uniform(-0.01,0.01)
                start=max(0.0,min(bar_dur-1.0/sr,start))
                events.append((t0+start, rng.uniform(0.05,0.09), 70, int(rng.uniform(35,60))))
        # tom swell every 4 bars -> Low floor tom (41)
        if bar%4==0:
            start = 2*sec_per_beat; dur=1.6*sec_per_beat
            events.append((t0+start, dur, 41, 50))
    return events

def render_perc_wav(path, job, events):
    sr=job["sr"]; total=np.zeros(int(sr*4*(60.0/job["bpm"])*job["bars"]),dtype=np.float32)
    for onset_sec, dur_sec, gm_note, vel in events:
        si=int(onset_sec*sr); n=int(dur_sec*sr)
        if n<=0 or si>=len(total): continue
        if gm_note==42:   # brush
            tone=synth_tone(0,dur_sec,sr,kind="noise"); tone=iir_lowpass(tone,0.06); tone=iir_highpass(tone,0.008); amp=vel/127.0*0.12
        elif gm_note==70: # maracas
            tone=synth_tone(0,dur_sec,sr,kind="noise"); tone=iir_lowpass(tone,0.03); amp=vel/127.0*0.15
        elif gm_note==41: # tom
            tone=synth_tone(70.0,dur_sec,sr,kind="tom"); tone=iir_lowpass(tone,0.02); amp=vel/127.0*0.6
        else:
            tone=synth_tone(0,dur_sec,sr,kind="noise"); amp=vel/127.0*0.1
        tone=(tone*amp).astype(np.float32)
        ei=min(len(total), si+len(tone))
        if ei>si: total[si:ei]+=tone[:ei-si]
    write_wav_mono(path,total,job["sr"])

# =========================
# ===== Style engines =====
# =========================

def render_style(job, stacksA, stacksB):
    """Optional make_tracks_v5-like style render; returns (audio, events_for_midi_or_None, crescendo_rows_or_None)"""
    style = job.get("style")
    if not style: return None, None, None
    sr=job["sr"]; bpm=job["bpm"]; bars=job["bars"]; color=job.get("color","neutral")
    sec_per_beat=60.0/bpm; bar_dur=4*sec_per_beat
    prog = stacksA + stacksB; plen=len(prog)
    # build MIDI chords list for pad/arp fallback
    midi_chords=[]; for_roots=[]
    for _, pcs, root_pc in prog:
        midi_chords.append(stack_to_midi(pcs, root_pc, job.get("root_oct_audio",3)))
        for_roots.append(12*(job.get("bass_oct",2)+1)+pcs[0])

    total_bars = bars
    if style == "pad":
        blocks=[]
        for bar in range(total_bars):
            notes = midi_chords[bar%plen]
            block=np.zeros(int(sr*bar_dur),dtype=np.float32)
            for i,n in enumerate(notes):
                block += synth_tone(midi_to_hz(n), bar_dur, sr, kind="pad", color=color) * (0.22 if i==0 else 0.18)
            blocks.append(np.tanh(block*0.8))
        return np.concatenate(blocks), None, None

    if style == "arp":
        blocks=[]
        step=sec_per_beat/4.0; pattern=[0,1,2,1, 2,1,0,1]
        for bar in range(total_bars):
            notes = midi_chords[bar%plen]
            arp = notes[-3:] if len(notes)>=3 else notes
            barbuf=np.zeros(int(sr*bar_dur),dtype=np.float32)
            for s,idx in enumerate(pattern):
                start=int(s*step*sr); dur=step*0.9
                ta=synth_tone(midi_to_hz(arp[idx%len(arp)]), dur, sr, kind="arp", color=color)
                ta=iir_lowpass(ta,0.04)
                end=min(len(barbuf), start+len(ta)); barbuf[start:end]+=ta[:end-start]*0.18
            blocks.append(np.tanh(barbuf*1.1))
        return np.concatenate(blocks), None, None

    if style == "rain":
        rain = job.get("rain", {})
        start = float(rain.get("start", 0.8)); end = float(rain.get("end", 0.8))
        jitter_ms=float(rain.get("jitter_ms",15)); thunder=bool(rain.get("thunder",False))
        thunder_every=int(rain.get("thunder_every",2))
        rng=np.random.default_rng(42)
        steps=8; base_step=bar_dur/steps; jitter_s=jitter_ms/1000.0
        blocks=[]; rows=[]; events=[]
        for bar in range(total_bars):
            alpha = bar/max(1,total_bars-1); density=(1-alpha)*start + alpha*end
            swell = 0.925 + 0.075*math.sin(2*math.pi*bar/max(1,total_bars))
            notes = midi_chords[bar%plen]
            order=[0,1,2,1, 3,2,1,0]
            barbuf=np.zeros(int(sr*bar_dur),dtype=np.float32)
            for s in range(steps):
                if rng.random()<=density:
                    idx=order[s%len(order)]
                    dur=base_step*rng.uniform(0.35,0.6)
                    start_t=s*base_step + rng.uniform(-jitter_s, jitter_s)
                    start_t=max(0.0, min(bar_dur-1.0/sr, start_t))
                    tone=synth_tone(midi_to_hz(notes[idx%len(notes)]), dur, sr, kind="arp", color=color)*swell
                    si=int(start_t*sr); ei=min(len(barbuf), si+len(tone))
                    if ei>si:
                        barbuf[si:ei]+=tone[:ei-si]
                        vel=int(75*swell*np.random.uniform(0.85,1.1)); vel=max(30,min(115,vel))
                        events.append((bar*bar_dur+start_t, dur, notes[idx%len(notes)], vel))
            if thunder and bar%max(1,thunder_every)==0:
                th=synth_tone(55.0, min(bar_dur*0.9,2.4), sr, kind="tom", color=color)
                L=min(len(barbuf), len(th)); barbuf[:L]+=th[:L]*0.9
            blocks.append(np.tanh(barbuf*1.15))
            rows.append({"bar":bar+1,"density":density,"swell":swell})
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

    for job in jobs:
        job = {**defaults, **job}
        os.makedirs(outdir, exist_ok=True)

        # Build stacks (Theme A + Theme B)
        stacksA, stacksB = stacks_from_config(job)

        # Optional PDF
        if job.get("export_pdf", defaults.get("export_pdf", False)) or job.get("exports",{}).get("pdf_path"):
            pdf_path = job.get("exports",{}).get("pdf_path", f"{job['name']}_chords_voicings.pdf")
            make_pdf(os.path.join(outdir,pdf_path),
                     f"{job['name']} — Chords & Voicings ({job['key']} {job['mode']})",
                     stacksA, stacksB)

        # Practice WAV (pad+bass+arp)
        if job.get("export_wav", defaults.get("export_wav", False)) and job.get("exports",{}).get("practice_wav"):
            render_practice_wav(os.path.join(outdir, job["exports"]["practice_wav"]), job, stacksA, stacksB)

        # Percussion events + WAV
        perc_events=None
        if job.get("export_wav", defaults.get("export_wav", False)) and job.get("exports",{}).get("percussion_wav"):
            perc_events = build_perc_events(job)
            render_perc_wav(os.path.join(outdir, job["exports"]["percussion_wav"]), job, perc_events)

        # MIDI exports
        if job.get("export_midi", defaults.get("export_midi", False)):
            ex=job.get("exports",{})
            # Arp MIDI
            if ex.get("midi_arp"):
                arp_events = build_arp_events(stacksA, stacksB, job["bpm"], job["bars"], job.get("root_oct_audio",3))
                write_arp_midi(os.path.join(outdir, ex["midi_arp"]), arp_events, job["bpm"])
            # Pad MIDI
            if ex.get("midi_pad"):
                midi_chords=[]
                for stk in (stacksA+stacksB):
                    _, pcs, root_pc = stk
                    midi_chords.append(stack_to_midi(pcs, root_pc, job.get("root_oct_audio",3)))
                write_pad_midi(os.path.join(outdir, ex["midi_pad"]), midi_chords, job["bpm"], job["bars"])
            # Bass MIDI
            if ex.get("midi_bass"):
                roots=[]
                for _, pcs, _ in (stacksA+stacksB):
                    roots.append(12*(job.get("bass_oct",2)+1) + pcs[0])
                write_bass_midi(os.path.join(outdir, ex["midi_bass"]), roots, job["bpm"], job["bars"])
            # Perc MIDI
            if ex.get("midi_perc"):
                if perc_events is None: perc_events = build_perc_events(job)
                write_perc_midi(os.path.join(outdir, ex["midi_perc"]), perc_events, job["bpm"])

        # Optional style render (pad/arp/rain)
        if job.get("style"):
            audio, events, rows = render_style(job, stacksA, stacksB)
            ex=job.get("exports",{})
            if audio is not None and ex.get("style_wav"):
                write_wav_mono(os.path.join(outdir, ex["style_wav"]), audio, job["sr"])
            if events is not None and ex.get("style_mid") and HAVE_MIDI:
                mf=MIDIFile(1); tr=0; mf.addTempo(tr,0,job["bpm"]); ch=0
                for onset, dur, note, vel in events:
                    mf.addNote(tr,ch, note, onset*(job["bpm"]/60.0), max(0.05,dur*(job["bpm"]/60.0)), vel)
                with open(os.path.join(outdir, ex["style_mid"]),"wb") as f: mf.writeFile(f); print("[MIDI]", ex["style_mid"])
            if rows is not None and ex.get("crescendo_csv"):
                with open(os.path.join(outdir, ex["crescendo_csv"]),"w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=["bar","density","swell"])
                    w.writeheader(); w.writerows(rows)
                print("[CSV]", ex["crescendo_csv"])

if __name__=="__main__":
    main()
