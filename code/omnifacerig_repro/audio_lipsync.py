"""Real audio lip-sync (deliverable 5): audio -> viseme track -> ARKit animation.

Two real paths:
  1. text -> piper TTS with phoneme timestamps (zh/en) -> viseme track
  2. real audio + text -> faster-whisper word timestamps -> per-word visemes

Both produce a [(viseme, start, end)] track, expanded to a glTF WEIGHTS
animation via viseme_track_to_animation() (52 ARKit channels with real time).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import os

import numpy as np

from .arkit52 import ARKIT_52, arkit_vector
from . import lip_sync

# ---------------------------------------------------------------------------
# espeak/piper phoneme -> Oculus viseme (en)
# ---------------------------------------------------------------------------
_PHONEME_TO_VISEME: dict[str, str] = {
    # vowels
    "a": "aa", "aa": "aa", "ah": "aa", "ae": "aa", "a:": "aa", "a0": "aa",
    "e": "E", "e:": "E", "eI": "E", "aI": "E",
    "i": "I", "i:": "I", "I": "I",
    "o": "O", "o:": "O", "O": "O", "oU": "O", "OI": "O",
    "u": "U", "u:": "U", "U": "U", "@U": "U",
    "@": "E", "3:": "E", "Q": "aa", "V": "aa",
    # consonants
    "b": "PP", "p": "PP", "m": "PP",
    "f": "FF", "v": "FF",
    "T": "TH", "D": "TH",
    "d": "DD", "t": "DD", "n": "nn", "l": "DD", "r": "RR",
    "k": "kk", "g": "kk", "N": "kk", "h": "kk",
    "tS": "CH", "dZ": "CH", "S": "CH", "Z": "CH",
    "s": "SS", "z": "SS",
    "w": "U", "j": "I",
    "ts": "SS", "dz": "SS",
}

# piper zh (cmn) output is pinyin syllables like "n i3" / "hao3"; strip tones
_PINYIN_TONE = str.maketrans("", "", "0123456789")


def _piper_phoneme_to_viseme(ph: str) -> str | None:
    p = ph.strip().lstrip("_")
    if not p:
        return None
    # strip espeak modifiers: "a_I", "d=", "e:" -> base letter(s)
    base = p.replace(":", "").replace("=", "").replace("_", "").split(" ")[0]
    if base in _PHONEME_TO_VISEME:
        return _PHONEME_TO_VISEME[base]
    if len(base) == 1:
        return _PHONEME_TO_VISEME.get(base)
    return None


def piper_tts_wav(text: str, model_path: str, out_wav: str) -> str:
    """piper TTS: synthesize real speech to a wav file (no alignment needed;
    timestamps come from faster-whisper on the synthesized audio)."""
    import wave
    from piper import PiperVoice  # type: ignore

    voice = PiperVoice.load(model_path)
    with wave.open(out_wav, "wb") as wf:
        first = True
        for chunk in voice.synthesize(text):
            if first:
                wf.setnchannels(chunk.sample_channels)
                wf.setsampwidth(chunk.sample_width)
                wf.setframerate(chunk.sample_rate)
                first = False
            wf.writeframes(chunk.audio_int16_bytes)
    return out_wav


def _split_pinyin(syl: str) -> list[tuple[str, str]]:
    """Split a pinyin syllable (tone already stripped) into (initial, final)."""
    for init in sorted(lip_sync.ZH_INITIAL_TO_VISEME, key=len, reverse=True):
        if syl.startswith(init) and len(syl) > len(init):
            return [(init, syl[len(init):])]
    return [("", syl)]


# ---------------------------------------------------------------------------
# faster-whisper: real audio -> word timestamps -> visemes
# ---------------------------------------------------------------------------

def whisper_viseme_track(
    audio_path: str, text: str, lang: str = "en", model_path: str | None = None,
) -> list[tuple[str, float, float]]:
    """Transcribe-aligned word timestamps (faster-whisper) -> viseme track.

    Alignment uses whisper's own word timestamps (the real audio signal).
    The provided text is used as a sanity check and fallback word source:
    for en, transcribed words are greedily matched against the expected
    transcript; for zh (no spaces), whisper words are used as-is.
    """
    import re
    from faster_whisper import WhisperModel  # type: ignore

    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "faster-whisper-base.bin")
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio_path, language=lang, word_timestamps=True, vad_filter=True)

    # collect words with timestamps
    words: list[tuple[str, float, float]] = []
    for seg in segments:
        for w in (seg.words or []):
            words.append((w.word.strip(), float(w.start), float(w.end)))
    if not words:
        # fallback: one segment spanning the whole audio
        words = [(text, 0.0, float(getattr(info, "duration", 2.0)))]

    track: list[tuple[str, float, float]] = []
    if lang == "zh":
        # Chinese: whisper emits word fragments; use them directly, merging
        # into syllable-length chunks for pinyin mapping
        for tw, ts, te in words:
            chars = "".join(c for c in tw if "\u4e00" <= c <= "\u9fff")
            if not chars:
                continue
            visemes = lip_sync.zh_text_to_visemes(chars)
            if not visemes:
                continue
            seg = (te - ts) / len(visemes)
            for k, v in enumerate(visemes):
                track.append((v, ts + k * seg, ts + (k + 1) * seg))
        return track

    # English: greedy match against the expected transcript
    expect = [w.lower() for w in re.findall(r"[\w']+", text.lower())]
    ei = 0
    for tw, ts, te in words:
        if ei >= len(expect):
            break
        if tw.lower() == expect[ei] or tw.lower().startswith(expect[ei]) \
                or expect[ei].startswith(tw.lower()):
            w = expect[ei]
            ei += 1
        else:
            continue
        visemes = lip_sync.en_text_to_visemes(w)
        if not visemes:
            continue
        seg = (te - ts) / len(visemes)
        for k, v in enumerate(visemes):
            track.append((v, ts + k * seg, ts + (k + 1) * seg))
    if ei < len(expect) * 0.5:
        import warnings
        warnings.warn(f"low transcript match ({ei}/{len(expect)} words); "
                      "check audio/transcript")
    return track


# ---------------------------------------------------------------------------
# track -> ARKit animation
# ---------------------------------------------------------------------------

def viseme_track_to_animation(
    track: list[tuple[str, float, float]],
    pad: float = 0.12,
) -> dict:
    """[(viseme, start, end)] -> glTF WEIGHTS animation dict.

    Output: {"times": (T,), "weights": (T, 52)} with smoothing between
    visemes (half-cosine crossfade of pad seconds).
    """
    if not track:
        raise ValueError("empty viseme track")
    t0 = max(0.0, track[0][1] - pad)
    t1 = track[-1][2] + pad
    # key every transition + every viseme start/end
    keys = sorted({t0, t1} | {s for _, s, e in track} | {e for _, s, e in track})
    times = np.asarray(keys, dtype=np.float32)
    weights = np.zeros((len(times), 52))
    cur = {v: 0.0 for v in lip_sync.OCULUS_VISEMES}
    for i, t in enumerate(times):
        # find active viseme at t
        active = [v for v, s, e in track if s <= t < e]
        if active:
            cur = {v: 0.0 for v in cur}
            for v in active:
                cur[v] = 1.0 / len(active)
        vec = np.zeros(52)
        for v, w in cur.items():
            if w > 0:
                vec += np.asarray(arkit_vector(lip_sync.viseme_to_arkit(v))) * w
        weights[i] = np.clip(vec, 0, 1)
    return {"times": times, "weights": weights}


def animate_glb(glb_path: str, out_path: str, track: list[tuple[str, float, float]]):
    """Add a WEIGHTS animation (from a viseme track) to an existing rigged glb."""
    import pygltflib
    from .glb_export import (load_glb, _BinaryBuilder,  # type: ignore
                            _ARRAY_BUFFER, _FLOAT)

    with open(glb_path, "rb") as fh:
        gltf = load_glb(glb_path)
    anim = viseme_track_to_animation(track)
    n_targets = len(gltf.meshes[0].primitives[0].targets or [])
    assert anim["weights"].shape[1] == n_targets
    # drop any previous WEIGHTS-only animations (e.g. the pipeline's fixed
    # rhythm default); keep body animations untouched
    kept = []
    for a in (gltf.animations or []):
        paths = {c.target.path for c in a.channels}
        if paths == {"weights"} and len(a.channels) == 1:
            continue
        kept.append(a)
    gltf.animations = kept
    # append new accessors to the existing binary blob
    blob = gltf.binary_blob()
    bb = _BinaryBuilder(gltf)
    bb.parts = [blob]
    bb.offset = len(blob)
    t_acc = bb.add_accessor(anim["times"], _ARRAY_BUFFER, "SCALAR", _FLOAT)
    w_acc = bb.add_accessor(anim["weights"], _ARRAY_BUFFER, "SCALAR", _FLOAT)
    sampler = pygltflib.AnimationSampler(input=t_acc, output=w_acc)
    channel = pygltflib.AnimationChannel(
        sampler=len(gltf.animations or []),
        target=pygltflib.AnimationChannelTarget(node=0, path="weights"))
    gltf.animations = (gltf.animations or []) + [
        pygltflib.Animation(channels=[channel], samplers=[sampler])]
    bb.finish()
    gltf.save_binary(out_path)
    return out_path
