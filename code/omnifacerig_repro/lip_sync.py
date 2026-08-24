"""Lip-sync: phoneme -> viseme -> ARKit 52 blendshape weights.

The challenge requires Chinese AND English audio lip-sync correctly aligned
(deliverable 5). No ready-made open-source Mandarin -> ARKit viseme solution
exists (see notes/components/04_audio_lipsync.md), so we build our own table:

    text -> pinyin syllables (pypinyin, lazy import)
          -> (initial, final) phonemes
          -> viseme sequence (Mandarin -> Oculus 15-viseme mapping below)
          -> ARKit 52 weights via VISEME_TO_ARKIT

English: Rhubarb's 10 viseme classes -> ARKit weights.

Timing is NOT computed here: duration per phoneme comes from the TTS front-end
(e.g., PaddleSpeech / piper with alignments); this module provides the static
mapping tables + weight expansion used by the animation stage.

The viseme -> ARKit weight values are initial calibration (based on the
standard Oculus viseme set and TalkingHead-style mappings); they should be
refined against recorded reference videos on the A100 machine.
"""

from __future__ import annotations

# Oculus 15-viseme set (industry standard, used by TalkingHead / Audio2Face)
OCULUS_VISEMES: list[str] = [
    "sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "I", "O", "U",
]

# Rhubarb viseme classes (v1.13, see rhubarb-lipsync documentation)
RHUBARB_VISEMES: list[str] = ["X", "A", "E", "I", "O", "U", "V", "W", "PB", "D"]

# ---------------------------------------------------------------------------
# Viseme -> ARKit 52 blendshape weights (initial calibration, [0, 1]).
# Only non-zero blendshapes are listed per viseme.
# ---------------------------------------------------------------------------
VISEME_TO_ARKIT: dict[str, dict[str, float]] = {
    "sil": {},
    # open vowels
    "aa": {"jawOpen": 0.50, "mouthStretchLeft": 0.20, "mouthStretchRight": 0.20,
           "mouthLowerDownLeft": 0.25, "mouthLowerDownRight": 0.25},
    "E":  {"jawOpen": 0.25, "mouthSmileLeft": 0.35, "mouthSmileRight": 0.35},
    "I":  {"jawOpen": 0.15, "mouthSmileLeft": 0.30, "mouthSmileRight": 0.30,
           "mouthStretchLeft": 0.15, "mouthStretchRight": 0.15},
    "O":  {"jawOpen": 0.30, "mouthFunnel": 0.55},
    "U":  {"jawOpen": 0.20, "mouthPucker": 0.60},
    # consonants
    "PP": {"mouthPucker": 0.50, "mouthPressLeft": 0.30, "mouthPressRight": 0.30,
           "mouthClose": 0.20},
    "FF": {"mouthUpperUpLeft": 0.30, "mouthUpperUpRight": 0.30,
           "mouthLowerDownLeft": 0.20, "mouthLowerDownRight": 0.20,
           "jawOpen": 0.15},
    "TH": {"tongueOut": 0.55, "jawOpen": 0.15, "mouthStretchLeft": 0.10,
           "mouthStretchRight": 0.10},
    "DD": {"jawOpen": 0.20, "mouthClose": 0.25, "mouthStretchLeft": 0.10,
           "mouthStretchRight": 0.10},
    "kk": {"jawOpen": 0.20, "mouthStretchLeft": 0.25, "mouthStretchRight": 0.25},
    "CH": {"jawOpen": 0.20, "mouthPucker": 0.40, "mouthFunnel": 0.20},
    "SS": {"jawOpen": 0.10, "mouthStretchLeft": 0.40, "mouthStretchRight": 0.40},
    "nn": {"jawOpen": 0.15, "mouthSmileLeft": 0.20, "mouthSmileRight": 0.20},
    "RR": {"jawOpen": 0.10, "mouthPucker": 0.35},
}

# Rhubarb class -> Oculus viseme (plus per-class tweaks)
RHUBARB_TO_OCULUS: dict[str, str] = {
    "X": "sil", "A": "aa", "E": "E", "I": "I", "O": "O", "U": "U",
    "V": "FF", "W": "U", "PB": "PP", "D": "DD",
}
RHUBARB_TWEAKS: dict[str, dict[str, float]] = {
    "W": {"mouthPucker": 0.80, "jawOpen": 0.10},   # rounded glide
    "V": {"mouthUpperUpLeft": 0.40, "mouthUpperUpRight": 0.40},
    "D": {"tongueOut": 0.10},
}

# ---------------------------------------------------------------------------
# Mandarin (pinyin) -> viseme mapping
# ---------------------------------------------------------------------------
# 声母 (initials) -> viseme
ZH_INITIAL_TO_VISEME: dict[str, str] = {
    "b": "PP", "p": "PP", "m": "PP",      # bilabial
    "f": "FF",                            # labiodental
    "d": "DD", "t": "DD",                 # alveolar stops
    "n": "nn", "l": "DD",                 # alveolar nasal / lateral
    "g": "kk", "k": "kk", "h": "kk",      # velar
    "j": "CH", "q": "CH", "x": "CH",      # alveolo-palatal
    "zh": "CH", "ch": "CH", "sh": "CH",   # retroflex affricates/fricative
    "r": "RR",                            # retroflex approximant
    "z": "SS", "c": "SS", "s": "SS",      # dental affricates/fricative
    "y": "I", "w": "U",                   # glides
}

# 韵母 (finals) -> viseme sequence. Compound finals produce a sequence
# (diphthong/triphthong mouth movement). "i" after retroflex/dental initials
# is apical and pronounced like a rhotic vowel -> E/RR.
ZH_FINAL_TO_VISEMES: dict[str, list[str]] = {
    "a": ["aa"], "o": ["O"], "e": ["E"],
    "i": ["I"], "u": ["U"], "v": ["U"],           # v = ü
    "ai": ["aa", "I"], "ei": ["E", "I"], "ui": ["U", "I"],
    "ao": ["aa", "O"], "ou": ["O", "U"], "iu": ["I", "U"],
    "ie": ["I", "E"], "ve": ["I", "E"], "ue": ["I", "E"],   # üe (ue: pypinyin)
    "er": ["E", "RR"],
    "an": ["aa", "nn"], "en": ["E", "nn"], "in": ["I", "nn"],
    "un": ["U", "nn"], "vn": ["I", "nn"],         # ün
    "ang": ["aa", "nn"], "eng": ["E", "nn"], "ing": ["I", "nn"],
    "ong": ["O", "nn"],
}

# apical "i" (after zh ch sh r z c s) is not [i]
_APICAL_INITIALS = {"zh", "ch", "sh", "r", "z", "c", "s"}


def pinyin_syllable_to_visemes(initial: str, final: str) -> list[str]:
    """Map one pinyin syllable (initial/final, no tone) to a viseme list."""
    visemes: list[str] = []
    if initial and initial in ZH_INITIAL_TO_VISEME:
        visemes.append(ZH_INITIAL_TO_VISEME[initial])
    if final == "i" and initial in _APICAL_INITIALS:
        visemes.append("E")          # zhi/chi/shi/ri/zi/ci/si: apical vowel
    elif final in ZH_FINAL_TO_VISEMES:
        visemes.extend(ZH_FINAL_TO_VISEMES[final])
    else:
        # unknown final: best-effort by first letter
        guess = {"a": "aa", "o": "O", "e": "E", "i": "I", "u": "U"}.get(final[:1])
        if guess:
            visemes.append(guess)
    return visemes or ["sil"]


def zh_sentence_to_pinyin(text: str) -> list[tuple[str, str]]:
    """Convert Chinese text to [(initial, final), ...] via pypinyin.

    Raises ImportError if pypinyin is not installed (it is a project dep).
    """
    try:
        from pypinyin import Style, lazy_pinyin  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pypinyin is required for Chinese lip-sync: pip install pypinyin") from exc
    out: list[tuple[str, str]] = []
    for syl in lazy_pinyin(text, style=Style.NORMAL, errors="ignore"):
        if not syl:
            continue
        syl = syl.lower()
        for init in sorted(ZH_INITIAL_TO_VISEME, key=len, reverse=True):
            if syl.startswith(init) and len(syl) > len(init):
                out.append((init, syl[len(init):]))
                break
        else:
            out.append(("", syl))
    return out


def zh_text_to_visemes(text: str) -> list[str]:
    """Full Chinese text -> concatenated viseme sequence (no timing)."""
    visemes: list[str] = []
    for initial, final in zh_sentence_to_pinyin(text):
        visemes.extend(pinyin_syllable_to_visemes(initial, final))
    return visemes


def rhubarb_viseme_to_arkit(viseme: str) -> dict[str, float]:
    """Rhubarb class -> ARKit weights (X -> neutral)."""
    if viseme == "X":
        return {}
    oculus = RHUBARB_TO_OCULUS.get(viseme, "sil")
    w = dict(VISEME_TO_ARKIT.get(oculus, {}))
    w.update(RHUBARB_TWEAKS.get(viseme, {}))
    return w


# Simple English grapheme -> viseme approximation (no phonemizer needed);
# upgrade path: Rhubarb (phonemes) when audio alignment is available.
_EN_LETTER_TO_VISEME: dict[str, str] = {
    "a": "aa", "e": "E", "i": "I", "o": "O", "u": "U", "y": "I",
    "b": "PP", "p": "PP", "m": "PP",
    "f": "FF", "v": "FF",
    "d": "DD", "t": "DD", "n": "nn", "l": "DD",
    "g": "kk", "k": "kk", "c": "kk", "h": "kk", "x": "kk",
    "j": "CH", "s": "SS", "z": "SS", "r": "RR", "w": "U",
}
_EN_DIGRAPHS = {"th": "DD", "ch": "CH", "sh": "CH", "ph": "FF", "ck": "kk", "qu": "U"}


def en_text_to_visemes(text: str) -> list[str]:
    """English text -> viseme sequence (grapheme approximation).

    Letters/digraphs map to Oculus visemes; consecutive letters of the same
    viseme collapse to one (approximation until a phonemizer front-end is
    wired in). Returns [] for empty/non-letter input.
    """
    out: list[str] = []
    i = 0
    t = text.lower()
    while i < len(t):
        if not t[i].isalpha():
            i += 1
            continue
        if i + 1 < len(t) and t[i:i + 2] in _EN_DIGRAPHS:
            g = _EN_DIGRAPHS[t[i:i + 2]]
            i += 2
        else:
            g = _EN_LETTER_TO_VISEME.get(t[i])
            i += 1
        if g and (not out or out[-1] != g):
            out.append(g)
    return out


def viseme_to_arkit(viseme: str) -> dict[str, float]:
    """Oculus viseme -> ARKit 52 weights dict."""
    return dict(VISEME_TO_ARKIT.get(viseme, {}))


def viseme_track_to_arkit_track(
    track: list[tuple[str, float, float]],
) -> list[tuple[dict[str, float], float, float]]:
    """Expand [(viseme, start, end), ...] to [(arkit weights, start, end), ...]."""
    return [(viseme_to_arkit(v), s, e) for v, s, e in track]
