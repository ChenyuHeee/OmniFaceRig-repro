"""ARKit 52 blendshapes and FACS mapping tables.

The ARKit 52 name set is the authoritative Apple list
(ARFaceAnchor.BlendShapeLocation); order follows the public documentation.
The FACS AU -> ARKit mapping is a best-effort community-standard mapping used
for driving ARKit rigs from FACS-style activations; values are initial
calibration weights in [0, 1] and should be refined against the official
canonical FACS template once it is available (A100/TA resource).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# ARKit 52 blendshapes (Apple ARFaceAnchor.BlendShapeLocation, 52 entries)
# ---------------------------------------------------------------------------
ARKIT_52: list[str] = [
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight", "tongueOut",
]

ARKIT_52_SET: frozenset[str] = frozenset(ARKIT_52)
assert len(ARKIT_52) == 52 and len(ARKIT_52_SET) == 52, "ARKit list must be 52 unique names"

# Symmetric pairs used by the pipeline to keep left/right consistency.
_LEFT_RIGHT_PAIRS: list[tuple[str, str]] = [
    ("browDown", "browDownLeft", "browDownRight"),
    ("browOuterUp", "browOuterUpLeft", "browOuterUpRight"),
    ("cheekSquint", "cheekSquintLeft", "cheekSquintRight"),
    ("eyeBlink", "eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeLookDown", "eyeLookDownLeft", "eyeLookDownRight"),
    ("eyeLookIn", "eyeLookInLeft", "eyeLookInRight"),
    ("eyeLookOut", "eyeLookOutLeft", "eyeLookOutRight"),
    ("eyeLookUp", "eyeLookUpLeft", "eyeLookUpRight"),
    ("eyeSquint", "eyeSquintLeft", "eyeSquintRight"),
    ("eyeWide", "eyeWideLeft", "eyeWideRight"),
    ("mouthDimple", "mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthFrown", "mouthFrownLeft", "mouthFrownRight"),
    ("mouthLowerDown", "mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthPress", "mouthPressLeft", "mouthPressRight"),
    ("mouthSmile", "mouthSmileLeft", "mouthSmileRight"),
    ("mouthStretch", "mouthStretchLeft", "mouthStretchRight"),
    ("mouthUpperUp", "mouthUpperUpLeft", "mouthUpperUpRight"),
    ("noseSneer", "noseSneerLeft", "noseSneerRight"),
]
BILATERAL_MAP: dict[str, tuple[str, str]] = {
    base: (left, right) for base, left, right in _LEFT_RIGHT_PAIRS
}

# ---------------------------------------------------------------------------
# FACS tier sizes (paper §3.6.4). The exact 155-name composition is not
# published; we define the tiers on top of the ARKit 52 set for the challenge
# deliverable ("support the ARKit 52 set").
# ---------------------------------------------------------------------------
FACS_TIERS = {"core": 13, "additional": 46, "full": 155}

# Proposed Core dialog set (13 shapes): basic dialog + emotion per paper §3.6.4.
CORE_13_DIALOG: list[str] = [
    "jawOpen", "mouthClose",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthPucker", "mouthFunnel",
    "mouthStretchLeft", "mouthStretchRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "browInnerUp",
]
assert len(CORE_13_DIALOG) == 13

# ---------------------------------------------------------------------------
# FACS Action Units -> ARKit 52 blendshape weights (initial calibration).
# Each entry: AU name -> list of (blendshape, weight).
# Standard AU semantics (Ekman & Friesen); mapping follows the widely used
# ARKit/FACS correspondence (Apple docs + community rigs).  Values are
# starting points for the expression template, to be tuned per character.
# ---------------------------------------------------------------------------
FACS_AU_TO_ARKIT: dict[str, list[tuple[str, float]]] = {
    # Brow
    "AU1_inner_brow_raiser":   [("browInnerUp", 1.0)],
    "AU2_outer_brow_raiser":   [("browOuterUpLeft", 1.0), ("browOuterUpRight", 1.0)],
    "AU4_brow_lowerer":        [("browDownLeft", 1.0), ("browDownRight", 1.0)],
    # Eyes
    "AU5_upper_lid_raiser":    [("eyeWideLeft", 1.0), ("eyeWideRight", 1.0)],
    "AU6_cheek_raiser":        [("cheekSquintLeft", 1.0), ("cheekSquintRight", 1.0)],
    "AU7_lid_tightener":       [("eyeSquintLeft", 0.8), ("eyeSquintRight", 0.8)],
    "AU43_eye_closure":        [("eyeBlinkLeft", 1.0), ("eyeBlinkRight", 1.0)],
    "AU45_blink":              [("eyeBlinkLeft", 1.0), ("eyeBlinkRight", 1.0)],
    "AU46_wink_left":          [("eyeBlinkLeft", 1.0)],
    "AU46_wink_right":         [("eyeBlinkRight", 1.0)],
    # Nose
    "AU9_nose_wrinkler":       [("noseSneerLeft", 0.7), ("noseSneerRight", 0.7)],
    # Mouth
    "AU10_upper_lip_raiser":   [("mouthUpperUpLeft", 0.6), ("mouthUpperUpRight", 0.6),
                                ("noseSneerLeft", 0.4), ("noseSneerRight", 0.4)],
    "AU12_lip_corner_puller":  [("mouthSmileLeft", 1.0), ("mouthSmileRight", 1.0)],
    "AU13_sharp_lip_puller":   [("mouthSmileLeft", 0.6), ("mouthSmileRight", 0.6),
                                ("mouthStretchLeft", 0.4), ("mouthStretchRight", 0.4)],
    "AU14_dimpler":            [("mouthDimpleLeft", 1.0), ("mouthDimpleRight", 1.0)],
    "AU15_lip_corner_depressor": [("mouthFrownLeft", 1.0), ("mouthFrownRight", 1.0)],
    "AU16_lower_lip_depressor": [("mouthLowerDownLeft", 1.0), ("mouthLowerDownRight", 1.0),
                                 ("jawOpen", 0.3)],
    "AU17_chin_raiser":        [("jawForward", 0.5), ("mouthShrugLower", 0.6),
                                ("mouthPressLeft", 0.4), ("mouthPressRight", 0.4)],
    "AU18_lip_puckerer":       [("mouthPucker", 1.0)],
    "AU20_lip_stretcher":      [("mouthStretchLeft", 1.0), ("mouthStretchRight", 1.0)],
    "AU22_lip_funneler":       [("mouthFunnel", 1.0)],
    "AU23_lip_tightener":      [("mouthPressLeft", 1.0), ("mouthPressRight", 1.0)],
    "AU24_lip_pressor":        [("mouthPressLeft", 1.0), ("mouthPressRight", 1.0),
                                ("mouthClose", 0.4)],
    "AU25_lips_part":          [("jawOpen", 0.5), ("mouthClose", -0.4)],
    "AU26_jaw_drop":           [("jawOpen", 1.0)],
    "AU27_mouth_stretch":      [("jawOpen", 0.8), ("mouthStretchLeft", 0.5),
                                ("mouthStretchRight", 0.5)],
    "AU28_lip_suck":           [("cheekPuff", 0.6), ("mouthPucker", 0.5)],
    "AU32_bite":               [("jawForward", 0.4), ("mouthPressLeft", 0.5),
                                ("mouthPressRight", 0.5)],
    # Jaw / tongue
    "AU_jaw_left":             [("jawLeft", 1.0)],
    "AU_jaw_right":            [("jawRight", 1.0)],
    "AU_jaw_forward":          [("jawForward", 1.0)],
    "AU_tongue_out":           [("tongueOut", 1.0)],
    # Eye gaze (virtual eyeball rotation; blendshape-driven approximation)
    "gaze_up":                 [("eyeLookUpLeft", 1.0), ("eyeLookUpRight", 1.0)],
    "gaze_down":               [("eyeLookDownLeft", 1.0), ("eyeLookDownRight", 1.0)],
    "gaze_left":               [("eyeLookInLeft", 1.0), ("eyeLookOutRight", 1.0)],
    "gaze_right":              [("eyeLookOutLeft", 1.0), ("eyeLookInRight", 1.0)],
}

# Reverse: blendshape -> primary AU (for documentation / reporting).
ARKIT_TO_AU: dict[str, str] = {
    name: au
    for au, entries in FACS_AU_TO_ARKIT.items()
    for name, _w in entries
}


def arkit_vector(activations: dict[str, float]) -> list[float]:
    """Expand a partial {blendshape: weight} dict to the full 52-vector."""
    return [float(activations.get(name, 0.0)) for name in ARKIT_52]


def au_vector(au_weights: dict[str, float]) -> list[float]:
    """Expand FACS AU weights to a full ARKit 52 vector (clamped to [0,1])."""
    out = {name: 0.0 for name in ARKIT_52}
    for au, w in au_weights.items():
        if au not in FACS_AU_TO_ARKIT:
            raise KeyError(f"unknown FACS AU: {au!r}")
        for name, unit in FACS_AU_TO_ARKIT[au]:
            out[name] = min(1.0, max(0.0, out[name] + float(w) * unit))
    return [out[name] for name in ARKIT_52]
