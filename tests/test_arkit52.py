"""Tests for the ARKit 52 blendshape tables."""

import numpy as np

from omnifacerig_repro.arkit52 import (
    ARKIT_52, ARKIT_52_SET, CORE_13_DIALOG, FACS_AU_TO_ARKIT,
    arkit_vector, au_vector,
)


def test_52_unique_names():
    assert len(ARKIT_52) == 52
    assert len(set(ARKIT_52)) == 52


def test_well_known_entries_present():
    for name in ["jawOpen", "eyeBlinkLeft", "mouthSmileRight", "browInnerUp",
                 "mouthPucker", "tongueOut", "eyeLookInLeft", "cheekPuff"]:
        assert name in ARKIT_52_SET


def test_core_13_subset():
    assert len(CORE_13_DIALOG) == 13
    assert set(CORE_13_DIALOG) <= ARKIT_52_SET


def test_facs_au_entries_valid():
    for au, entries in FACS_AU_TO_ARKIT.items():
        assert entries, au
        for name, w in entries:
            assert name in ARKIT_52_SET, f"{au} -> unknown blendshape {name}"
            # signed offsets are allowed (e.g. AU25 lips part = -mouthClose),
            # but must stay within activation range
            assert -1.0 <= w <= 1.0


def test_arkit_vector():
    v = arkit_vector({"jawOpen": 1.0, "eyeBlinkLeft": 0.5})
    assert len(v) == 52
    assert v[ARKIT_52.index("jawOpen")] == 1.0
    assert v[ARKIT_52.index("eyeBlinkLeft")] == 0.5
    assert v[ARKIT_52.index("tongueOut")] == 0.0


def test_au_vector():
    v = au_vector({"AU26_jaw_drop": 1.0, "AU45_blink": 1.0})
    assert len(v) == 52
    assert v[ARKIT_52.index("jawOpen")] == 1.0
    assert v[ARKIT_52.index("eyeBlinkLeft")] == 1.0
    assert v[ARKIT_52.index("eyeBlinkRight")] == 1.0
    assert np.all(np.asarray(v) >= 0.0) and np.all(np.asarray(v) <= 1.0)
