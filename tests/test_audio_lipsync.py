"""Tests for audio lip-sync module (pure functions; real TTS/ASR is
integration-tested on the A100 server)."""

import numpy as np
import pytest

from omnifacerig_repro.audio_lipsync import (
    _piper_phoneme_to_viseme, piper_viseme_track, whisper_viseme_track,
    viseme_track_to_animation,
)
from omnifacerig_repro.arkit52 import ARKIT_52


def test_phoneme_mapping():
    assert _piper_phoneme_to_viseme("d") == "DD"
    assert _piper_phoneme_to_viseme("b") == "PP"
    assert _piper_phoneme_to_viseme("a:") == "aa"
    assert _piper_phoneme_to_viseme("m") == "PP"
    assert _piper_phoneme_to_viseme("s") == "SS"
    assert _piper_phoneme_to_viseme("k") == "kk"
    assert _piper_phoneme_to_viseme("T") == "TH"
    assert _piper_phoneme_to_viseme("i:") == "I"
    assert _piper_phoneme_to_viseme("u:") == "U"


def test_track_to_animation():
    track = [("aa", 0.0, 0.2), ("PP", 0.2, 0.4), ("I", 0.4, 0.6)]
    anim = viseme_track_to_animation(track)
    assert anim["weights"].shape[1] == 52
    assert anim["times"][0] >= 0.0
    assert anim["times"][-1] >= 0.6
    assert np.all(anim["weights"] >= 0.0) and np.all(anim["weights"] <= 1.0)
    # mid-aa should have jawOpen>0
    t = anim["times"]
    i = int(np.argmin(np.abs(t - 0.1)))
    w = anim["weights"][i]
    assert w[ARKIT_52.index("jawOpen")] > 0.0


def test_empty_track_raises():
    with pytest.raises(ValueError):
        viseme_track_to_animation([])
