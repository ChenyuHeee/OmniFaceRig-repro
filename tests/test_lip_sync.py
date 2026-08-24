"""Tests for Chinese + English lip-sync mapping tables."""

from omnifacerig_repro.arkit52 import ARKIT_52_SET
from omnifacerig_repro.lip_sync import (
    OCULUS_VISEMES, RHUBARB_VISEMES, VISEME_TO_ARKIT,
    pinyin_syllable_to_visemes, zh_text_to_visemes,
    viseme_to_arkit, rhubarb_viseme_to_arkit,
)


def test_all_oculus_visemes_have_arkit_mapping():
    for v in OCULUS_VISEMES:
        w = viseme_to_arkit(v)
        assert set(w) <= ARKIT_52_SET
        assert all(0.0 <= x <= 1.0 for x in w.values())


def test_all_rhubarb_classes_valid():
    for v in RHUBARB_VISEMES:
        w = rhubarb_viseme_to_arkit(v)
        assert set(w) <= ARKIT_52_SET
    assert rhubarb_viseme_to_arkit("X") == {}


def test_mandarin_syllable_mapping():
    assert pinyin_syllable_to_visemes("b", "a") == ["PP", "aa"]
    assert pinyin_syllable_to_visemes("n", "i") == ["nn", "I"]
    assert pinyin_syllable_to_visemes("zh", "i") == ["CH", "E"]  # apical i
    assert pinyin_syllable_to_visemes("x", "ue") == ["CH", "I", "E"]
    assert pinyin_syllable_to_visemes("m", "ang") == ["PP", "aa", "nn"]


def test_zh_text_to_visemes():
    visemes = zh_text_to_visemes("你好")  # ni3 hao3
    assert "nn" in visemes and "I" in visemes
    assert "aa" in visemes or "O" in visemes


def test_zh_text_non_chinese_tolerant():
    visemes = zh_text_to_visemes("abc")
    assert isinstance(visemes, list)


def test_viseme_track_keys_valid():
    from omnifacerig_repro.lip_sync import viseme_track_to_arkit_track
    track = [("aa", 0.0, 0.2), ("PP", 0.2, 0.4)]
    out = viseme_track_to_arkit_track(track)
    assert len(out) == 2
    assert set(out[0][0]) <= ARKIT_52_SET
