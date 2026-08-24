"""End-to-end CPU pipeline demo test (mesh -> rigged GLB)."""

import os

import pytest

from omnifacerig_repro.arkit52 import ARKIT_52
from omnifacerig_repro.glb_export import load_glb
from omnifacerig_repro.pipeline import run_demo


def test_run_demo(tmp_path):
    out = os.path.join(tmp_path, "demo.glb")
    stats = run_demo(out_path=out)
    assert os.path.exists(out)
    assert stats["corr_err"] < 0.06
    assert stats["flipped_triangles_after_fit"] == 0
    assert stats["morph_targets"] == 52
    assert stats["vertices"] > 0 and stats["faces"] > 0
    assert len(stats["visemes"]) >= 2

    loaded = load_glb(out)
    names = loaded.meshes[0].extras["targetNames"]
    assert names == ARKIT_52
    assert len(loaded.skins) == 1
    assert len(loaded.animations) == 1
