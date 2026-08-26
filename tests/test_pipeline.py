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


def test_geometric_anchors_frontal_axis_detection():
    """Anchors must sit on the thinner horizontal axis (front), with the eye
    separation along the face-width axis — +X for Tripo-style models, +Z for
    TRELLIS-style models."""
    from scripts.stage1_real import geometric_anchors
    import numpy as np

    def synth(depth_axis, width=0.9, depth=0.16, h=1.0):
        """Ellipsoid humanoid: thin along depth_axis (0=x, 2=z)."""
        a = np.array([width, 0, width])
        b = np.array([depth, 0, depth])
        e = np.array([width, h, width])
        if depth_axis == 0:
            a[0] = b[0] = depth
        else:
            a[2] = b[2] = depth
        u = np.linspace(0, np.pi, 24)
        v = np.linspace(0, 2 * np.pi, 48)
        V = np.stack([np.outer(np.sin(u), np.cos(v)) * a[0],
                      np.outer(np.cos(u), np.ones_like(v)) * e[1] * 0.5,
                      np.outer(np.sin(u), np.sin(v)) * a[2]], axis=-1)
        V = V.reshape(-1, 3)
        V[:, 1] += 0.5
        return V

    for depth_axis, name in ((0, "thinX"), (2, "thinZ")):
        V = synth(depth_axis)
        anchors, idx = geometric_anchors(V)
        el, er, mouth = (np.asarray(anchors[k]) for k in
                         ("eye_left", "eye_right", "mouth_center"))
        # eyes separated along the face-width axis (not the depth axis)
        sep_axis = 2 if depth_axis == 0 else 0
        sep = abs(el[sep_axis] - er[sep_axis])
        depth_sep = abs(el[depth_axis] - er[depth_axis])
        assert sep > 0.1, (name, "eye separation along width", sep)
        assert depth_sep < 0.03, (name, "eyes must not straddle depth", depth_sep)
        # front surface: depth coordinate on the +depth_axis side
        assert mouth[depth_axis] > 0.02, (name, "mouth on front", mouth)
