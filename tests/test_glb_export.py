"""Tests for GLB export (morph targets + skinning + animation)."""

import os

import numpy as np
import pygltflib

from omnifacerig_repro.arkit52 import ARKIT_52
from omnifacerig_repro.glb_export import build_gltf, write_glb, load_glb
from omnifacerig_repro.pipeline import ellipsoid_mesh, demo_skeleton


def test_glb_roundtrip_morphs(tmp_path):
    V, F = ellipsoid_mesh(1.0, 1.2, 1.0, n_lat=6, n_lon=12)
    n = len(V)
    morphs = {name: np.zeros((n, 3)) for name in ARKIT_52}
    morphs["jawOpen"] = np.zeros((n, 3)) + np.array([0.0, -0.1, 0.05])
    skin = demo_skeleton(V, 1.2, 1.0)
    anim = {"times": np.array([0.0, 0.2]), "weights": np.zeros((2, 52))}

    path = os.path.join(tmp_path, "test.glb")
    gltf = build_gltf(V, F, morphs, skin=skin, animation=anim, name="t")
    write_glb(path, gltf)
    assert os.path.getsize(path) > 1000

    loaded = load_glb(path)
    assert loaded.asset.version == "2.0"
    mesh = loaded.meshes[0]
    assert mesh.extras is not None
    assert mesh.extras["targetNames"] == ARKIT_52
    prim = mesh.primitives[0]
    assert prim.attributes.POSITION is not None
    assert prim.attributes.JOINTS_0 is not None
    assert prim.attributes.WEIGHTS_0 is not None
    assert len(prim.targets) == 52
    # accessor counts: vertices
    pos_acc = loaded.accessors[prim.attributes.POSITION]
    assert pos_acc.count == n
    # skin
    assert len(loaded.skins) == 1
    assert len(loaded.skins[0].joints) == 4
    # animation
    assert len(loaded.animations) == 1
    assert len(loaded.animations[0].channels) == 1
    assert loaded.animations[0].channels[0].target.path == "weights"


def test_glb_minimal_no_skin(tmp_path):
    V, F = ellipsoid_mesh(1.0, 1.0, 1.0, n_lat=4, n_lon=8)
    path = os.path.join(tmp_path, "min.glb")
    gltf = build_gltf(V, F, {})
    write_glb(path, gltf)
    loaded = load_glb(path)
    assert loaded.scenes[0].nodes == [0]
    assert loaded.meshes[0].primitives[0].targets in (None, [])
