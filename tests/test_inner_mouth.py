"""Tests for the inner-mouth module (paper §3.6.2: teeth/gums/tongue)."""

import os

import numpy as np
import pytest

import omnifacerig_repro.inner_mouth as im
from omnifacerig_repro.arkit52 import ARKIT_52
from omnifacerig_repro.glb_export import build_gltf, write_glb, load_glb
from omnifacerig_repro.pipeline import ellipsoid_mesh, head_keypoints


def _snap_to_ellipsoid(p, a=1.0, b=1.15, c=1.0):
    t = 1.0 / np.sqrt((p[0] / a) ** 2 + (p[1] / b) ** 2 + (p[2] / c) ** 2)
    return p * t


@pytest.fixture(scope="module")
def head():
    """Synthetic head + mouth anchors snapped onto the ellipsoid surface."""
    V, F = ellipsoid_mesh(1.0, 1.15, 1.0, n_lat=18, n_lon=36)
    idx, kp = head_keypoints(V, 1.0, 1.15, 1.0)
    # kp order: eye_left_outer, eye_left_inner, eye_right_inner, eye_right_outer,
    #           mouth_left, mouth_right, mouth_top, mouth_bottom, brow_left, brow_right
    kp_s = np.array([_snap_to_ellipsoid(p) for p in kp])
    anchors = {
        "mouth_center": (kp_s[4] + kp_s[7]) / 2,
        "mouth_left": kp_s[4],
        "mouth_right": kp_s[5],
        "mouth_top": kp_s[6],
        "mouth_bottom": kp_s[7],
        "eye_left": kp_s[0],
        "eye_right": kp_s[3],
    }
    return {"V": V, "F": F, "anchors": anchors}


def test_mouth_cavity_from_anchors(head):
    cav = im.mouth_cavity(head["V"], head["anchors"])
    sx = np.linalg.norm(head["anchors"]["mouth_right"] - head["anchors"]["mouth_left"])
    sy = np.linalg.norm(head["anchors"]["mouth_top"] - head["anchors"]["mouth_bottom"])
    assert abs(cav["size"][0] - sx) < 1e-6
    assert abs(cav["size"][1] - sy) < 1e-6
    # right-handed orthonormal basis
    R = np.stack([cav["right"], cav["up"], cav["outward"]])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_build_human_parts_behind_lips(head):
    V, F, anchors = head["V"], head["F"], head["anchors"]
    parts = im.build_inner_mouth(V, F, anchors, archetype="human")
    names = [n for _, _, n in parts]
    assert names == ["teeth_upper", "teeth_lower", "gums_tongue"]
    cav = im.mouth_cavity(V, anchors)
    for pV, pF, name in parts:
        assert pF.min() >= 0 and pF.max() < len(pV)          # valid indices
        assert np.linalg.norm(pV, axis=1).min() > 0          # no NaN/degenerate
        # every part sits behind the lip plane (mouth_center)
        o = (pV - cav["center"]) @ cav["outward"]
        assert o.max() < 1e-6, f"{name} pokes through the lips"
    assert len(parts[0][0]) > 1000 and len(parts[2][0]) > 1000  # real assets


def test_placement_nonuniform_scale(head):
    V, F, anchors = head["V"], head["F"], head["anchors"]
    parts = im.build_inner_mouth(V, F, anchors, archetype="human")
    cav = im.mouth_cavity(V, anchors)
    sx = cav["size"][0]
    pV = parts[0][0]  # teeth_upper
    u = (pV - cav["center"]) @ cav["right"]
    # the dental arch spans the mouth width (up to the ARAP correction)
    assert abs((u.max() - u.min()) - sx) < 0.15 * sx


def test_arap_rbf_interpolates(head):
    """The RBF warp interpolates exactly at the arch control points."""
    import omnifacerig_repro.inner_mouth as inner
    # build the canonical->placed arch as the pipeline does
    ict = im.load_ict_facekit()
    assert ict is not None, "vendored ICT-FaceKit assets must be present"
    Vu, Fu, Vl, Fl = im._split_teeth_upper_lower(*ict["teeth"])
    raw = {"teeth_upper": (Vu, Fu), "teeth_lower": (Vl, Fl),
           "gums_tongue": ict["gums_tongue"]}
    canon, _ = im._canonical_parts(raw)
    cav = im.mouth_cavity(head["V"], head["anchors"])
    placed = im._place_parts(canon, cav)
    arch, archF, _ = im._control_arch(placed, cav, 5)
    rng = np.random.default_rng(0)
    target = arch + rng.normal(0, 0.02, arch.shape)
    out = im._gaussian_rbf(arch, target, arch)
    assert np.abs(out - target).max() < 1e-4


def test_sdf_no_penetration_after_build(head):
    V, F, anchors = head["V"], head["F"], head["anchors"]
    parts = im.build_inner_mouth(V, F, anchors, archetype="human")
    cav = im.mouth_cavity(V, anchors)
    pdict = {n: (pV, pF) for pV, pF, n in parts}
    stats = im.face_penetration_stats(pdict, V, F, cav["center"], 1.6 * cav["size"][0])
    assert stats["face_verts_inside"] == 0
    # forced penetration is detected and the paper-variant pushes the face out
    pushed = {n: (pV + 0.25 * cav["size"][2] * cav["outward"], pF)
              for n, (pV, pF) in pdict.items()}
    stats2 = im.face_penetration_stats(pushed, V, F, cav["center"], 1.6 * cav["size"][0])
    assert stats2["face_verts_inside"] > 0
    assert stats2["max_penetration"] > 0
    V2, moved = im.push_face_out_of_teeth(V, F, pushed)
    assert moved.sum() >= stats2["face_verts_inside"]
    # pushed verts moved outward (away from the teeth)
    assert np.linalg.norm(V2[moved] - V[moved], axis=1).min() > 0


def test_archetypes(head):
    V, F, anchors = head["V"], head["F"], head["anchors"]
    canine = im.build_inner_mouth(V, F, anchors, archetype="canine")
    human = im.build_inner_mouth(V, F, anchors, archetype="human")
    monster = im.build_inner_mouth(V, F, anchors, archetype="monster")
    flat = im.build_inner_mouth(V, F, anchors, archetype="flat")
    assert flat == []
    for parts in (canine, monster):
        assert [n for _, _, n in parts] == ["teeth_upper", "teeth_lower", "gums_tongue"]
    # canine fangs: upper tips extend further down, lower tips further up
    assert canine[0][0][:, 1].min() < human[0][0][:, 1].min() - 1e-3
    assert canine[1][0][:, 1].max() > human[1][0][:, 1].max() + 1e-3
    # monster: jagged crowns differ from the human shape
    assert not np.allclose(monster[0][0], human[0][0])


def test_compute_part_morphs_jaw(head):
    V, F, anchors = head["V"], head["F"], head["anchors"]
    parts = im.build_inner_mouth(V, F, anchors, archetype="human")
    n = len(V)
    morphs = {name: np.zeros((n, 3)) for name in ARKIT_52}
    lower = V[:, 1] < anchors["mouth_center"][1]
    morphs["jawOpen"][lower] = np.array([0.0, -0.05, 0.02])
    pm = im.compute_part_morphs(parts, morphs, anchors["mouth_center"], V)
    assert "teeth_upper" not in pm or "jawOpen" not in pm.get("teeth_upper", {})
    d_lo = pm["teeth_lower"]["jawOpen"]
    nz = np.any(np.abs(d_lo) > 1e-9, axis=1)
    assert nz.all()  # lower teeth follow fully
    assert np.allclose(d_lo[nz].mean(axis=0), [0.0, -0.05, 0.02], atol=1e-6)
    # gums_tongue follows in its lower half only
    d_gt = pm["gums_tongue"]["jawOpen"]
    nz_gt = np.any(np.abs(d_gt) > 1e-9, axis=1)
    assert 0 < nz_gt.sum() < len(d_gt)


def test_attach_to_glb(head, tmp_path):
    V, F, anchors = head["V"], head["F"], head["anchors"]
    parts = im.build_inner_mouth(V, F, anchors, archetype="human")
    n = len(V)
    morphs = {name: np.zeros((n, 3)) for name in ARKIT_52}
    morphs["jawOpen"][V[:, 1] < anchors["mouth_center"][1]] = np.array([0.0, -0.05, 0.02])
    pm = im.compute_part_morphs(parts, morphs, anchors["mouth_center"], V)

    gltf = build_gltf(V, F, morphs, name="head")
    n0 = len(gltf.meshes[0].primitives)
    im.attach_to_glb(gltf, parts, pm)
    assert len(gltf.meshes[0].primitives) == n0 + len(parts)
    for p in gltf.meshes[0].primitives:
        assert len(p.targets) == len(ARKIT_52)

    path = os.path.join(tmp_path, "inner.glb")
    write_glb(path, gltf)
    loaded = load_glb(path)
    assert len(loaded.meshes[0].primitives) == n0 + len(parts)

    # binary round-trip: part positions are bit-identical
    blob = loaded.binary_blob()
    NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    DT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}

    def read_acc(i):
        acc = loaded.accessors[i]
        bv = loaded.bufferViews[acc.bufferView]
        arr = np.frombuffer(blob, dtype=DT[acc.componentType],
                            count=acc.count * NCOMP[acc.type], offset=bv.byteOffset)
        return arr.reshape(-1, NCOMP[acc.type]).copy()

    for k, p in enumerate(loaded.meshes[0].primitives[n0:]):
        back = read_acc(p.attributes.POSITION)
        assert np.abs(back - parts[k][0].astype(np.float32)).max() == 0.0


def test_procedural_fallback(head, monkeypatch):
    """Without the ICT assets the pipeline falls back to procedural geometry."""
    monkeypatch.setattr(im, "load_ict_facekit", lambda: None)
    parts = im.build_inner_mouth(head["V"], head["F"], head["anchors"], archetype="human")
    assert [n for _, _, n in parts] == ["teeth_upper", "teeth_lower", "gums_tongue"]
    for pV, pF, _name in parts:
        assert pF.min() >= 0 and pF.max() < len(pV)
        assert len(pV) > 40
