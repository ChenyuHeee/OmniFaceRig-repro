"""End-to-end pipeline orchestrator + CPU-runnable demo.

Pipeline (paper arXiv:2606.08043):

    image -> [image-to-3D front: TRELLIS on A100, TODO] -> static mesh
           -> Stage 1: face template fitting (template_fit.py)
           -> Stage 2: face fusion (boolean cut + weld), inner-mouth
                       (ARAP + SDF, needs inner-mouth archetypes), FACS
                       blendshape transfer (geometry.deformation_transfer
                       + delta_mush) -> rigged GLB (glb_export.py)

`run_demo()` runs the full CPU-side pipeline on a synthetic head mesh:
template fitting -> 52 ARKit morph targets -> simple skeleton -> viseme
animation, and writes outputs/demo.glb. This exercises every stage that can
run without the A100 models; the image front and the official templates
plug in later.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree, ConvexHull

from . import arkit52
from .geometry import deformation_transfer, delta_mush, vertex_normals
from .template_fit import fit_template
from .glb_export import build_gltf, write_glb
from . import lip_sync

# ---------------------------------------------------------------------------
# Synthetic assets (stand-ins until the official T-pose glb / templates land)
# ---------------------------------------------------------------------------

def ellipsoid_mesh(rx: float, ry: float, rz: float, n_lat: int = 24, n_lon: int = 48):
    """Non-degenerate ellipsoid mesh (pole fans + quad rings), facing +z.

    Returns (V, F). All triangles have non-zero area (no pole degeneracy).
    """
    th = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    # vertices: [north pole, south pole, rings...]
    V = [np.array([0.0, 0.0, rz]), np.array([0.0, 0.0, -rz])]
    for t in th[1:-1]:
        ct, st = np.cos(t), np.sin(t)
        for ph in lon:
            V.append((rx * st * np.cos(ph), ry * st * np.sin(ph), rz * ct))
    V = np.asarray(V, dtype=float)
    ring0, ring1 = 2, 2 + (n_lat - 3) * n_lon  # first and last latitude ring
    F: list[list[int]] = []
    # north pole fan
    for j in range(n_lon):
        F.append([0, ring0 + j, ring0 + (j + 1) % n_lon])
    # quad rings
    for i in range(n_lat - 3):
        a = ring0 + i * n_lon
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            F.append([a + j, a + j2, a + n_lon + j2])
            F.append([a + j, a + n_lon + j2, a + n_lon + j])
    # south pole fan
    for j in range(n_lon):
        F.append([1, ring1 + (j + 1) % n_lon, ring1 + j])
    return V, np.asarray(F, dtype=int)


def _nearest(V: np.ndarray, p: np.ndarray) -> int:
    return int(cKDTree(V).query(np.asarray(p, dtype=float))[1])


def head_keypoints(V: np.ndarray, rx: float, ry: float, rz: float) -> tuple[np.ndarray, np.ndarray]:
    """Face feature keypoints on the ellipsoid head (eyes, mouth, boundary).

    Stand-in for the segmentation+landmark keypoint stage (paper Sec. 3.4),
    which runs on the A100 stack (MediaPipe / fine-tuned Sapiens / SAM).
    """
    pts = {
        "eye_left_outer": (-0.42 * rx, 0.18 * ry, 0.55 * rz),
        "eye_left_inner": (-0.22 * rx, 0.18 * ry, 0.62 * rz),
        "eye_right_inner": (0.22 * rx, 0.18 * ry, 0.62 * rz),
        "eye_right_outer": (0.42 * rx, 0.18 * ry, 0.55 * rz),
        "mouth_left": (-0.28 * rx, -0.18 * ry, 0.45 * rz),
        "mouth_right": (0.28 * rx, -0.18 * ry, 0.45 * rz),
        "mouth_top": (0.0, -0.06 * ry, 0.52 * rz),
        "mouth_bottom": (0.0, -0.32 * ry, 0.42 * rz),
        "brow_left": (-0.32 * rx, 0.38 * ry, 0.62 * rz),
        "brow_right": (0.32 * rx, 0.38 * ry, 0.62 * rz),
    }
    idx = np.array([_nearest(V, p) for p in pts.values()], dtype=int)
    return idx, np.asarray(list(pts.values()), dtype=float)


# ---------------------------------------------------------------------------
# Stage 2 Step 1: face fusion (boolean cut + nearest-point welding)
# ---------------------------------------------------------------------------

def merge_face_region(
    in_V: np.ndarray, in_F: np.ndarray,
    fit_V: np.ndarray, fit_F: np.ndarray,
    keypoint_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Replace the input face region with the fitted template mesh.

    1. convex hull of the keypoints -> old face region to cut out
    2. template boundary vertices welded to the remaining input boundary
       (nearest-point welding, paper Sec. 3.6.1; degenerate faces dropped)

    Returns (merged V, merged F, remap) where remap maps template vertex
    index -> merged mesh vertex index (non-welded template vertices).
    Production version uses the exact boolean cut on the original mesh
    (trimesh boolean with manifold backend) + ICT-FaceKit inner-mouth
    attachment; the geometry is identical here.
    """
    in_V = np.asarray(in_V, dtype=float)
    fit_V = np.asarray(fit_V, dtype=float)
    hull = ConvexHull(in_V[keypoint_idx])
    A, b = hull.equations[:, :3], hull.equations[:, 3]
    # The keypoint hull lies inside a convex head surface: inflate it so the
    # face-region faces (centroids on the surface) are actually removed.
    margin = 0.08 * float(np.ptp(in_V))
    centers = (in_V[in_F[:, 0]] + in_V[in_F[:, 1]] + in_V[in_F[:, 2]]) / 3.0
    inside = (centers @ A.T + b < margin).all(axis=1)
    keep = ~inside

    # boundary of the hole: edges of removed faces that are also edges of kept faces
    rem_f = in_F[inside]
    keep_f = in_F[keep]
    hole_edges = set()
    for f in rem_f:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            hole_edges.add(tuple(sorted((int(e[0]), int(e[1])))))
    keep_edge_set = set()
    for f in keep_f:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            keep_edge_set.add(tuple(sorted((int(e[0]), int(e[1])))))
    boundary = sorted(hole_edges - keep_edge_set)
    if not boundary:
        raise ValueError("no boundary between face region and remaining mesh")
    hole_verts = sorted({v for e in boundary for v in e})

    # template boundary vertices (edges with < 2 incident faces)
    fit_edge_count: dict[tuple, int] = {}
    for f in fit_F:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            k = tuple(sorted((int(e[0]), int(e[1]))))
            fit_edge_count[k] = fit_edge_count.get(k, 0) + 1
    tpl_boundary = sorted({v for e, c in fit_edge_count.items() if c == 1 for v in e})

    # weld: snap template boundary vertices onto nearest input hole vertices
    tree = cKDTree(in_V[hole_verts])
    d, j = tree.query(fit_V[tpl_boundary])
    weld_map = {tv: hole_verts[jj] for tv, jj in zip(tpl_boundary, j)}

    # remap template faces: welded vertices reuse input indices, others get new ones
    remap: dict[int, int] = {}
    next_idx = len(in_V)
    out_F: list[list[int]] = []
    for f in fit_F:
        nf = []
        ok = True
        for v in f:
            v = int(v)
            if v in weld_map:
                nf.append(weld_map[v])
            else:
                if v not in remap:
                    remap[v] = next_idx
                    next_idx += 1
                nf.append(remap[v])
        if len(set(nf)) == 3:
            out_F.append(nf)
    out_V = np.vstack([in_V, fit_V[list(remap.keys())]])
    return out_V, np.asarray(out_F, dtype=int), remap


# ---------------------------------------------------------------------------
# Stage 2 Step 4: synthetic canonical shapes + ARKit morph transfer
# ---------------------------------------------------------------------------

def synthesize_canonical_shapes(tpl_V: np.ndarray, rx, ry, rz) -> dict[str, np.ndarray]:
    """Stand-in canonical FACS shapes on the template.

    Real pipeline: shapes come from the canonical rigged template (ICT-FaceKit
    ARKit set, vendored reference) via deformation transfer; here we synthesize
    a few semantically-correct shapes on the synthetic head.
    """
    n = len(tpl_V)
    zero = np.zeros((n, 3))
    shapes = {name: zero.copy() for name in arkit52.ARKIT_52}

    y_mouth = -0.18 * ry
    lower = tpl_V[:, 1] < y_mouth
    jaw = np.zeros((n, 3))
    factor = 0.5 + 0.5 * (y_mouth - tpl_V[lower, 1]) / (2 * ry)
    jaw[lower] = factor[:, None] * np.array([0.0, -0.22 * ry, 0.10 * rz])
    shapes["jawOpen"] = jaw
    shapes["jawLeft"] = jaw * np.array([1.0, 0.0, 0.0]) * 0.4
    shapes["jawRight"] = -jaw * np.array([1.0, 0.0, 0.0]) * 0.4
    shapes["mouthClose"] = -jaw * 0.4

    for side, sx in (("Left", -1.0), ("Right", 1.0)):
        c = np.array([sx * 0.32 * rx, 0.18 * ry, 0.58 * rz])
        d = tpl_V - c
        r = np.linalg.norm(d, axis=1)
        w = np.exp(-(r / (0.16 * rx)) ** 2)
        blink = np.zeros((n, 3))
        blink[:, 1] = w * (c[1] - tpl_V[:, 1]) * 0.9
        shapes[f"eyeBlink{side}"] = blink
        shapes[f"eyeWide{side}"] = -blink * 0.5
        shapes[f"eyeLookUp{side}"] = w[:, None] * np.array([0.0, 0.0, 0.08 * rz])
        shapes[f"eyeLookDown{side}"] = w[:, None] * np.array([0.0, 0.0, -0.08 * rz])

    for side, sx in (("Left", -1.0), ("Right", 1.0)):
        c = np.array([sx * 0.3 * rx, -0.18 * ry, 0.46 * rz])
        d = tpl_V - c
        r = np.linalg.norm(d, axis=1)
        w = np.exp(-(r / (0.2 * rx)) ** 2)
        sm = np.zeros((n, 3))
        sm[:, 0] = w * sx * 0.10 * rx
        sm[:, 1] = w * 0.04 * ry
        shapes[f"mouthSmile{side}"] = sm
        shapes[f"mouthFrown{side}"] = -sm * np.array([1.0, 0.6, 0.0])
        shapes[f"mouthStretch{side}"] = sm * np.array([1.6, 0.3, 0.0])
        shapes[f"mouthUpperUp{side}"] = w[:, None] * np.array([0.0, 0.10 * ry, 0.0])
        shapes[f"mouthLowerDown{side}"] = w[:, None] * np.array([0.0, -0.10 * ry, 0.0])

    for side, sx in (("Left", -1.0), ("Right", 1.0)):
        c = np.array([sx * 0.32 * rx, 0.34 * ry, 0.62 * rz])
        d = tpl_V - c
        r = np.linalg.norm(d, axis=1)
        w = np.exp(-(r / (0.14 * rx)) ** 2)
        shapes[f"browInnerUp"] = shapes.get("browInnerUp", zero.copy()) + w[:, None] * np.array([0.0, 0.08 * ry, 0.0])
        shapes[f"browDown{side}"] = w[:, None] * np.array([0.0, -0.06 * ry, 0.0])

    shapes["mouthPucker"] = shapes["mouthSmileLeft"] * -0.5 + shapes["mouthSmileRight"] * -0.5
    shapes["mouthFunnel"] = shapes["mouthPucker"] * np.array([0.8, 1.4, 0.4])
    shapes["tongueOut"] = shapes["mouthLowerDownLeft"] * 0.3 + shapes["mouthLowerDownRight"] * 0.3
    shapes["tongueOut"][:, 2] += 0.12 * rz * np.exp(-((tpl_V[:, 0] / (0.15 * rx)) ** 2)) * (
        tpl_V[:, 1] < y_mouth)
    return shapes


def build_arkit_morphs(
    tpl_V: np.ndarray, F: np.ndarray, fit_V: np.ndarray,
    canonical: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Transfer canonical shapes to the fitted mesh (same topology) and
    Delta-Mush-smooth the result. Returns {name: delta} for all 52 shapes."""
    out: dict[str, np.ndarray] = {}
    for name in arkit52.ARKIT_52:
        src_def = tpl_V + canonical[name]
        transferred = deformation_transfer(tpl_V, src_def, F, fit_V)
        smooth = delta_mush(tpl_V, F, transferred, iterations=2, alpha=0.6)
        out[name] = smooth - fit_V
    return out


# ---------------------------------------------------------------------------
# Simple skeleton (demo) - production body rig comes from the official glb
# ---------------------------------------------------------------------------

def demo_skeleton(V: np.ndarray, ry: float, rz: float):
    """head/jaw/eyeL/eyeR joints + proximity-based skin weights."""
    joints = [
        ("head", np.eye(4)),
        ("jaw", np.array([[1, 0, 0, 0], [0, 1, 0, -0.55 * ry], [0, 0, 1, 0], [0, 0, 0, 1]])),
        ("eyeL", np.array([[1, 0, 0, -0.32 * 0.9], [0, 1, 0, 0.18 * 0.9], [0, 0, 1, 0.58 * 0.9], [0, 0, 0, 1]])),
        ("eyeR", np.array([[1, 0, 0, 0.32 * 0.9], [0, 1, 0, 0.18 * 0.9], [0, 0, 1, 0.58 * 0.9], [0, 0, 0, 1]])),
    ]
    names = [j[0] for j in joints]
    centers = [j[1][:3, 3] for j in joints]
    # gaussian falloff weights
    d = np.stack([np.linalg.norm(V - c, axis=1) for c in centers], axis=1)  # (n,4)
    sigma = np.array([0.9, 0.45, 0.25, 0.25])
    W = np.exp(-(d / sigma) ** 2)
    W = W / W.sum(axis=1, keepdims=True)
    # keep only top-2 weights per vertex
    k = 2
    jidx = np.zeros((len(V), k), dtype=np.uint16)
    jw = np.zeros((len(V), k), dtype=np.float32)
    for i in range(len(V)):
        top = np.argsort(W[i])[::-1][:k]
        jidx[i] = top
        s = W[i, top].sum()
        jw[i] = W[i, top] / s if s > 0 else 0
    return {"joint_names": names, "joint_transforms": [j[1] for j in joints],
            "joint_indices": jidx, "joint_weights": jw}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo(out_path: str = "outputs/demo.glb", seed: int = 0) -> dict:
    """Full CPU pipeline demo: synthetic head -> rigged GLB with ARKit 52
    morphs, skeleton and a Chinese viseme animation."""
    import os
    rng = np.random.default_rng(seed)

    # input "image-to-3D" result: a slightly different head
    rx, ry, rz = 1.0, 1.15, 1.0
    in_V, in_F = ellipsoid_mesh(rx * 0.96, ry * 1.05, rz * 0.98)
    in_V += rng.normal(0, 0.01, in_V.shape)

    # template (stand-in for the official face template)
    tpl_V, tpl_F = ellipsoid_mesh(rx, ry, rz)
    tpl_idx, _ = head_keypoints(tpl_V, rx, ry, rz)
    in_idx, in_kp = head_keypoints(in_V, rx * 0.96, ry * 1.05, rz * 0.98)
    in_kp = in_kp + rng.normal(0, 0.015, in_kp.shape)  # keypoint noise

    # Stage 1
    fit_V, info1 = fit_template(tpl_V, tpl_F, tpl_idx, in_kp)
    corr_err = float(np.linalg.norm(fit_V[tpl_idx] - in_kp, axis=1).mean())

    # Stage 2: face fusion
    merged_V, merged_F, remap = merge_face_region(in_V, in_F, fit_V, tpl_F, in_idx)
    # canonical shapes live on the template topology; transfer to fitted mesh
    canonical = synthesize_canonical_shapes(tpl_V, rx, ry, rz)
    fit_morphs = build_arkit_morphs(tpl_V, tpl_F, fit_V, canonical)

    # map fitted-template deltas onto the merged mesh (input verts stay 0)
    merged_morphs: dict[str, np.ndarray] = {}
    for name, delta in fit_morphs.items():
        full = np.zeros((len(merged_V), 3))
        for tv, mv in remap.items():
            full[mv] = delta[tv]
        merged_morphs[name] = full

    # skeleton + animation (Chinese lip-sync demo)
    skin = demo_skeleton(merged_V, ry, rz)
    visemes = lip_sync.zh_text_to_visemes("你好世界")
    dt = 0.18
    times = np.arange(len(visemes) + 1, dtype=float) * dt
    weights = np.zeros((len(visemes) + 1, 52))
    for i, v in enumerate(visemes):
        vec = np.array(arkit52.arkit_vector(lip_sync.viseme_to_arkit(v)))
        weights[i] = vec
    anim = {"times": times, "weights": weights}

    normals = vertex_normals(merged_V, merged_F)
    gltf = build_gltf(merged_V, merged_F, merged_morphs, normals=normals,
                      skin=skin, animation=anim, name="demo_character")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_glb(out_path, gltf)

    # validation stats
    from .geometry import face_normals as fn
    nrm = fn(fit_V, tpl_F)
    flips = int((np.einsum("ij,ij->i", fn(tpl_V, tpl_F), nrm) < 0).sum())
    return {
        "out": out_path, "vertices": len(merged_V), "faces": len(merged_F),
        "corr_err": corr_err, "flipped_triangles_after_fit": flips,
        "morph_targets": len(merged_morphs), "visemes": visemes,
        "fit": info1,
    }


if __name__ == "__main__":
    import json
    stats = run_demo()
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
