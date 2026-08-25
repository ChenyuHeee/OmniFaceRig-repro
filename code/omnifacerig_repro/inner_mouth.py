"""Inner-mouth assets: teeth / gums / tongue synthesis + placement (paper §3.6.2).

The input character mesh contains no oral cavity, so the inner-mouth geometry
(teeth, gums, tongue) is synthesized from a small library of template
archetypes (Table 6: human / canine / monster / flat) rather than registered.

Pipeline (paper §3.6.2, reproduced):

    1. load the Teeth + Gums-and-tongue sub-meshes from the vendored
       ICT-FaceKit neutral mesh (MIT, USC-ICT/ICT-FaceKit) - or fall back to
       procedurally generated placeholder geometry when the assets are absent;
    2. archetype transform in a canonical frame (canine fangs / monster
       jagged teeth / flat = no oral structure);
    3. mouth-cavity placement: rigid transform with non-uniform scale from the
       mouth anchors (mouth_center / mouth_left / mouth_right / mouth_top /
       mouth_bottom), so the dental arch matches THIS character's mouth box;
    4. ARAP initial placement: a small control arch (2 rows x N columns of the
       dental arch) is deformed with geometry.arap_deform, pinning the arch
       corners to the mouth-cavity corners, and the per-vertex displacement is
       propagated to the full parts with a Gaussian RBF warp;
    5. SDF penetration refinement: the convex hull of the teeth is converted
       to an SDF (scipy half-space form, dependency-free); face vertices
       intersecting the teeth volume are counted and the whole mouth block is
       iteratively pushed deeper into the cavity along the SDF gradient
       direction until the face is clear (non-destructive to the accepted
       outer surface; the paper's outward face-push variant is also provided
       as push_face_out_of_teeth).

API:
    build_inner_mouth(V, F, anchors, archetype="human", options=None)
        -> list[(mesh_V, mesh_F, name)]   # name: teeth_upper / teeth_lower / gums_tongue
    attach_to_glb(gltf, parts, part_morphs=None) -> pygltflib.GLTF2
    compute_part_morphs(parts, morphs, mouth_center, face_V, options=None) -> dict
    mouth_cavity(V, anchors, options=None, F=None) -> dict
    face_penetration_stats(parts, face_V, face_F, ...) -> dict
    push_face_out_of_teeth(face_V, face_F, parts, margin=1e-3) -> (V', moved_mask)

The inner mouth is emitted as independent mesh primitives sharing the outer
mesh's morph-weight list, mirroring the official FINAL_WORK_DEMO.glb layout
(one mesh node, several primitives: body + teeth / gums / tongue).
"""

from __future__ import annotations

import functools
import os
import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, ConvexHull

from .geometry import arap_deform, vertex_normals

try:  # pygltflib is a hard dependency of the package; import once here
    import pygltflib
except Exception:  # pragma: no cover - import error surfaces at call time
    pygltflib = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHETYPES = ("human", "canine", "monster", "flat")

# ICT-FaceKit (MIT) neutral mesh + group index ranges, from the upstream
# README topology table (1-based inclusive start / exclusive end as printed).
_ICT_OBJ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "vendor", "ict_facekit", "FaceXModel", "generic_neutral_mesh.obj",
)
_ICT_GROUPS = {
    # name      : (vertex range, polygon range) as printed in the README
    "gums_tongue": ((14062, 17038), (14034, 17005)),
    "teeth":       ((17039, 21450), (17006, 21495)),
}

# Outer-mesh ARKit shapes that drive the lower jaw; the lower halves of the
# inner-mouth parts follow the average outer-mesh delta of these shapes.
JAW_SHAPES = (
    "jawOpen", "jawLeft", "jawRight", "jawForward", "mouthClose",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthFunnel", "mouthPucker",
)

DEFAULT_OPTIONS = {
    "mouth_width_frac": 0.80,   # mouth width / inter-eye distance (fallback)
    "mouth_height_frac": 0.55,  # mouth height / mouth width (fallback)
    "depth_frac": 0.55,         # mouth-cavity depth / mouth width
    "back_gap_frac": 0.12,      # gap between lip plane and teeth front / depth
    "fang_gain": 0.45,          # canine fang elongation (canonical units)
    "arap_iterations": 40,
    "arap_n_cols": 5,
    "sdf_margin": 1e-3,
    "sdf_max_iter": 3,
    "sdf_radius_frac": 1.6,     # penetration-check radius / mouth width
    "morph_scale": 1.0,
}


# ---------------------------------------------------------------------------
# OBJ IO + ICT-FaceKit loading
# ---------------------------------------------------------------------------

def _read_obj(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ reader: 'v' positions + 'f' triangle indices (0-based)."""
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                V.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f":
                F.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    return np.asarray(V, dtype=float), np.asarray(F, dtype=int)


def _extract_group(V: np.ndarray, F: np.ndarray, v_range, f_range
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Sub-mesh by polygon range (1-based as printed); remap used vertices.

    The ICT groups share boundary vertices, so we keep the polygon range and
    re-index the referenced vertex set (the printed vertex range is advisory)."""
    faces = F[f_range[0] - 1:f_range[1] - 1]
    used = np.unique(faces)
    remap = {int(g): i for i, g in enumerate(used)}
    Fs = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]] for a, b, c in faces])
    return V[used], Fs


@functools.lru_cache(maxsize=1)
def load_ict_facekit(path: str | None = None
                     ) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Load the ICT-FaceKit Teeth / Gums-and-tongue sub-meshes (MIT).

    Returns {"teeth": (V, F), "gums_tongue": (V, F)} in the ICT neutral-mesh
    frame (units ~ cm, +y up, +z forward), or None when the vendored asset is
    missing (the pipeline then falls back to procedural geometry)."""
    path = path or _ICT_OBJ
    if not os.path.exists(path):
        return None
    V, F = _read_obj(path)
    out = {}
    for name, (v_range, f_range) in _ICT_GROUPS.items():
        out[name] = _extract_group(V, F, v_range, f_range)
    return out


# ---------------------------------------------------------------------------
# Procedural placeholder geometry (documented fallback, not scanned identity)
# ---------------------------------------------------------------------------

def _box_arch(row_y: float, n_cols: int = 6, width: float = 0.9,
              depth: float = 0.5, sweep: float = 1.1,
              tooth_w: float = 0.10, tooth_h: float = 0.42,
              tooth_d: float = 0.24) -> tuple[np.ndarray, np.ndarray]:
    """A U-shaped row of box "teeth" along a half-ellipse arch.

    Canonical frame: bbox ~ [-0.5, 0.5]^3, y up, +z forward. Returns (V, F);
    each tooth is an 8-corner box (12 triangles) and the arch ends sweep back
    in z like molars."""
    V: list[np.ndarray] = []
    F: list[list[int]] = []
    for i in range(n_cols):
        t = -sweep + 2.0 * sweep * i / (n_cols - 1)
        cx, cz = 0.5 * width * np.cos(t), 0.5 * depth * np.sin(t)
        lo = np.array([cx - tooth_w / 2, row_y - tooth_h / 2, cz - tooth_d / 2])
        hi = np.array([cx + tooth_w / 2, row_y + tooth_h / 2, cz + tooth_d / 2])
        base = len(V)
        corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                            [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                            [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                            [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
        V.append(corners)
        for q in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                  (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5)):
            a, b, c, d = (base + x for x in q)
            F += [[a, b, c], [a, c, d]]
    return np.vstack(V), np.asarray(F, dtype=int)


def _gum_band(row_y: float, n_cols: int = 6, width: float = 0.9,
              depth: float = 0.5, sweep: float = 1.1,
              band_h: float = 0.16, band_d: float = 0.5
              ) -> tuple[np.ndarray, np.ndarray]:
    """Curved gum band along the same arch as _box_arch (extruded strip)."""
    V, F = [], []
    ts = np.linspace(-sweep, sweep, n_cols)
    for zf in (-1.0, 1.0):  # back / front rows of the strip
        for t in ts:
            cx, cz = 0.5 * width * np.cos(t), 0.5 * depth * np.sin(t)
            V.append([cx, row_y - band_h / 2, cz + zf * band_d / 2])
            V.append([cx, row_y + band_h / 2, cz + zf * band_d / 2])
    V = np.asarray(V)
    for i in range(n_cols - 1):
        for zf in range(2):
            a = zf * 2 * n_cols + 2 * i
            b = a + 2
            F.append([a, b, b + 1]); F.append([a, b + 1, a + 1])
    return V, np.asarray(F, dtype=int)


def _ellipsoid(rx: float, ry: float, rz: float, n_lat: int = 12, n_lon: int = 20
               ) -> tuple[np.ndarray, np.ndarray]:
    """Small ellipsoid (no pole degeneracy), +z forward."""
    th = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    V = [np.array([0.0, 0.0, rz]), np.array([0.0, 0.0, -rz])]
    for t in th[1:-1]:
        ct, st = np.cos(t), np.sin(t)
        for ph in lon:
            V.append((rx * st * np.cos(ph), ry * st * np.sin(ph), rz * ct))
    V = np.asarray(V, dtype=float)
    ring0, ring1 = 2, 2 + (n_lat - 3) * n_lon
    F: list[list[int]] = []
    for j in range(n_lon):
        F.append([0, ring0 + j, ring0 + (j + 1) % n_lon])
    for i in range(n_lat - 3):
        a = ring0 + i * n_lon
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            F.append([a + j, a + j2, a + n_lon + j2])
            F.append([a + j, a + n_lon + j2, a + n_lon + j])
    for j in range(n_lon):
        F.append([1, ring1 + (j + 1) % n_lon, ring1 + j])
    return V, np.asarray(F, dtype=int)


def procedural_parts() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Procedural placeholder inner-mouth geometry (canonical frame).

    Used only when the ICT-FaceKit assets are unavailable. Simplified boxes /
    bands / ellipsoid (documented placeholder, NOT derived from any scanned
    identity). Returns the same part names as the ICT path so the downstream
    pipeline is identical."""
    teeth_u, teeth_u_F = _box_arch(0.25, n_cols=6)
    teeth_l, teeth_l_F = _box_arch(-0.25, n_cols=6)
    gum_u_V, gum_u_F = _gum_band(0.12)
    gum_l_V, gum_l_F = _gum_band(-0.12)
    tV, tF = _ellipsoid(0.35, 0.12, 0.5)
    tongue = (tV + np.array([0.0, -0.15, 0.10]), tF)
    # merge gums + tongue into one "gums_tongue" part (ICT group structure)
    off = len(gum_u_V) + len(gum_l_V)
    gtV = np.vstack([gum_u_V, gum_l_V, tongue[0]])
    gtF = np.vstack([gum_u_F, gum_l_F + len(gum_u_V), tongue[1] + off])
    return {"teeth_upper": (teeth_u, teeth_u_F), "teeth_lower": (teeth_l, teeth_l_F),
            "gums_tongue": (gtV, gtF)}


# ---------------------------------------------------------------------------
# Canonical frame + part splitting
# ---------------------------------------------------------------------------

def _components(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Connected-component label per vertex."""
    row = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    col = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = coo_matrix((np.ones(len(row)), (row, col)), shape=(len(V), len(V))).tocsr()
    _, labels = connected_components(A, directed=False)
    return labels


def _split_teeth_upper_lower(V: np.ndarray, F: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the ICT teeth group into upper / lower rows.

    The ICT teeth are ~33 disconnected components (one per tooth); each
    component is classified by its centroid height relative to the median
    component height. Returns (Vu, Fu, Vl, Fl)."""
    labels = _components(V, F)
    comps = np.unique(labels)
    cy = np.array([V[labels == c, 1].mean() for c in comps])
    med = np.median(cy)
    upper = np.isin(labels, comps[cy > med])
    lower = ~upper
    if not upper.any() or not lower.any():
        raise ValueError("teeth split failed: no upper/lower rows found")
    remap = {int(i): j for j, i in enumerate(np.where(upper)[0])}
    Vu = V[upper]
    Fu = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]]
                   for a, b, c in F if upper[a] and upper[b] and upper[c]])
    remap = {int(i): j for j, i in enumerate(np.where(lower)[0])}
    Vl = V[lower]
    Fl = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]]
                   for a, b, c in F if lower[a] and lower[b] and lower[c]])
    return Vu, Fu, Vl, Fl


def _canonical_parts(parts: dict[str, tuple[np.ndarray, np.ndarray]]
                     ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Center parts on the teeth centroid and scale by the teeth bbox.

    Canonical frame: teeth bbox ~ [-0.5, 0.5]^3, y up, +z forward. Returns
    (canonical parts, frame) with frame = {"center": (3,), "size": (3,)}."""
    teeth = parts.get("teeth_upper", parts.get("teeth"))
    if teeth is None:
        raise ValueError("no teeth part to define the canonical frame")
    TV = teeth[0]
    center = TV.mean(axis=0)
    size = TV.max(axis=0) - TV.min(axis=0)
    size[size < 1e-9] = 1.0
    out = {}
    for name, (V, F) in parts.items():
        out[name] = ((V - center) / size, F)
    return out, {"center": center, "size": size}


# ---------------------------------------------------------------------------
# Archetype transforms (paper Table 6), applied in canonical space
# ---------------------------------------------------------------------------

def _tip_weight(y: np.ndarray, upper: bool) -> np.ndarray:
    """0 at the gum/root end of a tooth row, 1 at the incisal tip.

    For the upper teeth the tip is at the bottom (min y); for the lower teeth
    the tip is at the top (max y)."""
    if upper:
        return np.clip((y - y.max()) / (y.min() - y.max() + 1e-9), 0.0, 1.0)
    return np.clip((y - y.min()) / (y.max() - y.min() + 1e-9), 0.0, 1.0)


def _canine_fangs(parts: dict, gain: float) -> dict:
    """Elongate the four canine columns (|x| ~ 0.32) downward/upward into
    fangs and stretch the arch along the muzzle (+z) for long-muzzled
    characters (dogs / wolves / foxes)."""
    out = {}
    for name, (V, F) in parts.items():
        V = np.asarray(V, dtype=float).copy()
        ax = np.abs(V[:, 0])
        w_col = np.exp(-((ax - 0.32) / 0.08) ** 2)  # peak at the canine columns
        if "upper" in name:
            tip = _tip_weight(V[:, 1], upper=True)
            V[:, 1] -= gain * w_col * tip   # fangs point down
        elif "lower" in name:
            tip = _tip_weight(V[:, 1], upper=False)
            V[:, 1] += gain * w_col * tip   # fangs point up
        V[:, 2] *= 1.20  # longer muzzle
        out[name] = (V, F)
    return out


def _monster_jagged(parts: dict, seed: int = 42) -> dict:
    """Stylized irregular teeth: deterministic sawtooth + seeded noise on the
    crown tips (fantasy creatures with non-standard teeth)."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, (V, F) in parts.items():
        V = np.asarray(V, dtype=float).copy()
        if "teeth" in name:
            upper = "upper" in name
            tip = _tip_weight(V[:, 1], upper=upper)
            jag = 0.15 * np.sin(V[:, 0] * 22.0) + 0.08 * rng.normal(size=len(V))
            V[:, 1] += (-jag * tip if upper else jag * tip)
        out[name] = (V, F)
    return out


def apply_archetype(parts: dict, archetype: str, options: dict | None = None) -> dict:
    """Apply the Table 6 archetype transform in canonical space. 'flat'
    returns an empty dict (characters with minimal / absent oral structure)."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    if archetype == "human":
        return parts
    if archetype == "flat":
        return {}
    if archetype == "canine":
        return _canine_fangs(parts, opts["fang_gain"])
    if archetype == "monster":
        return _monster_jagged(parts)
    raise ValueError(f"unknown archetype {archetype!r} (choose from {ARCHETYPES})")


# ---------------------------------------------------------------------------
# Mouth cavity from anchors
# ---------------------------------------------------------------------------

def mouth_cavity(V: np.ndarray, anchors: dict, options: dict | None = None,
                 F: np.ndarray | None = None) -> dict:
    """Derive the mouth cavity (center / size / inward axis) from anchors.

    anchors: 3D positions; "mouth_center" is required. Optional
    mouth_left/right/top/bottom (e.g. from MediaPipe 61/291/13/14) give the
    real opening shape; otherwise the cavity size is estimated from the eye
    distance / head proportions.

    The inward axis points from the mouth into the head along the mouth axis
    (the surface normal at the mouth, winding-corrected against the local
    mesh centroid; when F is not given, the direction to the centroid of the
    mesh region local to the mouth is used instead). Both variants are robust
    for full-body characters, where the whole-mesh centroid is pulled away by
    the torso/legs. Returns
        {"center": (3,), "size": (sx, sy, sz), "inward": (3,) unit,
         "outward": (3,), "right": (3,), "up": (3,)}.
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    center = np.asarray(anchors["mouth_center"], dtype=float)
    if center.shape != (3,):
        raise ValueError("anchors['mouth_center'] must be a 3D position")

    if "mouth_left" in anchors and "mouth_right" in anchors:
        ml = np.asarray(anchors["mouth_left"], float)
        mr = np.asarray(anchors["mouth_right"], float)
        horiz = mr - ml
        sx = np.linalg.norm(horiz)
        horiz = horiz / (sx + 1e-12)
    elif "eye_left" in anchors and "eye_right" in anchors:
        el = np.asarray(anchors["eye_left"], float)
        er = np.asarray(anchors["eye_right"], float)
        horiz = er - el
        horiz = horiz / (np.linalg.norm(horiz) + 1e-12)
        sx = opts["mouth_width_frac"] * np.linalg.norm(er - el)
    else:  # fall back to head width
        mn, mx = V.min(axis=0), V.max(axis=0)
        horiz = np.array([1.0, 0.0, 0.0])
        sx = 0.22 * (mx[0] - mn[0])

    if "mouth_top" in anchors and "mouth_bottom" in anchors:
        sy = float(np.linalg.norm(np.asarray(anchors["mouth_top"], float)
                                  - np.asarray(anchors["mouth_bottom"], float)))
    else:
        sy = opts["mouth_height_frac"] * sx
    sx = max(sx, 1e-3)
    sy = max(sy, 1e-3)
    sz = opts["depth_frac"] * sx

    # inward axis: from the mouth into the head. Base = direction to the
    # centroid of the mesh region local to the mouth (2x the mouth box),
    # robust for full-body characters where the whole-mesh centroid is pulled
    # away by the torso/legs. When F is available, refine with the surface
    # normal at the mouth (the true mouth axis), winding-corrected: the
    # outward normal must point AWAY from the local interior.
    Vf = np.asarray(V, dtype=float)
    r_loc = 2.0 * max(sx, sy)
    d2 = ((Vf - center) ** 2).sum(axis=1)
    sel = np.where(d2 <= r_loc ** 2)[0]
    interior = np.zeros(3)
    if len(sel) > 10:
        interior = Vf[sel].mean(axis=0) - center
    nrm = float(np.linalg.norm(interior))
    if nrm <= 1e-9:
        interior = Vf.mean(axis=0) - center
        nrm = float(np.linalg.norm(interior))
    interior = interior / nrm if nrm > 1e-9 else np.array([0.0, 0.0, -1.0])
    if F is not None:
        nv = vertex_normals(Vf, np.asarray(F, dtype=int))
        j = int(np.argmin(d2))
        outward = nv[j]
        nrm = float(np.linalg.norm(outward))
        if nrm > 1e-9:
            outward = outward / nrm
            if float(np.dot(outward, -interior)) < 0:  # flip inward-wound meshes
                outward = -outward
            inward = -outward
        else:
            inward = interior
    else:
        inward = interior
    outward = -inward

    # up: world +y projected onto the plane perpendicular to inward
    up = np.array([0.0, 1.0, 0.0])
    up = up - inward * float(np.dot(up, inward))
    nrm = np.linalg.norm(up)
    if nrm < 1e-6:
        up = np.array([0.0, 0.0, 1.0]) - inward * float(np.dot(np.array([0.0, 0.0, 1.0]), inward))
        nrm = np.linalg.norm(up)
    up = up / (nrm + 1e-12)
    # right: complete a right-handed orthonormal basis (right x up = outward),
    # flipped so it agrees with the anchor horizontal axis when available.
    right = np.cross(up, outward)
    nrm = np.linalg.norm(right)
    if nrm < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / nrm
        if np.dot(right, horiz) < 0:
            right = -right

    return {"center": center, "size": np.array([sx, sy, sz]),
            "inward": inward, "outward": outward,
            "right": right, "up": up}


def _place_parts(parts: dict, cavity: dict, options: dict | None = None) -> dict:
    """Rigid transform with non-uniform scale into the mouth cavity.

    Canonical axes are mapped onto the cavity axes: +x -> right, +y -> up,
    +z (teeth forward) -> outward. The teeth front (the actual canonical +z
    extent of the parts, which archetype transforms may stretch) is parked a
    small gap behind the lip plane (mouth_center), so nothing pokes through
    the closed lips even for off-axis mouth openings or fanged archetypes."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    gap = opts["back_gap_frac"] * cavity["size"][2]
    front_canon = max(float(V[:, 2].max()) for V, _F in parts.values())
    center = cavity["center"] + (front_canon * cavity["size"][2] + gap) * cavity["inward"]
    size = cavity["size"]
    # R rows are the world axes (right, up, outward): canonical coords rotated
    # into the world frame = v @ R (v @ R.T would give axis coordinates).
    R = np.stack([cavity["right"], cavity["up"], cavity["outward"]])
    out = {}
    for name, (V, F) in parts.items():
        out[name] = ((V * size) @ R + center, F)
    return out


# ---------------------------------------------------------------------------
# ARAP initial placement (control arch + RBF warp)
# ---------------------------------------------------------------------------

def _row_pts(V: np.ndarray, take_top: bool, n_cols: int,
             right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Sample n_cols representative points from a teeth row (crown half),
    binned along the mouth's right axis (u = V @ right)."""
    u = V @ right
    v = V @ up
    med = np.quantile(v, 0.5)
    band_idx = np.where(v >= med if take_top else v <= med)[0]
    if len(band_idx) < n_cols:
        band_idx = np.arange(len(V))
    ub = u[band_idx]
    xs = np.quantile(ub, np.linspace(0.0, 1.0, n_cols))
    band_mean = V[band_idx].mean(axis=0)
    u_mean = float(ub.mean())
    pts = []
    for i in range(n_cols):
        if i == 0:
            sel = ub <= xs[1]
        elif i == n_cols - 1:
            sel = ub > xs[-2]
        else:
            sel = (ub > xs[i - 1]) & (ub <= xs[i + 1])
        idx = band_idx[np.where(sel)[0]]
        if len(idx):
            pts.append(V[idx].mean(axis=0))
        else:
            pts.append(band_mean + (xs[i] - u_mean) * right)
    return np.asarray(pts, dtype=float)


def _control_arch(parts: dict, cavity: dict, n_cols: int = 5
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample the dental arch: top row from the upper teeth, bottom row from
    the lower teeth, n_cols columns binned by the mouth's right axis. Returns
    (pts (2*n_cols, 3), control faces, {"corners": [lt, rt, lb, rb] indices})."""
    up = parts.get("teeth_upper")
    lo = parts.get("teeth_lower")
    if up is None or lo is None:
        raise ValueError("control arch needs teeth_upper + teeth_lower parts")
    right, up_axis = cavity["right"], cavity["up"]
    top = _row_pts(up[0], True, n_cols, right, up_axis)
    bot = _row_pts(lo[0], False, n_cols, right, up_axis)
    pts = np.vstack([top, bot])
    F: list[list[int]] = []
    for c in range(n_cols - 1):
        a, b = c, c + 1
        c1, c2 = n_cols + c, n_cols + c + 1
        F += [[a, b, c2], [a, c2, c1]]
    corners = [0, n_cols - 1, n_cols, 2 * n_cols - 1]  # lt, rt, lb, rb
    return pts, np.asarray(F, dtype=int), {"corners": corners}


def _gaussian_rbf(rest: np.ndarray, target: np.ndarray, query: np.ndarray,
                  eps: float | None = None, ridge: float = 1e-6) -> np.ndarray:
    """Ridge-regularized Gaussian RBF displacement warp.

    Solves (Phi + lambda I) w = target - rest and evaluates
    query + Phi(query, rest) @ w. No polynomial term: the warp interpolates
    (nearly) exactly at the control points and decays to zero away from them,
    so it can never extrapolate the arch correction onto distant vertices."""
    rest = np.asarray(rest, dtype=float)
    target = np.asarray(target, dtype=float)
    query = np.asarray(query, dtype=float)
    n = len(rest)
    if eps is None:
        d = cKDTree(rest).query(rest, k=2)[0][:, 1]
        eps = max(float(np.median(d)), 1e-6)
    eps2 = eps * eps

    def phi(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        return np.exp(-d2 / eps2)

    Phi = phi(rest, rest)
    reg = ridge * float(np.trace(Phi)) / n
    w = np.linalg.solve(Phi + reg * np.eye(n), target - rest)
    return query + phi(query, rest) @ w


def _arap_fit(parts: dict, cavity: dict, options: dict | None = None) -> dict:
    """ARAP initial placement of the dental arch (paper §3.6.2).

    The control-arch corners are pinned to the mouth-cavity corners and the
    arch is deformed with geometry.arap_deform; the resulting displacement is
    propagated to every part vertex via a Gaussian RBF warp. When the anchors
    provide no usable mouth corners the warp is the identity (rigid placement
    only).

    The ARAP runs in the mouth plane (span of cavity['right'] / cavity['up'])
    in aspect-normalized coordinates (unit square control mesh, well
    conditioned): the arch spread is adapted to THIS character's mouth box
    without dragging the teeth forward through the closed lips (depth is fixed
    by the rigid placement step)."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    n_cols = opts["arap_n_cols"]
    try:
        arch, archF, meta = _control_arch(parts, cavity, n_cols)
    except ValueError:
        return parts
    right, up = cavity["right"], cavity["up"]
    sx, sy = cavity["size"][0], cavity["size"][1]
    # plane coordinates of the arch
    U = arch @ right
    Vv = arch @ up
    u0, v0 = float(U.mean()), float(Vv.mean())
    w, h = float(U.max() - U.min()), float(Vv.max() - Vv.min())
    w = max(w, 1e-6)
    h = max(h, 1e-6)
    # aspect-normalized control mesh: spans [-0.5, 0.5]^2
    arch_n = np.column_stack([(U - u0) / w, (Vv - v0) / h, np.zeros(len(arch))])
    corner_targets_n = {
        0: np.array([(-sx / 2 - u0) / w, (+sy / 2 - v0) / h, 0.0]),
        n_cols - 1: np.array([(+sx / 2 - u0) / w, (+sy / 2 - v0) / h, 0.0]),
        n_cols: np.array([(-sx / 2 - u0) / w, (-sy / 2 - v0) / h, 0.0]),
        2 * n_cols - 1: np.array([(+sx / 2 - u0) / w, (-sy / 2 - v0) / h, 0.0]),
    }
    handles = np.array(sorted(corner_targets_n), dtype=int)
    targets_n = np.stack([corner_targets_n[h] for h in handles])
    arch_def_n = arap_deform(arch_n, archF, handles, targets_n,
                             iterations=opts["arap_iterations"])
    # normalized displacement -> world (in-plane only, depth preserved)
    dU = (arch_def_n[:, 0] - arch_n[:, 0]) * w
    dV = (arch_def_n[:, 1] - arch_n[:, 1]) * h
    arch_def = arch + dU[:, None] * right + dV[:, None] * up
    # propagate the displacement LOCALLY around the arch (smooth falloff), so
    # the crown region adapts to the mouth box while roots/molars stay fixed;
    # cap the per-component displacement as a safety net.
    max_disp = 0.20 * sx
    tree = cKDTree(arch)
    out = {}
    for name, (V, F) in parts.items():
        disp = _gaussian_rbf(arch, arch_def, V) - V
        d = tree.query(V, k=1)[0]
        r_in, r_out = 0.30 * sx, 0.55 * sx
        t = np.clip((r_out - d) / max(r_out - r_in, 1e-9), 0.0, 1.0)
        wfall = t * t * (3.0 - 2.0 * t)  # smoothstep
        disp = disp * wfall[:, None]
        disp = np.clip(disp, -max_disp, max_disp)
        out[name] = (V + disp, F)
    return out


# ---------------------------------------------------------------------------
# SDF penetration refinement (paper §3.6.2)
# ---------------------------------------------------------------------------

def _convex_hull_sdf(verts: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Signed distance of points to the convex hull of `verts` (scipy only).

    For a convex polyhedron defined by half-spaces n_i . x + d_i <= 0
    (ConvexHull.equations rows), the (conservative) signed distance is
    max_i (n_i . x + d_i): negative inside, positive outside, zero on the
    boundary. This is a lower bound on the true distance magnitude and is
    exact when the closest feature is a face interior — sufficient for the
    iterative penetration push of §3.6.2. No trimesh/rtree dependency."""
    hull = ConvexHull(np.asarray(verts, dtype=float))
    A, b = hull.equations[:, :3], hull.equations[:, 3]
    return np.asarray(points, dtype=float) @ A.T + b  # (m, nfaces)


def face_penetration_stats(parts: dict, face_V: np.ndarray, face_F: np.ndarray,
                           mouth_center: np.ndarray | None = None,
                           radius: float | None = None) -> dict:
    """How much of the face surface intersects the teeth volume.

    Converts the teeth convex hull to an SDF (scipy half-space form, see
    _convex_hull_sdf) and evaluates the face vertices (optionally restricted
    to a sphere around mouth_center). Returns
    {"face_verts_inside": int, "max_penetration": float (>0 when inside),
     "mean_penetration": float, "checked": int}."""
    TVs = [V for name, (V, _F) in parts.items() if "teeth" in name]
    if not TVs:
        return {"face_verts_inside": 0, "max_penetration": 0.0,
                "mean_penetration": 0.0, "checked": 0}
    verts = np.vstack(TVs)
    try:
        sd_all = _convex_hull_sdf(verts, np.asarray(face_V, dtype=float))
    except Exception:
        return {"face_verts_inside": 0, "max_penetration": 0.0,
                "mean_penetration": 0.0, "checked": 0}
    if mouth_center is not None:
        d2 = ((np.asarray(face_V) - np.asarray(mouth_center, float)) ** 2).sum(axis=1)
        if radius is None:
            radius = float(np.linalg.norm(np.asarray(mouth_center, float)
                                          - verts.mean(axis=0))) + 1e-3
        sel = np.where(d2 <= radius ** 2)[0]
    else:
        sel = np.arange(len(face_V))
    if len(sel) == 0:
        return {"face_verts_inside": 0, "max_penetration": 0.0,
                "mean_penetration": 0.0, "checked": 0}
    sd = sd_all[sel].max(axis=1)  # conservative convex SDF value
    inside = sd < 0.0
    return {
        "face_verts_inside": int(inside.sum()),
        "max_penetration": float(-sd[inside].min()) if inside.any() else 0.0,
        "mean_penetration": float(-sd[inside].mean()) if inside.any() else 0.0,
        "checked": int(len(sel)),
    }


def _hull_equations(parts: dict):
    """Convex-hull half-space equations (A (m,3), b (m,)) of the teeth, or
    (None, None) when unavailable."""
    TVs = [V for name, (V, _F) in parts.items() if "teeth" in name]
    if not TVs:
        return None, None
    verts = np.vstack(TVs)
    try:
        hull = ConvexHull(verts)
    except Exception:
        return None, None
    return hull.equations[:, :3], hull.equations[:, 3]


def _sdf_refine(parts: dict, face_V: np.ndarray, face_F: np.ndarray,
                cavity: dict, options: dict | None = None
                ) -> tuple[dict, dict]:
    """Iteratively push the mouth block deeper into the cavity until no face
    vertex intersects the teeth hull (non-destructive to the outer surface).

    This mirrors the paper's SDF step but pushes the teeth inward along the
    SDF gradient direction instead of the face outward, keeping the accepted
    outer mesh byte-identical; push_face_out_of_teeth() provides the paper's
    outward variant. The hull is computed once; after each rigid push the SDF
    values are shifted analytically (signed distance is translation-invariant),
    so no re-triangulation is needed. Returns (parts, stats)."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    margin = opts["sdf_margin"]
    radius = opts["sdf_radius_frac"] * cavity["size"][0]
    inward = cavity["inward"]
    center = cavity["center"]
    A, b = _hull_equations(parts)
    if A is None:
        return parts, {"iterations": 0, "total_push": 0.0,
                       "final": {"face_verts_inside": 0, "max_penetration": 0.0,
                                 "mean_penetration": 0.0, "checked": 0}}
    d2 = ((np.asarray(face_V) - center) ** 2).sum(axis=1)
    sel = np.where(d2 <= radius ** 2)[0]
    if len(sel) == 0:
        return parts, {"iterations": 0, "total_push": 0.0,
                       "final": {"face_verts_inside": 0, "max_penetration": 0.0,
                                 "mean_penetration": 0.0, "checked": 0}}
    pts = np.asarray(face_V)[sel]
    A_inward = A @ inward
    total_push = 0.0
    n_iter = 0
    for _ in range(opts["sdf_max_iter"]):
        n_iter += 1
        # sd of pts to the hull shifted by total_push * inward
        sd = pts @ A.T + b - total_push * A_inward
        sd_max = sd.max(axis=1)
        inside = sd_max < margin
        if not inside.any():
            break
        push = float(-sd_max[inside].min()) + margin
        total_push += push
        for name in parts:
            V, F = parts[name]
            parts[name] = (V + push * inward, F)
    final = face_penetration_stats(parts, face_V, face_F, center, radius)
    return parts, {"iterations": n_iter, "total_push": float(total_push),
                   "final": final}


def push_face_out_of_teeth(face_V: np.ndarray, face_F: np.ndarray, parts: dict,
                           margin: float = 1e-3, steps: int = 10
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Paper-exact SDF refinement: push face vertices that intersect the teeth
    volume outward along the SDF gradient (returns modified face verts + the
    moved-vertex mask). build_inner_mouth() does not use this by default — it
    keeps the outer surface untouched — but the variant is provided for
    fidelity to §3.6.2 and verification."""
    TVs = [V for name, (V, _F) in parts.items() if "teeth" in name]
    if not TVs:
        return np.asarray(face_V, float).copy(), np.zeros(len(face_V), dtype=bool)
    verts = np.vstack(TVs)
    hull = ConvexHull(verts)
    A, b = hull.equations[:, :3], hull.equations[:, 3]
    V = np.asarray(face_V, float).copy()
    moved = np.zeros(len(V), dtype=bool)
    for _ in range(steps):
        sd = V @ A.T + b
        sd_max = sd.max(axis=1)
        inside = sd_max < margin
        if not inside.any():
            break
        moved |= inside
        # gradient of the max-plane SDF = normal of the argmax face
        j = sd[inside].argmax(axis=1)
        grad = A[j]
        # outward = +grad (n . x + d > 0 means outside in the half-space form)
        nrm = np.linalg.norm(grad, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        V[inside] += grad / nrm * (margin - sd_max[inside, None])
    return V, moved


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_inner_mouth(V: np.ndarray, F: np.ndarray, anchors: dict,
                      archetype: str = "human", options: dict | None = None
                      ) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Synthesize and place the inner-mouth parts for a character mesh.

    V, F: outer (face/head) mesh — used for the cavity estimate, the inward
    axis and the SDF penetration refinement.
    anchors: dict of 3D anchor positions; "mouth_center" required, optional
    mouth_left/right/top/bottom, eye_left/eye_right.
    archetype: "human" | "canine" | "monster" | "flat" (paper Table 6).

    Returns [(mesh_V, mesh_F, name), ...] with name in
    ("teeth_upper", "teeth_lower", "gums_tongue") — "flat" yields [].
    """
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    ict = load_ict_facekit()
    if ict is not None:
        # drop tiny stray components from the gums-tongue group (ICT slivers)
        gtV, gtF = ict["gums_tongue"]
        labels = _components(gtV, gtF)
        keep = np.bincount(labels) >= 10
        keep_mask = keep[labels]
        remap = {int(i): j for j, i in enumerate(np.where(keep_mask)[0])}
        gtV = gtV[keep_mask]
        gtF = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]]
                        for a, b, c in gtF
                        if keep_mask[a] and keep_mask[b] and keep_mask[c]])
        # split teeth into upper / lower rows
        Vu, Fu, Vl, Fl = _split_teeth_upper_lower(*ict["teeth"])
        raw = {"teeth_upper": (Vu, Fu), "teeth_lower": (Vl, Fl),
               "gums_tongue": (gtV, gtF)}
    else:
        raw = procedural_parts()
        warnings.warn("ICT-FaceKit assets not found; using procedural "
                      "placeholder inner-mouth geometry")

    canonical, _frame = _canonical_parts(raw)
    canonical = apply_archetype(canonical, archetype, opts)
    if not canonical:
        return []  # flat archetype: minimal / absent oral structure

    cavity = mouth_cavity(V, anchors, opts, F)
    placed = _place_parts(canonical, cavity, opts)
    fitted = _arap_fit(placed, cavity, opts)
    fitted, _sdf_stats = _sdf_refine(fitted, V, F, cavity, opts)

    order = {"teeth_upper": 0, "teeth_lower": 1, "gums_tongue": 2}
    parts = [(fitted[name][0], fitted[name][1], name) for name in fitted]
    parts.sort(key=lambda p: order.get(p[2], 9))
    return parts


# ---------------------------------------------------------------------------
# Jaw-follow morph deltas for the inner parts
# ---------------------------------------------------------------------------

def compute_part_morphs(parts: list, morphs: dict,
                        mouth_center: np.ndarray | None = None,
                        face_V: np.ndarray | None = None,
                        options: dict | None = None) -> dict:
    """Per-part morph deltas so the inner mouth animates with the jaw.

    For every JAW_SHAPES entry present in `morphs` (outer-mesh deltas), the
    jaw-driven part of each inner part rigidly follows the mean outer-mesh
    delta below mouth_center: teeth_lower and the tongue follow fully, upper
    teeth stay fixed, gums_tongue follows in its lower half. Returns
    {part_name: {shape: (n,3)}} listing only non-zero shapes."""
    opts = {**DEFAULT_OPTIONS, **(options or {})}
    scale = opts["morph_scale"]
    if mouth_center is not None and face_V is not None:
        face_lower = np.asarray(face_V)[:, 1] < float(np.asarray(mouth_center)[1])
    else:
        n = len(next(iter(morphs.values())))
        face_lower = np.ones(n, dtype=bool)
    out: dict = {}
    for pV, _pF, name in parts:
        if "upper" in name:
            lower = np.zeros(len(pV), dtype=bool)      # upper teeth are skull-fixed
        elif "lower" in name:
            lower = np.ones(len(pV), dtype=bool)       # lower teeth follow the jaw
        else:
            lower = pV[:, 1] < float(pV[:, 1].mean())  # gums/tongue: lower half
        part_deltas: dict[str, np.ndarray] = {}
        for shape in JAW_SHAPES:
            if shape not in morphs:
                continue
            d = np.asarray(morphs[shape], dtype=float)
            if d.ndim != 2 or d.shape[1] != 3 or len(d) != len(face_lower):
                continue
            mean = d[face_lower].mean(axis=0)
            if np.abs(mean).max() <= 0:
                continue
            delta = np.zeros((len(pV), 3))
            delta[lower] = mean * scale
            if np.abs(delta).max() > 0:
                part_deltas[shape] = delta
        if part_deltas:
            out[name] = part_deltas
    return out


# ---------------------------------------------------------------------------
# GLB attachment (official demo layout: one mesh, several primitives)
# ---------------------------------------------------------------------------

def attach_to_glb(gltf, parts: list, part_morphs: dict | None = None):
    """Append inner-mouth parts as extra primitives of gltf.meshes[0].

    Mirrors the official FINAL_WORK_DEMO.glb layout: a single mesh node whose
    primitives share the same morph-weight list, so the inner mouth animates
    with the outer mesh. part_morphs (from compute_part_morphs) supplies
    jaw-follow deltas; parts without them get zero morph targets.

    The binary buffer of `gltf` is extended in place (call build_gltf first).
    Returns the same gltf object.
    """
    if pygltflib is None:
        raise ImportError("pygltflib is required for attach_to_glb")
    if not parts:
        return gltf
    if not gltf.meshes:
        raise ValueError("gltf has no mesh to attach parts to")
    mesh = gltf.meshes[0]
    blob = gltf.binary_blob()
    if not blob:
        raise ValueError("gltf has no binary blob - call build_gltf() first")

    # number of morph targets + their order (outer mesh conventions)
    n_targets = 0
    target_names: list[str] = []
    if mesh.weights:
        n_targets = len(mesh.weights)
    elif mesh.primitives and mesh.primitives[0].targets:
        n_targets = len(mesh.primitives[0].targets)
    if n_targets:
        if mesh.extras is not None and mesh.extras.get("targetNames"):
            target_names = list(mesh.extras["targetNames"])
        else:
            from .arkit52 import ARKIT_52
            target_names = list(ARKIT_52)
        if len(target_names) != n_targets:
            target_names = [f"morph{i}" for i in range(n_targets)]

    app = _BinaryAppender(gltf, blob)
    zero_by_count: dict[int, int] = {}
    for pV, pF, name in parts:
        pV = np.asarray(pV, dtype=np.float32)
        pF = np.asarray(pF, dtype=np.uint16 if len(pV) <= 65535 else np.uint32)
        nrm = vertex_normals(pV, pF).astype(np.float32)
        pos = app.add_accessor(pV, "VEC3", 5126, target=_BinaryAppender._ARRAY_BUFFER)
        nrm_acc = app.add_accessor(nrm, "VEC3", 5126, target=_BinaryAppender._ARRAY_BUFFER)
        idx = app.add_accessor(pF.ravel(), "SCALAR",
                               5123 if pF.dtype == np.uint16 else 5125,
                               with_minmax=False, target=_BinaryAppender._ELEMENT_ARRAY_BUFFER)
        attrs = pygltflib.Attributes(POSITION=pos, NORMAL=nrm_acc)
        # A skinned mesh must carry JOINTS_0/WEIGHTS_0 on every primitive:
        # three.js GLTFLoader turns each primitive of a skinned mesh into a
        # SkinnedMesh and calls normalizeSkinWeights(), which throws
        # "Cannot read properties of undefined (reading 'count')" when
        # geometry.attributes.skinWeight is missing.  Rigid-bind the inner
        # mouth to the head joint (weight 1.0) so it follows the head.
        if gltf.skins:
            skin = gltf.skins[0]
            head_idx = 0
            for _i, _jn in enumerate(skin.joints):
                if gltf.nodes[_jn].name == "mixamorig:Head":
                    head_idx = _i
                    break
            n = len(pV)
            jidx = np.zeros((n, 4), dtype=np.uint16)
            jidx[:, 0] = head_idx
            jw = np.zeros((n, 4), dtype=np.float32)
            jw[:, 0] = 1.0
            attrs.JOINTS_0 = app.add_accessor(jidx, "VEC4", 5123, with_minmax=False)
            attrs.WEIGHTS_0 = app.add_accessor(jw, "VEC4", 5126)
        targets = []
        if n_targets:
            pm = (part_morphs or {}).get(name, {})
            for shape in target_names:
                delta = pm.get(shape)
                if delta is None:
                    if len(pV) not in zero_by_count:
                        zero_by_count[len(pV)] = app.add_accessor(
                            np.zeros((len(pV), 3), dtype=np.float32), "VEC3", 5126,
                            target=_BinaryAppender._ARRAY_BUFFER)
                    targets.append(pygltflib.Attributes(POSITION=zero_by_count[len(pV)]))
                else:
                    targets.append(pygltflib.Attributes(POSITION=app.add_accessor(
                        np.asarray(delta, dtype=np.float32), "VEC3", 5126,
                        target=_BinaryAppender._ARRAY_BUFFER)))
        prim = pygltflib.Primitive(attributes=attrs, indices=idx)
        if targets:
            prim.targets = targets
        mesh.primitives.append(prim)
    app.finish()
    return gltf


class _BinaryAppender:
    """Appends typed arrays to the end of an existing GLB binary blob."""

    _ARRAY_BUFFER = 34962
    _ELEMENT_ARRAY_BUFFER = 34963

    def __init__(self, gltf, blob: bytes):
        self.gltf = gltf
        self.blob = bytearray(blob)
        self.offset = len(self.blob)

    def _pad(self, n: int) -> None:
        rem = self.offset % n
        if rem:
            self.blob += b"\x00" * (n - rem)
            self.offset += n - rem

    def add_view(self, arr: np.ndarray, target: int) -> int:
        self._pad(4)
        view = pygltflib.BufferView(
            buffer=0, byteOffset=self.offset, byteLength=arr.nbytes, target=target,
        )
        self.gltf.bufferViews.append(view)
        self.blob += arr.tobytes()
        self.offset += arr.nbytes
        return len(self.gltf.bufferViews) - 1

    def add_accessor(self, arr: np.ndarray, type_: str, component_type: int,
                     with_minmax: bool = True, target: int = _ARRAY_BUFFER) -> int:
        view = self.add_view(arr, target)
        acc = pygltflib.Accessor(
            bufferView=view, byteOffset=0, componentType=component_type,
            count=len(arr), type=type_,
        )
        if with_minmax and component_type == 5126:
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            acc.min = np.asarray(lo).tolist()
            acc.max = np.asarray(hi).tolist()
        self.gltf.accessors.append(acc)
        return len(self.gltf.accessors) - 1

    def finish(self) -> None:
        self._pad(4)
        self.gltf.buffers[0].byteLength = self.offset
        self.gltf.set_binary_blob(bytes(self.blob))
