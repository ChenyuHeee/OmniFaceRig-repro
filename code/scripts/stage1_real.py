"""Real-data pipeline: official T-pose GLB (+ optional 2D image) -> rigged GLB.

Runs on the A100 server:
  1. load official glb (static mesh, ~1M verts)
  2. frontal 2D view: use the official 2D image if given, else render the mesh
     (pyrender EGL) — MediaPipe Face Landmarker (478 pts) on that view
  3. back-project landmarks to 3D keypoints (orthographic + surface snap)
  4. face region = mesh vertices near the keypoint cloud (head part)
  5. ARKit 52 morph targets: region-based shape generators anchored at the
     detected eye/mouth centers + delta-mush smoothing
  6. Mixamo-style skeleton (53 joints, exact demo naming) placed by character
     proportions + proximity skin weights
  7. lip-sync animation (Mandarin viseme track -> ARKit weights)
  8. export rigged glb (52 morphs + skin + animation)

Usage:
  python stage1_real.py --glb ai3d_01.glb [--img image.png] [--out out.glb]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from omnifacerig_repro.arkit52 import ARKIT_52, arkit_vector
from omnifacerig_repro.geometry import delta_mush, smoothing_matrix, vertex_normals
from omnifacerig_repro.glb_export import build_gltf, write_glb, load_glb
from omnifacerig_repro import lip_sync
from omnifacerig_repro.pipeline import synthesize_canonical_shapes  # reuse generators


# ---------------------------------------------------------------------------
# glb IO (pygltflib)
# ---------------------------------------------------------------------------

def load_mesh(glb_path):
    """Return (V, F, texinfo) of the first primitive (merged across meshes)."""
    gltf = load_glb(glb_path)
    Vs, Fs = [], []
    offset = 0
    blob = gltf.binary_blob()
    NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
    DT = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}

    def read_acc(acc_idx):
        acc = gltf.accessors[acc_idx]
        bv = gltf.bufferViews[acc.bufferView]
        n = NCOMP[acc.type]
        arr = np.frombuffer(blob, dtype=DT[acc.componentType],
                            count=acc.count * n, offset=bv.byteOffset)
        return arr.reshape(-1, n).copy()

    for m in gltf.meshes:
        for p in m.primitives:
            pos = read_acc(p.attributes.POSITION)
            idx = read_acc(p.indices).ravel()
            Vs.append(pos)
            Fs.append(offset + idx.reshape(-1, 3))
            offset += len(pos)
    return np.vstack(Vs), np.vstack(Fs), gltf


# ---------------------------------------------------------------------------
# landmarks (MediaPipe) -> 3D keypoints
# ---------------------------------------------------------------------------

def detect_landmarks(img_bgr, model_path=None):
    """MediaPipe FaceLandmarker (Tasks API, mediapipe>=1.0) -> 478 landmarks."""
    if model_path is None:
        # default: next to the script, or repo root of the workdir
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, "face_landmarker.task"),
                     os.path.join(here, "..", "..", "face_landmarker.task")):
            if os.path.exists(cand):
                model_path = cand
                break
    assert model_path and os.path.exists(model_path), f"model not found: {model_path}"
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceLandmarkerOptions(
        base_options=base, num_faces=1, output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    with vision.FaceLandmarker.create_from_options(opts) as detector:
        rgb = img_bgr[:, :, ::-1]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        return np.array([[p.x, p.y, p.z] for p in lm])  # 478 x 3, normalized


def keypoints_from_landmarks(V, F, lms, img_h, img_w):
    """Map 2D landmarks (normalized) to 3D keypoints on the mesh surface.

    Orthographic frontal assumption (T-pose front view): the mesh is
    projected onto the image plane; for each landmark we take the mesh
    vertex whose projected position is closest (front surface wins because
    we only search vertices with z >= z_centroid of the head region).
    """
    mn = V.min(axis=0)
    mx = V.max(axis=0)
    scale_x = mx[0] - mn[0]
    scale_y = mx[1] - mn[1]
    # landmark pixel -> mesh x/y (image: x right, y down; mesh: x right, y up)
    px = (lms[:, 0] * img_w - img_w / 2.0) / (img_w / 2.0) * (scale_x / 2.0) + (mn[0] + mx[0]) / 2
    py = (1.0 - lms[:, 1]) * scale_y + mn[1]
    # only the front (z > zmin + 0.35*range) surface can be visible
    zmin, zmax = mn[2], mx[2]
    front = V[:, 2] > zmin + 0.30 * (zmax - zmin)
    Vf = V[front]
    idx_map = np.where(front)[0]
    from scipy.spatial import cKDTree
    tree = cKDTree(Vf[:, :2])
    d, j = tree.query(np.column_stack([px, py]))
    return idx_map[j], np.column_stack([px, py, Vf[j, 2]])


# ---------------------------------------------------------------------------
# face region + canonical shape generators anchored at real landmarks
# ---------------------------------------------------------------------------

def geometric_anchors(V):
    """Face anchors from mesh geometry (humanoid T-pose, frontal +z).

    head cluster = top 18% of body height, central x band; face features by
    classical proportions (eyes ~50%, mouth ~25% up from the chin); anchor
    z from the front (+z) surface nearest the (x,y) position.
    """
    mn, mx = V.min(axis=0), V.max(axis=0)
    H, W = mx[1] - mn[1], mx[0] - mn[0]
    y_top = mx[1]
    head_mask = (V[:, 1] > y_top - 0.18 * H) & (np.abs(V[:, 0]) < 0.20 * W)
    idx = np.where(head_mask)[0]
    if len(idx) < 100:
        raise RuntimeError("head cluster not found")
    Vh = V[idx]
    chin = Vh[:, 1].min()
    hh = y_top - chin
    hw = Vh[:, 0].max() - Vh[:, 0].min()
    zc = Vh[:, 2].mean()
    front = Vh[:, 2] > zc  # +z assumed frontal

    def anchor(x, y):
        d2 = (Vh[:, 0] - x) ** 2 + (Vh[:, 1] - y) ** 2
        d2[~front] = np.inf
        j = int(np.argmin(d2))
        return np.array([x, y, Vh[j, 2]])

    cx = Vh[:, 0].mean()
    eye_y = chin + 0.52 * hh
    mouth_y = chin + 0.25 * hh
    eye_dx = 0.30 * hw
    return {
        "eye_left": anchor(cx - eye_dx, eye_y),
        "eye_right": anchor(cx + eye_dx, eye_y),
        "mouth_center": anchor(cx, mouth_y),
    }, idx


def face_region(V, F, keypoints, radius_frac=0.28):
    """Vertices within radius_frac * head-height of the face keypoint cloud."""
    center = keypoints.mean(axis=0)
    head_h = (V[:, 1].max() - V[:, 1].min())
    r = radius_frac * head_h
    d = np.linalg.norm(V - center, axis=1)
    return np.where(d < r)[0]


def anchor_from_keypoints(keypoints, lms, names):
    """Return dict of 3D anchor positions for the shape generators."""
    # lms: normalized 2D; keypoints: corresponding 3D
    out = {}
    for name, idx in names.items():
        if idx < len(keypoints):
            out[name] = keypoints[idx]
    return out


def build_real_morphs(V, F, region_idx, anchors, ry_scale=1.0):
    """52 ARKit morph deltas via region generators anchored at real landmarks.

    Reuses the semantic generators from pipeline.synthesize_canonical_shapes
    but centered at the detected eye/mouth anchors of THIS character.
    """
    n = len(V)
    shapes = {name: np.zeros((n, 3)) for name in ARKIT_52}
    if not anchors:
        return shapes
    eye_l = anchors.get("eye_left", None)
    eye_r = anchors.get("eye_right", None)
    mouth_c = anchors.get("mouth_center", None)
    # fall back to geometry if landmark anchors missing
    reg = region_idx

    def w_around(center, sigma):
        w = np.zeros(n)
        if center is None:
            return w
        d = np.linalg.norm(V - center, axis=1)
        w = np.exp(-(d / sigma) ** 2)
        w[~np.isin(np.arange(n), reg)] = 0.0
        return w

    sig_eye = 0.06 * (V[:, 1].max() - V[:, 1].min())
    sig_mouth = 0.08 * (V[:, 1].max() - V[:, 1].min())

    # jaw open: lower face (below mouth) moves down
    if mouth_c is not None:
        lower = V[:, 1] < mouth_c[1]
        lower &= np.isin(np.arange(n), reg)
        jaw = np.zeros((n, 3))
        f = (mouth_c[1] - V[:, 1]) / (V[:, 1].max() - V[:, 1].min())
        f = np.clip(f, 0, 0.5)
        jaw[lower] = np.column_stack([np.zeros(lower.sum()), -0.10 * f[lower],
                                      0.05 * f[lower]])
        shapes["jawOpen"] = jaw
        shapes["mouthClose"] = -jaw * 0.4
        shapes["jawLeft"] = jaw * np.array([0.6, 0, 0])
        shapes["jawRight"] = -jaw * np.array([0.6, 0, 0])

    for side, c, s in (("Left", eye_l, -1.0), ("Right", eye_r, 1.0)):
        if c is None:
            continue
        w = w_around(c, sig_eye)
        blink = np.zeros((n, 3))
        blink[:, 1] = w * (c[1] - V[:, 1]) * 0.9
        shapes[f"eyeBlink{side}"] = blink
        shapes[f"eyeWide{side}"] = -blink * 0.5
        shapes[f"eyeLookUp{side}"] = w[:, None] * np.array([0, 0, 0.05])
        shapes[f"eyeLookDown{side}"] = w[:, None] * np.array([0, 0, -0.05])

    if mouth_c is not None:
        for side, s in (("Left", -1.0), ("Right", 1.0)):
            c = mouth_c + np.array([s * sig_mouth * 1.6, 0, 0])
            w = w_around(c, sig_mouth)
            sm = np.zeros((n, 3))
            sm[:, 0] = w * s * 0.05
            sm[:, 1] = w * 0.02
            shapes[f"mouthSmile{side}"] = sm
            shapes[f"mouthFrown{side}"] = -sm * np.array([1, 0.6, 0])
            shapes[f"mouthStretch{side}"] = sm * np.array([1.6, 0.3, 0])
            shapes[f"mouthUpperUp{side}"] = w[:, None] * np.array([0, 0.05, 0])
            shapes[f"mouthLowerDown{side}"] = w[:, None] * np.array([0, -0.05, 0])
        shapes["mouthPucker"] = (shapes["mouthSmileLeft"] + shapes["mouthSmileRight"]) * -0.5
        shapes["mouthFunnel"] = shapes["mouthPucker"] * np.array([0.8, 1.4, 0.4])

    # smooth every shape on the face region (delta mush, full-mesh topology)
    for k in list(shapes.keys()):
        d = shapes[k]
        if np.abs(d).max() == 0:
            continue
        smoothed = delta_mush(V, F, V + d, iterations=2, alpha=0.6)
        shapes[k] = smoothed - V
    return shapes


# ---------------------------------------------------------------------------
# Mixamo-style skeleton (53 joints, demo naming) by proportions
# ---------------------------------------------------------------------------

MIXAMO_JOINTS = [
    ("mixamorig:Hips", None), ("mixamorig:Spine", "mixamorig:Hips"),
    ("mixamorig:Spine1", "mixamorig:Spine"), ("mixamorig:Spine2", "mixamorig:Spine1"),
    ("mixamorig:Neck", "mixamorig:Spine2"), ("mixamorig:Head", "mixamorig:Neck"),
    ("mixamorig:LeftShoulder", "mixamorig:Spine2"),
    ("mixamorig:LeftArm", "mixamorig:LeftShoulder"),
    ("mixamorig:LeftForeArm", "mixamorig:LeftArm"),
    ("mixamorig:LeftHand", "mixamorig:LeftForeArm"),
    ("mixamorig:LeftHandThumb1", "mixamorig:LeftHand"),
    ("mixamorig:LeftHandThumb2", "mixamorig:LeftHandThumb1"),
    ("mixamorig:LeftHandThumb3", "mixamorig:LeftHandThumb2"),
    ("mixamorig:LeftHandIndex1", "mixamorig:LeftHand"),
    ("mixamorig:LeftHandIndex2", "mixamorig:LeftHandIndex1"),
    ("mixamorig:LeftHandIndex3", "mixamorig:LeftHandIndex2"),
    ("mixamorig:LeftHandMiddle1", "mixamorig:LeftHand"),
    ("mixamorig:LeftHandMiddle2", "mixamorig:LeftHandMiddle1"),
    ("mixamorig:LeftHandMiddle3", "mixamorig:LeftHandMiddle2"),
    ("mixamorig:LeftHandRing1", "mixamorig:LeftHand"),
    ("mixamorig:LeftHandRing2", "mixamorig:LeftHandRing1"),
    ("mixamorig:LeftHandRing3", "mixamorig:LeftHandRing2"),
    ("mixamorig:LeftHandPinky1", "mixamorig:LeftHand"),
    ("mixamorig:LeftHandPinky2", "mixamorig:LeftHandPinky1"),
    ("mixamorig:LeftHandPinky3", "mixamorig:LeftHandPinky2"),
    ("mixamorig:RightShoulder", "mixamorig:Spine2"),
    ("mixamorig:RightArm", "mixamorig:RightShoulder"),
    ("mixamorig:RightForeArm", "mixamorig:RightArm"),
    ("mixamorig:RightHand", "mixamorig:RightForeArm"),
    ("mixamorig:RightHandThumb1", "mixamorig:RightHand"),
    ("mixamorig:RightHandThumb2", "mixamorig:RightHandThumb1"),
    ("mixamorig:RightHandThumb3", "mixamorig:RightHandThumb2"),
    ("mixamorig:RightHandIndex1", "mixamorig:RightHand"),
    ("mixamorig:RightHandIndex2", "mixamorig:RightHandIndex1"),
    ("mixamorig:RightHandIndex3", "mixamorig:RightHandIndex2"),
    ("mixamorig:RightHandMiddle1", "mixamorig:RightHand"),
    ("mixamorig:RightHandMiddle2", "mixamorig:RightHandMiddle1"),
    ("mixamorig:RightHandMiddle3", "mixamorig:RightHandMiddle2"),
    ("mixamorig:RightHandRing1", "mixamorig:RightHand"),
    ("mixamorig:RightHandRing2", "mixamorig:RightHandRing1"),
    ("mixamorig:RightHandRing3", "mixamorig:RightHandRing2"),
    ("mixamorig:RightHandPinky1", "mixamorig:RightHand"),
    ("mixamorig:RightHandPinky2", "mixamorig:RightHandPinky1"),
    ("mixamorig:RightHandPinky3", "mixamorig:RightHandPinky2"),
    ("mixamorig:LeftUpLeg", "mixamorig:Hips"),
    ("mixamorig:LeftLeg", "mixamorig:LeftUpLeg"),
    ("mixamorig:LeftFoot", "mixamorig:LeftLeg"),
    ("mixamorig:LeftToeBase", "mixamorig:LeftFoot"),
    ("mixamorig:RightUpLeg", "mixamorig:Hips"),
    ("mixamorig:RightLeg", "mixamorig:RightUpLeg"),
    ("mixamorig:RightFoot", "mixamorig:RightLeg"),
    ("mixamorig:RightToeBase", "mixamorig:RightFoot"),
    ("neutral_bone", None),
]


def build_skeleton(V):
    """Place the 53 Mixamo joints by character proportions (T-pose)."""
    mn, mx = V.min(axis=0), V.max(axis=0)
    H = mx[1] - mn[1]
    W = mx[0] - mn[0]
    cx, cz = (mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2
    y = lambda frac: mn[1] + frac * H
    x = lambda frac: cx + frac * W
    pos = {
        "mixamorig:Hips": (cx, y(0.52), cz),
        "mixamorig:Spine": (cx, y(0.56), cz),
        "mixamorig:Spine1": (cx, y(0.62), cz),
        "mixamorig:Spine2": (cx, y(0.68), cz),
        "mixamorig:Neck": (cx, y(0.80), cz),
        "mixamorig:Head": (cx, y(0.88), cz),
        "mixamorig:LeftShoulder": (x(-0.08), y(0.70), cz),
        "mixamorig:LeftArm": (x(-0.33), y(0.70), cz),
        "mixamorig:LeftForeArm": (x(-0.44), y(0.70), cz),
        "mixamorig:LeftHand": (x(-0.48), y(0.70), cz),
        "mixamorig:RightShoulder": (x(0.08), y(0.70), cz),
        "mixamorig:RightArm": (x(0.33), y(0.70), cz),
        "mixamorig:RightForeArm": (x(0.44), y(0.70), cz),
        "mixamorig:RightHand": (x(0.48), y(0.70), cz),
        "mixamorig:LeftUpLeg": (x(-0.09), y(0.52), cz),
        "mixamorig:LeftLeg": (x(-0.09), y(0.28), cz),
        "mixamorig:LeftFoot": (x(-0.09), y(0.06), cz),
        "mixamorig:LeftToeBase": (x(-0.09), y(0.02), cz),
        "mixamorig:RightUpLeg": (x(0.09), y(0.52), cz),
        "mixamorig:RightLeg": (x(0.09), y(0.28), cz),
        "mixamorig:RightFoot": (x(0.09), y(0.06), cz),
        "mixamorig:RightToeBase": (x(0.09), y(0.02), cz),
        "neutral_bone": (cx, y(0.52), cz),
    }
    # finger joints: tiny offsets from the hand
    for side, s in (("Left", -1), ("Right", 1)):
        hx = x(0.48 * s)
        for chain, offs in (("Thumb", [(0, 0.005), (0, 0.012), (0, 0.019)]),
                            ("Index", [(0.012 * s, 0), (0.02 * s, 0), (0.028 * s, 0)]),
                            ("Middle", [(0.022 * s, 0), (0.036 * s, 0), (0.05 * s, 0)]),
                            ("Ring", [(0.032 * s, 0), (0.044 * s, 0), (0.056 * s, 0)]),
                            ("Pinky", [(0.042 * s, 0), (0.052 * s, 0), (0.062 * s, 0)])):
            for k, (dx, dy) in enumerate(offs):
                pos[f"mixamorig:{side}{chain}{k+1}"] = (hx + dx, y(0.70) + dy, cz)
    transforms = []
    names = []
    for name, parent in MIXAMO_JOINTS:
        names.append(name)
        p = pos.get(name, (cx, y(0.5), cz))
        T = np.eye(4)
        T[:3, 3] = p
        transforms.append(T)
    return {"joint_names": names, "joint_transforms": transforms}


def skin_by_proximity(V, skeleton, k=2, sigma_frac=0.10):
    """Gaussian falloff skin weights over joint positions."""
    names = skeleton["joint_names"]
    centers = np.array([T[:3, 3] for T in skeleton["joint_transforms"]])
    H = V[:, 1].max() - V[:, 1].min()
    sigma = sigma_frac * H
    d = np.linalg.norm(V[:, None, :] - centers[None, :, :], axis=2)  # (n, nj)
    W = np.exp(-(d / sigma) ** 2)
    # zero weight for joints far above/below (spine handles torso)
    jidx = np.zeros((len(V), k), dtype=np.uint16)
    jw = np.zeros((len(V), k), dtype=np.float32)
    for i in range(len(V)):
        top = np.argsort(W[i])[::-1][:k]
        s = W[i, top].sum()
        jidx[i] = top
        jw[i] = W[i, top] / s if s > 0 else 0
    return jidx, jw


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(glb_path, img_path=None, out_path="outputs/rigged.glb", text="你好世界"):
    V, F, gltf = load_mesh(glb_path)
    print(f"mesh: {len(V)} verts, {len(F)} faces; bbox H={V[:,1].max()-V[:,1].min():.3f}")

    if img_path is not None:
        import cv2
        img = cv2.imread(img_path)
        assert img is not None, f"cannot read image {img_path}"
        img_h, img_w = img.shape[:2]
        lms = detect_landmarks(img)
        assert lms is not None, "no face detected"
        kp_idx, kp_pos = keypoints_from_landmarks(V, F, lms, img_h, img_w)
        print(f"landmarks: {len(lms)} detected -> {len(kp_pos)} 3D keypoints")
        # anchor names (MediaPipe indices: left eye 33/133, right eye 362/263,
        # mouth 61/291/0)
        anchors = {
            "eye_left": kp_pos[33], "eye_right": kp_pos[263],
            "mouth_center": kp_pos[13],
        }
        region_idx = face_region(V, F, kp_pos)
    else:
        anchors, head_idx = geometric_anchors(V)
        region_idx = head_idx
        print("geometric anchors:", {k: [round(x, 4) for x in v] for k, v in anchors.items()})

    morphs = build_real_morphs(V, F, region_idx, anchors)
    print(f"morph targets: {sum(1 for k,v in morphs.items() if np.abs(v).max()>0)}/{len(morphs)} non-zero")

    skeleton = build_skeleton(V)
    jidx, jw = skin_by_proximity(V, skeleton)
    skeleton["joint_indices"] = jidx
    skeleton["joint_weights"] = jw

    visemes = lip_sync.zh_text_to_visemes(text)
    dt = 0.18
    times = np.arange(len(visemes) + 1, dtype=float) * dt
    weights = np.zeros((len(visemes) + 1, 52))
    for i, v in enumerate(visemes):
        weights[i] = arkit_vector(lip_sync.viseme_to_arkit(v))
    anim = {"times": times, "weights": weights}

    normals = vertex_normals(V, F)
    gltf_out = build_gltf(V, F, morphs, normals=normals, skin=skeleton,
                          animation=anim, name=os.path.basename(glb_path))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_glb(out_path, gltf_out)
    print("written:", out_path)
    return {"out": out_path, "verts": len(V), "faces": len(F),
            "morphs": len(morphs), "joints": len(skeleton["joint_names"]),
            "visemes": visemes}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--img", default=None)
    ap.add_argument("--out", default="outputs/rigged.glb")
    ap.add_argument("--text", default="你好世界")
    args = ap.parse_args()
    stats = run(args.glb, args.img, args.out, args.text)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
