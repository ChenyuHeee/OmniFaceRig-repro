#!/usr/bin/env python3
"""TRELLIS image-to-mesh front-end (issue #3): one image -> textured mesh GLB.

Pipeline position
-----------------
交付物 2 的链路是:

    image ──> [trellis_front.py] ──> static textured mesh .glb
          ──> [stage1_real.py --glb <glb>] ──> rigged .glb
              (52 ARKit morphs + Mixamo 53 joints + lip-sync animation)

stage1_real.py 的输入契约 (its ``--glb`` argument) is any conformant
glTF 2.0 .glb: ``stage1_real.load_mesh`` reads the first primitive's
POSITION + indices via pygltflib and ignores everything else, so a
TRELLIS PBR-textured GLB and the ``--mock`` placeholder GLB both satisfy
it.  In other words: ``out_glb_path`` returned by ``image_to_mesh`` is
exactly what you pass to ``stage1_real.py --glb``.

API
---
    image_to_mesh(image_path, out_glb_path, device="cuda") -> out_glb_path

    * real mode  (default): runs microsoft/TRELLIS-image-large
      (TrellisImageTo3DPipeline + postprocessing_utils.to_glb; needs an
      NVIDIA GPU >= 16 GB VRAM and the weights, see deploy_trellis.sh).
      Raises TrellisFrontError with an actionable message when the
      weights / runtime are missing.
    * mock mode  (--mock): builds a placeholder "ellipsoid head" GLB with
      numpy/trimesh/PIL only (no weights, no GPU), so the stage1_real.py
      input contract is testable end-to-end on any machine.

CLI
---
    python scripts/trellis_front.py --image in.png --out out.glb
    python scripts/trellis_front.py --image in.png --out out.glb --mock

Environment
-----------
    TRELLIS_MODEL_PATH   local dir of TRELLIS-image-large weights
                         (default: ~/.cache/trellis/TRELLIS-image-large;
                         set to "microsoft/TRELLIS-image-large" to let
                         huggingface_hub download instead)
    TRELLIS_DINOV2_PTH   local dinov2_vitl14_reg_pretrain.pth (optional;
                         the image encoder. Default location is the torch
                         hub cache, which deploy_trellis.sh pre-seeds)
    TORCH_HOME           torch hub cache dir (default ~/.cache/torch)

Network notes (measured 2026-08-25): huggingface.co direct is unreachable
from the dev machine; hf-mirror.com serves the weights at ~8-11 MB/s.
deploy_trellis.sh automates the whole fetch with curl -C - resume.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# defaults / paths
# ---------------------------------------------------------------------------

TRELLIS_HF_REPO = "microsoft/TRELLIS-image-large"
DEFAULT_WEIGHTS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "trellis",
                                   "TRELLIS-image-large")

# pipeline.json references these 6 checkpoints at inference time
# (the 2 *_enc_* VAE encoders are only needed for training).
# NOTE: names are the *file basenames* under ckpts/ (pipeline.json model
# values), e.g. slat_dec_gs_* — not the pipeline model keys slat_decoder_gs_*.
INFERENCE_CKPTS = [
    "ss_dec_conv3d_16l8_fp16",
    "ss_flow_img_dit_L_16l8_fp16",
    "slat_dec_gs_swin8_B_64l8gs32_fp16",
    "slat_dec_rf_swin8_B_64l8r16_fp16",
    "slat_dec_mesh_swin8_B_64l8m256c_fp16",
    "slat_flow_img_dit_L_64l8p2_fp16",
]


class TrellisFrontError(RuntimeError):
    """Raised when the real TRELLIS path cannot run (weights/runtime)."""


# ---------------------------------------------------------------------------
# mock front-end: placeholder "ellipsoid head" GLB (no weights, no GPU)
# ---------------------------------------------------------------------------

def _ellipsoid_mesh(rx: float, ry: float, rz: float, n_lat: int = 24,
                    n_lon: int = 48):
    """Non-degenerate ellipsoid mesh (pole fans + quad rings), facing +z.

    Same construction as omnifacerig_repro.pipeline.ellipsoid_mesh but kept
    local so this script is self-contained for deployment.
    Returns (V, F) with all triangles having non-zero area.
    """
    th = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    V = [np.array([0.0, 0.0, rz]), np.array([0.0, 0.0, -rz])]
    for t in th[1:-1]:
        ct, st = np.cos(t), np.sin(t)
        for ph in lon:
            V.append((rx * st * np.cos(ph), ry * st * np.sin(ph), rz * ct))
    V = np.asarray(V, dtype=float)
    ring0, ring1 = 2, 2 + (n_lat - 3) * n_lon
    F: List[List[int]] = []
    for j in range(n_lon):
        F.append([0, ring0 + j, ring0 + (j + 1) % n_lon])
    for i in range(n_lat - 3):
        a0 = ring0 + i * n_lon
        a1 = a0 + n_lon
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            F.append([a0 + j, a1 + j, a1 + j2])
            F.append([a0 + j, a1 + j2, a0 + j2])
    for j in range(n_lon):
        F.append([ring1 + j, 1, ring1 + (j + 1) % n_lon])
    return V, np.asarray(F, dtype=np.int64)


def _spherical_uv(V: np.ndarray) -> np.ndarray:
    """Unit-sphere UV parametrisation (u wraps around +z, v = latitude)."""
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    u = 0.5 + np.arctan2(z, x) / (2.0 * np.pi)
    v = 0.5 - np.arcsin(np.clip(y / np.maximum(r, 1e-12), -1.0, 1.0)) / np.pi
    return np.column_stack([u, v]).astype(np.float32)


def _placeholder_texture(img_path: Optional[str]) -> "PIL.Image.Image":
    """Small procedural texture: skin tone + stylised eyes/mouth.

    Sized to the input image aspect so the placeholder vaguely resembles
    the prompt (or a square default when the image is unreadable).
    """
    from PIL import Image, ImageDraw

    if img_path and os.path.exists(img_path):
        try:
            w0, h0 = Image.open(img_path).size
        except Exception:
            w0, h0 = 512, 512
    else:
        w0, h0 = 512, 512
    scale = 256.0 / max(w0, h0)
    W = max(64, int(w0 * scale))
    H = max(64, int(h0 * scale))
    img = Image.new("RGB", (W, H), (236, 214, 196))          # skin tone
    d = ImageDraw.Draw(img)
    eye_y = int(H * 0.42)
    eye_dx = int(W * 0.18)
    r = max(2, int(min(W, H) * 0.035))
    d.ellipse([W // 2 - eye_dx - r, eye_y - r, W // 2 - eye_dx + r, eye_y + r],
              fill=(60, 42, 36))
    d.ellipse([W // 2 + eye_dx - r, eye_y - r, W // 2 + eye_dx + r, eye_y + r],
              fill=(60, 42, 36))
    mouth_y = int(H * 0.62)
    mw = int(W * 0.16)
    d.arc([W // 2 - mw, mouth_y - int(H * 0.04), W // 2 + mw, mouth_y + int(H * 0.05)],
          start=0, end=180, fill=(120, 70, 60), width=max(2, int(H * 0.012)))
    return img


def _mock_image_to_mesh(image_path: str, out_glb_path: str) -> str:
    """Placeholder front-end: ellipsoid head + procedural texture -> GLB."""
    try:
        import trimesh
    except ImportError as e:  # pragma: no cover
        raise TrellisFrontError(
            f"mock mode needs trimesh (pip install trimesh): {e}") from e

    # size the head from the input image aspect ratio (width/height)
    ax, ay = 1.0, 1.0
    if image_path and os.path.exists(image_path):
        try:
            from PIL import Image
            w0, h0 = Image.open(image_path).size
            ax = w0 / max(h0, 1)
            ay = h0 / max(w0, 1)
        except Exception:
            pass
    rx, ry, rz = 0.95 * ax, 1.20 * ay, 0.95
    # dense enough that stage1_real.geometric_anchors finds its head cluster
    # (top 18% height band, central 20% width band) with >= 100 vertices:
    # n_lat=48/n_lon=96 -> 354 cluster verts; n_lat=24/n_lon=48 -> only 84.
    V, F = _ellipsoid_mesh(rx, ry, rz, n_lat=48, n_lon=96)

    uv = _spherical_uv(V / np.array([rx, ry, rz]))
    tex = _placeholder_texture(image_path)
    visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
    mesh = trimesh.Trimesh(vertices=V, faces=F, visual=visual, process=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_glb_path)) or ".", exist_ok=True)
    # glb export with embedded texture
    mesh.export(out_glb_path, file_type="glb")
    return out_glb_path


# ---------------------------------------------------------------------------
# real front-end: TRELLIS image-to-3D (weights + GPU required)
# ---------------------------------------------------------------------------

def _weights_dir() -> str:
    return os.environ.get("TRELLIS_MODEL_PATH", DEFAULT_WEIGHTS_DIR)


def _dinov2_hub_cache_path() -> str:
    torch_home = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"),
                                                           ".cache", "torch"))
    return os.path.join(torch_home, "hub", "checkpoints",
                        "dinov2_vitl14_reg_pretrain.pth")


def _check_weights(model_path: str) -> None:
    """Raise TrellisFrontError with an actionable message if weights missing."""
    problems = []
    pipeline = os.path.join(model_path, "pipeline.json")
    if not os.path.exists(pipeline):
        problems.append(f"missing {pipeline}")
    for name in INFERENCE_CKPTS:
        for ext in (".json", ".safetensors"):
            p = os.path.join(model_path, "ckpts", name + ext)
            if not os.path.exists(p):
                problems.append(f"missing {p}")
    if problems:
        raise TrellisFrontError(
            "TRELLIS weights are not complete at "
            f"'{model_path}'.\n  " + "\n  ".join(problems) +
            "\n\nFix: run code/scripts/deploy_trellis.sh (downloads "
            "TRELLIS-image-large from hf-mirror with curl -C - resume; "
            "measured ~8-11 MB/s, ~6 min for the 3.3 GB). Or set "
            "TRELLIS_MODEL_PATH to the weights dir, or to "
            f"'{TRELLIS_HF_REPO}' to use huggingface_hub directly.")

    # image encoder (dinov2_vitl14_reg) — torch hub cache or explicit pth
    dino = os.environ.get("TRELLIS_DINOV2_PTH") or _dinov2_hub_cache_path()
    if not os.path.exists(dino):
        raise TrellisFrontError(
            "TRELLIS image encoder weights (dinov2_vitl14_reg) not found at "
            f"'{dino}' (torch.hub would fetch them from dl.fbaipublicfiles.com, "
            "which is blocked on this network).\n\nFix: deploy_trellis.sh "
            "downloads them from hf-mirror into the torch hub cache, or set "
            "TRELLIS_DINOV2_PTH to the .pth path.")


def _patch_torch_hub_dinov2(pth_path: str) -> None:
    """Make torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    use a local .pth instead of the blocked dl.fbaipublicfiles.com URL."""
    import torch.hub as hub

    if os.path.exists(_dinov2_hub_cache_path()):
        return  # cache pre-seeded; torch.hub already skips the network
    orig = hub.load

    def patched(repo_or_dir, model, *args, **kwargs):
        if (repo_or_dir == "facebookresearch/dinov2"
                and model == "dinov2_vitl14_reg"):
            kwargs.setdefault("pretrained", True)
            kwargs["weights"] = pth_path  # str path -> local file:// URL
        return orig(repo_or_dir, model, *args, **kwargs)

    hub.load = patched


def _trellis_image_to_mesh(image_path: str, out_glb_path: str, device: str,
                           simplify: float, texture_size: int, seed: int,
                           use_rembg: bool, model_path: str,
                           formats: Optional[List[str]] = None) -> str:
    """Real TRELLIS inference: image -> PBR-textured mesh GLB."""
    _check_weights(model_path)

    try:
        import torch
        from PIL import Image
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils
    except ImportError as e:
        raise TrellisFrontError(
            "TRELLIS runtime not installed (need the microsoft/TRELLIS "
            f"source checkout + torch + deps): {e}\n"
            "Fix: run code/scripts/deploy_trellis.sh") from e

    if not torch.cuda.is_available():
        raise TrellisFrontError(
            f"device '{device}' requested but torch.cuda.is_available() is "
            "False. TRELLIS v1 needs an NVIDIA GPU with >= 16 GB VRAM "
            "(A100/A6000 verified by the authors). Run the real front-end on "
            "the GPU server; use --mock for CPU-only contract testing.")

    # dinov2 image encoder: prefer pre-seeded torch hub cache, else local pth
    dino = os.environ.get("TRELLIS_DINOV2_PTH") or _dinov2_hub_cache_path()
    if os.path.exists(dino):
        _patch_torch_hub_dinov2(dino)

    # spconv needs this set before the first pipeline call (README)
    os.environ.setdefault("SPCONV_ALGO", "native")

    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.to(torch.device(device))

    image = Image.open(image_path).convert("RGB")
    preprocess = True
    if use_rembg:
        try:
            import rembg  # noqa: F401  (may download u2net.onnx on first use)
        except ImportError:
            warnings.warn("rembg not installed; skipping background removal "
                          "(naive crop will be used).", RuntimeWarning)
            preprocess = False
    else:
        preprocess = False

    outputs = pipeline.run(
        image,
        seed=seed,
        formats=formats or ["gaussian", "mesh"],  # skip radiance field decode
        preprocess_image=preprocess,
    )
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=simplify, texture_size=texture_size,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_glb_path)) or ".", exist_ok=True)
    glb.export(out_glb_path)
    return out_glb_path


# ---------------------------------------------------------------------------
# public API + CLI
# ---------------------------------------------------------------------------

def image_to_mesh(image_path: str, out_glb_path: str, device: str = "cuda",
                  mock: bool = False, simplify: float = 0.95,
                  texture_size: int = 1024, seed: int = 1,
                  use_rembg: bool = True,
                  model_path: Optional[str] = None) -> str:
    """image -> static textured mesh GLB (real TRELLIS or --mock placeholder).

    Returns ``out_glb_path``, ready to be passed to
    ``stage1_real.py --glb <out_glb_path>``.
    """
    if not os.path.exists(image_path):
        raise TrellisFrontError(f"input image not found: {image_path}")
    if mock:
        return _mock_image_to_mesh(image_path, out_glb_path)
    return _trellis_image_to_mesh(
        image_path, out_glb_path, device=device, simplify=simplify,
        texture_size=texture_size, seed=seed, use_rembg=use_rembg,
        model_path=model_path or _weights_dir())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="TRELLIS image-to-mesh front-end (issue #3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--out", required=True, help="output .glb path")
    ap.add_argument("--device", default="cuda",
                    help="torch device for real mode (cuda/cpu)")
    ap.add_argument("--mock", action="store_true",
                    help="use the placeholder ellipsoid-head front-end "
                         "(no weights, no GPU)")
    ap.add_argument("--model-path", default=None,
                    help="TRELLIS-image-large weights dir (default: "
                         f"{DEFAULT_WEIGHTS_DIR})")
    ap.add_argument("--simplify", type=float, default=0.95,
                    help="TRELLIS mesh simplification ratio")
    ap.add_argument("--texture-size", type=int, default=1024,
                    help="TRELLIS GLB texture size")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-rembg", action="store_true",
                    help="skip rembg background removal (real mode)")
    args = ap.parse_args(argv)

    try:
        out = image_to_mesh(
            args.image, args.out, device=args.device, mock=args.mock,
            simplify=args.simplify, texture_size=args.texture_size,
            seed=args.seed, use_rembg=not args.no_rembg,
            model_path=args.model_path)
    except TrellisFrontError as e:
        print(f"[trellis-front] error: {e}", file=sys.stderr)
        return 1
    size = os.path.getsize(out)
    print(json.dumps({
        "mode": "mock" if args.mock else "trellis",
        "out": out, "bytes": size,
        "next": f"python scripts/stage1_real.py --glb {out}",
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
