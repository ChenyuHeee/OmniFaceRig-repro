"""Tests for the TRELLIS image-to-mesh front-end contract (issue #3).

The two contract points under test:
  1. mock mode  -> a valid textured GLB that stage1_real.py --glb can consume
     (POSITION + indices via pygltflib, plus texture data);
  2. real mode  -> raises a clear, actionable error when the TRELLIS
     weights/runtime are missing (no silent failure).
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
SCRIPTS = os.path.join(CODE, "scripts")
for p in (CODE, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from trellis_front import TrellisFrontError, image_to_mesh  # noqa: E402


def _make_test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (64, 80), (200, 180, 160))
    p = os.path.join(tmp_path, "input.png")
    img.save(p)
    return p


def test_mock_produces_glb_consumable_by_stage1(tmp_path):
    """mock image_to_mesh -> GLB with V/F + texture that stage1 can load."""
    img = _make_test_image(tmp_path)
    out = os.path.join(tmp_path, "mock_head.glb")

    ret = image_to_mesh(img, out, mock=True)
    assert ret == out
    assert os.path.getsize(out) > 1000

    # input contract of stage1_real.py: first primitive POSITION + indices
    from stage1_real import load_mesh
    V, F, gltf = load_mesh(out)
    assert len(V) > 100 and len(F) > 100
    assert V.shape[1] == 3 and F.shape[1] == 3
    assert np.isfinite(V).all()

    # "纹理" half of the contract: TEXCOORD_0 + material/texture in the glb
    prim = gltf.meshes[0].primitives[0]
    assert prim.attributes.TEXCOORD_0 is not None
    assert gltf.materials and gltf.textures


def test_real_without_weights_raises_clear_error(tmp_path):
    """real mode with missing weights must fail loudly with a fix hint."""
    img = _make_test_image(tmp_path)
    out = os.path.join(tmp_path, "real.glb")
    with pytest.raises(TrellisFrontError) as ei:
        image_to_mesh(img, out, device="cpu", model_path="/nonexistent/trellis")
    msg = str(ei.value)
    assert "pipeline.json" in msg or "weights" in msg
    assert "deploy_trellis.sh" in msg  # actionable recovery path


def test_missing_input_image_raises(tmp_path):
    """missing image -> TrellisFrontError even in mock mode."""
    with pytest.raises(TrellisFrontError):
        image_to_mesh(os.path.join(tmp_path, "nope.png"), "x.glb", mock=True)
