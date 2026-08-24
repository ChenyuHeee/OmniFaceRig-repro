# code/ — OmniFaceRig reproduction pipeline

Pure-Python (numpy/scipy/trimesh/pygltflib, all MIT/Apache/BSD — no GPL)
implementation of the paper pipeline (arXiv:2606.08043), CPU-runnable for
the geometry stages; the image-to-3D front and the parsing models run on the
A100 server.

## Package layout

| module | stage | contents |
|---|---|---|
| `omnifacerig_repro/arkit52.py` | data | ARKit 52 blendshape list (Apple), FACS AU → ARKit mapping tables, Core-13 dialog set |
| `omnifacerig_repro/geometry.py` | Stage 2 | cotangent Laplacian, ARAP (Sorkine & Alexa), sparse deformation transfer (Sumner & Popovic, vectorized, with v4 third-edge + vertex pin), Delta Mush, SDF helpers (trimesh) |
| `omnifacerig_repro/template_fit.py` | Stage 1 | rigid alignment (paper Eq. 1, Umeyama) + non-rigid fit (paper Eq. 2: E_corr Huber / E_smooth / E_edge / E_tri / E_flip / E_reg, analytic gradient, L-BFGS-B) |
| `omnifacerig_repro/lip_sync.py` | deliverable 5 | Mandarin pinyin → viseme (Oculus 15) → ARKit weights; English Rhubarb viseme → ARKit; timing comes from the TTS front-end |
| `omnifacerig_repro/glb_export.py` | output | pygltflib GLB writer: POSITION/NORMAL/TEXCOORD + 52 morph targets (extras.targetNames) + skin (JOINTS_0/WEIGHTS_0 + IBM) + WEIGHTS animation |
| `omnifacerig_repro/pipeline.py` | orchestrator | `run_demo()`: synthetic head → Stage 1 fit → face fusion → 52 ARKit morphs (DT + Delta Mush) → skeleton → viseme animation → `outputs/demo.glb` |

## Run

```bash
cd code
pip install -e .            # or: pip install numpy scipy trimesh pygltflib pypinyin pytest
python -m omnifacerig_repro.pipeline   # writes outputs/demo.glb
python -m pytest ../tests   # 28 tests
```

## Status vs. A100/server stages

Implemented and tested on CPU:
- Stage 1 template registration (Eq. 1/2) with keypoint fallbacks
- Stage 2 face fusion (convex-hull cut + nearest-point welding),
  deformation transfer (same-topology), Delta Mush, SDF push-out helpers
- ARKit 52 morph-target GLB export (validated round-trip)
- Mandarin + English viseme → ARKit mapping tables

Requires the A100 server (next steps):
- image-to-3D front (TRELLIS v1/v2) for "upload an image" input
- face parsing ensemble (MediaPipe / fine-tuned Sapiens / SAM) for real
  keypoints; current demo uses synthetic keypoints
- inner-mouth archetypes (ICT-FaceKit teeth/gums/tongue) + ARAP/SDF fitting
- Mixamo-style body skeleton auto-rig (the official demo GLB uses
  `mixamorig:*` joint naming) and full-body animation clips
- web preview service on the reserved ports (32170→8000 / 32171→8001)

Reference formats (verified from the TA resources 2026-08-24):
- input `glb and image testing/glb/ai3d_01.glb`: static mesh, ~1M verts,
  POSITION/NORMAL/TEXCOORD_0, no skeleton/morphs
- output `FINAL_WORK_DEMO.glb`: 53-joint Mixamo skeleton + 36 animations +
  exactly the 52 ARKit blendshape targets + separate inner-mouth meshes
