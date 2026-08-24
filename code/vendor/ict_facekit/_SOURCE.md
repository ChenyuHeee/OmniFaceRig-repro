# ICT-FaceKit (vendored subset) — provenance

- **Upstream**: https://github.com/USC-ICT/ICT-FaceKit (commit `da5f95a`, master)
- **License**: MIT — Copyright (c) 2020 USC Institute for Creative Technologies
  (see `LICENSE` in this directory)
- **Downloaded**: 2026-08-25 via `curl -L https://api.github.com/repos/USC-ICT/ICT-FaceKit/tarball/master`
  (git over 443 is blocked on this machine; the GitHub API channel works)
- **Kept** (what this pipeline uses):
  - `LICENSE`, `README.md` — license + topology documentation
  - `FaceXModel/generic_neutral_mesh.obj` — the 26719-vertex neutral head mesh;
    the Teeth (`[17039:21450]` verts / `[17006:21495]` polys) and Gums-and-tongue
    (`[14062:17038]` / `[14034:17005]`) sub-meshes are extracted by
    `omnifacerig_repro/inner_mouth.py::load_ict_facekit`
  - `FaceXModel/vertex_indices.json` — vertex group definitions
  - `FaceXModel/ICTFaceModelMaterial.mtl` — material reference
  - `Scripts/ict_face_model.py`, `Scripts/face_model_io.py` — official loader (reference)
- **Not kept**: the full tarball (~133 MB, includes 53 identity blendshape OBJs and
  sample data) is NOT committed — the inner-mouth pipeline only needs the neutral
  mesh. Re-download with the command above if the full model is ever needed.
- **Identity note**: ICT-FaceKit's Light model is a template/statistical model, NOT
  a scanned individual's identity (unlike the CARV3D-marked meshes excluded from
  `code/vendor/vasiliskatr/`); it is safe to ship under its MIT license.
