# vendored: vasiliskatr/deformation_transfer_ARkit_blendshapes

MIT-licensed reference implementation of sparse deformation transfer
(Sumner & Popovic 2004) applied to ARKit blendshape generation. Vendored on
2026-08-24 from https://github.com/vasiliskatr/deformation_transfer_ARkit_blendshapes
(commit 5faaf6a), under its MIT license (see `LICENSE` in this directory).

## What is kept

- `local_packages/deformationTransfer.py` — the S&P solver (numba-based)
- `local_packages/tools3d_.py` — OBJ IO / rigid alignment helpers
- `landmarks/` — landmark index lists for the ARKit (1220-vtx) and ICT
  FaceKit (26719-vtx) topologies
- `LICENSE` — MIT, Copyright (c) 2022 vasiliskatr
- `README.upstream.md` — the original README (kept for attribution)

## What is NOT kept, and why

- `data/` (Neutral.obj + 52 ARKit blendshape OBJs) was **excluded**: the OBJ
  headers carry a CARV3D notice — "It is illegal to use this identity without
  the approval of CARV3D" — i.e. the meshes are a scanned real person's
  identity, not free assets despite the repo-level MIT license. Downloading
  them from the upstream repo for internal reference is at your own risk;
  do **not** ship any geometry derived from them in the challenge deliverable.

## Relationship to our implementation

`omnifacerig_repro/geometry.py::deformation_transfer` reimplements the same
equations (E' Vinv = T with the v4 third-edge convention) vectorized with
scipy only (no numba), plus a vertex pin to fix the translation nullspace.
The vendored code is kept as a reference oracle for validation.
