# OmniFaceRig-repro

Reproduction of **[OmniFaceRig](https://omnifacerig.github.io/)** (Meta Reality Labs, SIGGRAPH Asia / TOG 2026, arXiv:2606.08043):
**upload a 2D character image → get a rigged GLB with a motion skeleton + 52 ARKit facial blendshapes + real audio lip-sync (Chinese & English).**

## 🖥️ Live demo (A100 server)

> **http://175.155.64.171:32170/** — pick a pre-rigged character (23 official T-pose avatars), preview 52 blendshapes, play lip-sync animations, or upload your own image for the full pipeline (image → mesh → rig → expression → lip-sync, ~2–3 min).

## Pipeline

```
2D image ──► TRELLIS (image→3D mesh, ~2min) ──► stage1_real.py
                (or official T-pose GLB)           ├─ 53-joint Mixamo-style skeleton (proportional skinning)
                                                  ├─ 52 ARKit blendshapes (sparse morph targets, delta-field smoothed)
                                                  ├─ inner mouth: teeth / gums / tongue (ICT-FaceKit assets + ARAP/SDF)
                                                  └─ viseme idle animation
       ──► animate_audio.py: text → piper TTS → faster-whisper word timestamps → visemes → ARKit WEIGHTS animation
```

## Deliverables / acceptance (all verified on 23/23 official characters)

| Check | Result |
|---|---|
| Motion skeleton | 53 Mixamo-style joints per character (requirement ≥ 20) |
| Unbroken expressions | flipped-triangle area 0.003%–0.044% at weight 1.0 (threshold < 0.1%) |
| Inner mouth | teeth upper/lower + gums/tongue, positioned & animated with the jaw |
| ARKit 52 | target names exactly match the official ARKit 52 set |
| Lip-sync | Chinese (zh_CN-huayan) & English (en_US-lessac) piper voices + faster-whisper alignment |

## Repo layout

```
code/
  omnifacerig_repro/   core library: glb export, arkit52, inner_mouth, lip_sync, pipeline
  scripts/             stage1_real.py (rig), trellis_front.py (image→mesh), animate_audio.py, deliverables_check.py
  webapp/              Flask preview app (self-hosted three.js, no CDN)
  vendor/              ICT-FaceKit & Vasiliskatr (licensed, see their LICENSE files)
tests/                 60+ unit tests (pytest)
paper/                 OmniFaceRig paper (arXiv)
notes/                 progress, acceptance report, component deep-dives
docs/assignment/       original challenge requirements (text summary)
```

## Run locally

```bash
cd code
pip install -e .            # pygltflib, numpy, scipy, pyyaml, ...
python -m pytest ../tests   # 60+ tests, no external assets needed

# rig an official T-pose GLB:
python scripts/stage1_real.py --glb avatar.glb --inner-mouth --out rigged.glb --text "Hello world"
# audio lip-sync on the rigged result:
python scripts/animate_audio.py --glb rigged.glb --out talk.glb --text "Hello world" --lang en
# image → mesh (needs TRELLIS weights + GPU):
python scripts/trellis_front.py --image avatar.png --out mesh.glb
```

The full image→rigged loop also runs from the web UI (`image_to_mesh=1`), which the A100 deployment uses.

## Notes

- The 23 official T-pose avatar GLBs and 2D images are course-provided assets and are **not** redistributed in this repository (see `docs/assignment/` for the original requirements).
- Vendored third-party code keeps its own licenses (`code/vendor/*/LICENSE`); three.js r160 is vendored for the offline web preview.
- Full technical write-ups live in `notes/` (deployment records, acceptance report, per-component deep dives).
