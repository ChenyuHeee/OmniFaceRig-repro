"""OmniFaceRig reproduction pipeline.

Pipeline overview (paper arXiv:2606.08043, SIGGRAPH Asia/TOG 2026):

    image -> [image-to-3D front, needs GPU] -> static mesh
           -> Stage 1: face template fitting (rigid Eq.1 + non-rigid Eq.2)
           -> Stage 2: face fusion / inner-mouth (ARAP+SDF) / UV repack
                      / FACS->ARKit blendshape transfer (deformation transfer
                        + Delta Mush) -> rigged GLB (morph targets + skeleton)

Modules:
    arkit52       -- ARKit 52 blendshape names + FACS AU mapping tables
    geometry      -- cotangent Laplacian / ARAP / deformation transfer / Delta Mush
    template_fit  -- Stage 1 rigid (Eq.1) + non-rigid (Eq.2) template registration
    lip_sync      -- Mandarin + English phoneme -> viseme -> ARKit weights
    glb_export    -- pygltflib-based GLB writer (morph targets + skinning)
    inner_mouth   -- inner-mouth assets (teeth/gums/tongue): ICT-FaceKit or
                     procedural, archetypes (Table 6), ARAP+RBF placement,
                     SDF penetration refinement, jaw-follow morphs (Sec. 3.6.2)
    pipeline      -- orchestrator with a CPU-runnable end-to-end demo
"""

__version__ = "0.1.0"
