"""GLB export with morph targets (ARKit 52) + optional skeleton (skinning).

pygltflib-based (MIT), per the stack decision in
notes/components/03_geometry_rigging.md: trimesh cannot export morph
targets/skinning, so the final glb is assembled here.

Output layout:
  - one mesh, one primitive: POSITION + (optional) NORMAL + (optional)
    JOINTS_0/WEIGHTS_0, plus one morph target per ARKit shape (POSITION
    deltas), names in mesh.extras.targetNames (three.js / Blender
    convention)
  - optional Skin with inverse bind matrices and a WEIGHTS animation
    channel so the blendshape rig is immediately previewable
"""

from __future__ import annotations

import numpy as np
import pygltflib

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UNSIGNED_INT = 5125
_UNSIGNED_SHORT = 5123


class _BinaryBuilder:
    """Appends typed arrays into one 4-byte-aligned binary blob."""

    def __init__(self, gltf: pygltflib.GLTF2):
        self.gltf = gltf
        self.parts: list[bytes] = []
        self.offset = 0
        self.buffer_views: list[int] = []

    def _pad(self, n: int) -> None:
        if self.offset % n:
            pad = n - (self.offset % n)
            self.parts.append(b"\x00" * pad)
            self.offset += pad

    def add_view(self, arr: np.ndarray, target: int) -> int:
        self._pad(4)
        view = pygltflib.BufferView(
            buffer=0, byteOffset=self.offset, byteLength=arr.nbytes, target=target,
        )
        self.gltf.bufferViews.append(view)
        self.buffer_views.append(len(self.gltf.bufferViews) - 1)
        self.parts.append(arr.tobytes())
        self.offset += arr.nbytes
        return self.buffer_views[-1]

    def add_accessor(self, arr: np.ndarray, target: int, type_: str,
                     component_type: int, with_minmax: bool = True) -> int:
        view = self.add_view(arr, target)
        acc = pygltflib.Accessor(
            bufferView=view, byteOffset=0, componentType=component_type,
            count=len(arr), type=type_,
        )
        if with_minmax and component_type == _FLOAT:
            # glTF requires min/max as arrays (one element per component)
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            acc.min = [float(lo)] if np.isscalar(lo) else np.asarray(lo).tolist()
            acc.max = [float(hi)] if np.isscalar(hi) else np.asarray(hi).tolist()
        self.gltf.accessors.append(acc)
        return len(self.gltf.accessors) - 1

    def add_sparse_accessor(
        self, full_count: int, values: np.ndarray, indices: np.ndarray,
        type_: str = "VEC3", component_type: int = _FLOAT,
    ) -> int:
        """Sparse accessor: full_count elements, only `indices` carry values.

        Used for morph targets so a 1M-vertex mesh with a small face region
        stays small (the official demo stores ~1571 face verts per shape).
        """
        assert component_type == _FLOAT and type_ == "VEC3"
        idx_view = self.add_view(np.asarray(indices, dtype=np.uint32),
                                 _ARRAY_BUFFER)
        val_view = self.add_view(np.asarray(values, dtype=np.float32),
                                 _ARRAY_BUFFER)
        sparse = pygltflib.Sparse(
            count=len(indices),
            indices=pygltflib.AccessorSparseIndices(
                bufferView=idx_view, byteOffset=0, componentType=_UNSIGNED_INT),
            values=pygltflib.AccessorSparseValues(
                bufferView=val_view, byteOffset=0),
        )
        acc = pygltflib.Accessor(
            componentType=_FLOAT, count=full_count, type="VEC3", sparse=sparse,
        )
        lo, hi = values.min(axis=0), values.max(axis=0)
        acc.min = np.asarray(lo).tolist()
        acc.max = np.asarray(hi).tolist()
        self.gltf.accessors.append(acc)
        return len(self.gltf.accessors) - 1

    def finish(self) -> None:
        self._pad(4)
        blob = b"".join(self.parts)
        self.gltf.buffers[0].byteLength = len(blob)
        self.gltf.set_binary_blob(blob)


def build_gltf(
    V: np.ndarray,
    F: np.ndarray,
    morphs: dict[str, np.ndarray],
    normals: np.ndarray | None = None,
    skin: dict | None = None,
    animation: dict | None = None,
    name: str = "character",
    sparse_morphs: bool = True,
    morph_eps: float = 1e-9,
) -> pygltflib.GLTF2:
    """Assemble a GLTF2 object.

    morphs:   {name: (n,3) vertex deltas} - morph target order is preserved
    skin:     optional {
                joint_names: [str], joint_transforms: [(4,4) world matrices],
                joint_indices: (n,k) uint16, joint_weights: (n,k) float32 }
    animation: optional {"times": (t,), "weights": (t, K) float32} - WEIGHTS
               channel on the mesh node.
    sparse_morphs: store morph deltas as sparse accessors (only non-zero
               vertices), like the official FINAL_WORK_DEMO.glb (1571/1M).
    """
    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.uint32 if len(V) > 65535 else np.uint16)
    n = len(V)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)

    gltf = pygltflib.GLTF2()
    gltf.asset = pygltflib.Asset(version="2.0", generator="omnifacerig-repro")
    gltf.buffers = [pygltflib.Buffer()]
    gltf.bufferViews = []
    gltf.accessors = []
    bb = _BinaryBuilder(gltf)

    # --- geometry ---
    pos_acc = bb.add_accessor(V, _ARRAY_BUFFER, "VEC3", _FLOAT)
    if normals is not None:
        nrm_acc = bb.add_accessor(normals, _ARRAY_BUFFER, "VEC3", _FLOAT)
    idx_acc = bb.add_accessor(F.ravel(), _ELEMENT_ARRAY_BUFFER, "SCALAR",
                              _UNSIGNED_INT if F.dtype == np.uint32 else _UNSIGNED_SHORT,
                              with_minmax=False)

    attrs = pygltflib.Attributes(POSITION=pos_acc)
    if normals is not None:
        attrs.NORMAL = nrm_acc
    prim = pygltflib.Primitive(attributes=attrs, indices=idx_acc)

    # --- morph targets ---
    target_accs: list[int] = []
    zero_acc: int | None = None
    for name_key in morphs:
        delta = np.asarray(morphs[name_key], dtype=np.float32)
        assert delta.shape == (n, 3), f"morph {name_key}: delta shape {delta.shape} != {(n,3)}"
        if sparse_morphs:
            nz = np.any(np.abs(delta) > morph_eps, axis=1)
            if nz.any():
                idx = np.where(nz)[0]
                target_accs.append(bb.add_sparse_accessor(n, delta[idx], idx))
            else:
                # empty shape: share one dense zero accessor across shapes
                if zero_acc is None:
                    zero_acc = bb.add_accessor(delta, _ARRAY_BUFFER, "VEC3", _FLOAT)
                target_accs.append(zero_acc)
        else:
            target_accs.append(bb.add_accessor(delta, _ARRAY_BUFFER, "VEC3", _FLOAT))
    if target_accs:
        prim.targets = [pygltflib.Attributes(POSITION=a) for a in target_accs]

    mesh = pygltflib.Mesh(name=name, primitives=[prim])
    mesh.weights = [0.0] * len(target_accs)
    if target_accs:
        mesh.extras = {"targetNames": list(morphs.keys())}
    gltf.meshes.append(mesh)

    mesh_node = pygltflib.Node(name=name, mesh=0)
    gltf.nodes.append(mesh_node)
    mesh_node_idx = len(gltf.nodes) - 1  # 0

    # --- skinning ---
    skin_index = None
    if skin is not None:
        joints = skin["joint_names"]
        transforms = skin["joint_transforms"]
        jidx = np.asarray(skin["joint_indices"], dtype=np.uint16)
        jw = np.asarray(skin["joint_weights"], dtype=np.float32)
        assert jidx.shape == jw.shape and jidx.shape[1] <= len(joints)
        assert jidx.max() < len(joints), "joint index out of range"
        assert jidx.shape[0] == n, "skin weights must cover every vertex"
        # glTF requires vec4 attributes: pad with joint 0 at zero weight
        k = jidx.shape[1]
        if k < 4:
            pad_j = np.zeros((n, 4 - k), dtype=np.uint16)
            pad_w = np.zeros((n, 4 - k), dtype=np.float32)
            jidx = np.hstack([jidx, pad_j])
            jw = np.hstack([jw, pad_w])
        # joint nodes + inverse bind matrices
        joint_nodes: list[int] = []
        for jname, T in zip(joints, transforms):
            pos = T[:3, 3]
            node = pygltflib.Node(name=jname, translation=pos.tolist())
            gltf.nodes.append(node)
            joint_nodes.append(len(gltf.nodes) - 1)
        ibm_arr = np.asarray(
            [np.linalg.inv(T).T.reshape(16) for T in transforms], dtype=np.float32
        )  # column-major 4x4 per joint
        ibm_acc = bb.add_accessor(ibm_arr, _ARRAY_BUFFER, "MAT4", _FLOAT)
        joints_acc = bb.add_accessor(jidx, _ARRAY_BUFFER, "VEC4", _UNSIGNED_SHORT,
                                     with_minmax=False)
        weights_acc = bb.add_accessor(jw, _ARRAY_BUFFER, "VEC4", _FLOAT)
        skin_obj = pygltflib.Skin(
            inverseBindMatrices=ibm_acc, joints=joint_nodes, skeleton=joint_nodes[0],
        )
        gltf.skins.append(skin_obj)
        skin_index = 0
        attrs.JOINTS_0 = joints_acc
        attrs.WEIGHTS_0 = weights_acc
        gltf.scenes = [pygltflib.Scene(nodes=joint_nodes + [mesh_node_idx])]
    else:
        gltf.scenes = [pygltflib.Scene(nodes=[mesh_node_idx])]
    gltf.scene = 0
    if skin_index is not None:
        mesh_node.skin = skin_index

    # --- animation (blendshape weights) ---
    if animation is not None:
        times = np.asarray(animation["times"], dtype=np.float32)
        weights = np.asarray(animation["weights"], dtype=np.float32)
        assert weights.shape == (len(times), len(target_accs))
        t_acc = bb.add_accessor(times, _ARRAY_BUFFER, "SCALAR", _FLOAT)
        w_acc = bb.add_accessor(weights, _ARRAY_BUFFER, "SCALAR", _FLOAT)
        sampler = pygltflib.AnimationSampler(input=t_acc, output=w_acc)
        channel = pygltflib.AnimationChannel(
            sampler=0,
            target=pygltflib.AnimationChannelTarget(node=0, path="weights"),
        )
        gltf.animations = [pygltflib.Animation(channels=[channel], samplers=[sampler])]

    bb.finish()
    return gltf


def write_glb(path: str, gltf: pygltflib.GLTF2) -> str:
    """Serialize GLTF2 to a .glb file."""
    gltf.save_binary(path)
    return path


def load_glb(path: str) -> pygltflib.GLTF2:
    """Read back a .glb (for verification)."""
    with open(path, "rb") as fh:
        return pygltflib.GLTF2().load_from_bytes(fh.read())
