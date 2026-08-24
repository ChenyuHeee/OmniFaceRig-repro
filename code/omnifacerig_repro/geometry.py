"""Geometry primitives for the reproduction pipeline.

All implementations are self-written on numpy/scipy (MIT) per the tech-stack
decision in notes/components/03_geometry_rigging.md (no GPL: no libigl, no
CGAL). Algorithms:

* cotangent Laplacian / uniform smoothing matrix
* ARAP deformation (Sorkine & Alexa 2007) - local-global with SVD rotations
* sparse deformation transfer (Sumner & Popovic 2004) - per-triangle affine
  transfer with v4 (third-edge) unknowns, data/identity/smoothness terms;
  formulation follows the MIT reference vasiliskatr/deformation_transfer_
  ARkit_blendshapes (vendored under code/vendor/), rewritten vectorized
  without numba
* Delta Mush smoothing (detail-preserving smoothing of transferred shapes)
* SDF helpers (trimesh) for inner-mouth collision refinement

Conventions: V (n,3) float64 vertex positions, F (m,3) int triangle indices.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# ---------------------------------------------------------------------------
# Normals & Laplacians
# ---------------------------------------------------------------------------

def face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Unnormalized triangle normals (cross products), shape (m,3)."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    return np.cross(e1, e2)


def vertex_normals(V: np.ndarray, F: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Area-weighted vertex normals, shape (n,3)."""
    fn = face_normals(V, F)
    nrm = np.zeros_like(V)
    np.add.at(nrm, F[:, 0], fn)
    np.add.at(nrm, F[:, 1], fn)
    np.add.at(nrm, F[:, 2], fn)
    if normalize:
        lens = np.linalg.norm(nrm, axis=1, keepdims=True)
        lens[lens == 0] = 1.0
        nrm /= lens
    return nrm


def _unique_edges(F: np.ndarray) -> np.ndarray:
    """Unique undirected edges, shape (e,2), sorted."""
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def cotangent_laplacian(V: np.ndarray, F: np.ndarray) -> sparse.csr_matrix:
    """Cotangent Laplacian L (n,n). (L @ x)[i] = sum_j w_ij (x_i - x_j),
    with w_ij = 0.5 * (cot alpha + cot beta) over the two triangles
    incident to edge (i,j)."""
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = len(V)
    e_ab = V[F[:, 1]] - V[F[:, 0]]
    e_ac = V[F[:, 2]] - V[F[:, 0]]
    e_bc = V[F[:, 2]] - V[F[:, 1]]
    # cotangent of the angle at each corner (0 for degenerate triangles)
    def _cot(u, v):
        denom = np.linalg.norm(np.cross(u, v), axis=1)
        out = np.zeros(len(u))
        ok = denom > 1e-14
        out[ok] = np.einsum("ij,ij->i", u[ok], v[ok]) / denom[ok]
        return out

    cot_a = _cot(e_ab, e_ac)
    cot_b = _cot(-e_ab, e_bc)
    cot_c = _cot(-e_ac, -e_bc)
    # edge (a,b) opposite c -> cot_c ; edge (b,c) opposite a -> cot_a
    # edge (c,a) opposite b -> cot_b
    half = 0.5
    e_ab_w, e_bc_w, e_ca_w = half * cot_c, half * cot_a, half * cot_b
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0], F[:, 2]])
    vals = np.concatenate([-e_ab_w, -e_ab_w, -e_bc_w, -e_bc_w, -e_ca_w, -e_ca_w])
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    diag = -np.asarray(L.sum(axis=1)).ravel()  # +sum_j w_ij on the diagonal
    L = L + sparse.diags(diag)
    L.eliminate_zeros()
    return L.tocsr()


def smoothing_matrix(F: np.ndarray, n: int, normalize: bool = True) -> sparse.csr_matrix:
    """Uniform adjacency smoothing operator S: (S @ x)[i] = mean of x over
    neighbors of i (rest topology)."""
    e = _unique_edges(F)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    vals = np.ones(len(rows), dtype=float)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    if normalize:
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        S = sparse.diags(1.0 / deg) @ A
    else:
        S = A
    return S.tocsr()


# ---------------------------------------------------------------------------
# ARAP (Sorkine & Alexa 2007)
# ---------------------------------------------------------------------------

def arap_deform(
    V: np.ndarray,
    F: np.ndarray,
    handles: np.ndarray,
    target_pos: np.ndarray,
    iterations: int = 40,
    tol: float = 1e-9,
) -> np.ndarray:
    """As-rigid-as-possible deformation with pinned handle vertices.

    V:        rest positions (n,3)
    F:        triangles (m,3)
    handles:  vertex indices pinned to target_pos (k,)
    target_pos: target positions for the handles (k,3)
    Returns deformed positions (n,3).
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = len(V)
    L = cotangent_laplacian(V, F)
    L = L.tocsc()

    # neighbor structure with weights (from Laplacian off-diagonals)
    Lc = L.tocsr()
    neigh = []
    for i in range(n):
        start, stop = Lc.indptr[i], Lc.indptr[i + 1]
        neigh.append((Lc.indices[start:stop], Lc.data[start:stop]))

    is_handle = np.zeros(n, dtype=bool)
    is_handle[handles] = True

    Vp = V.copy()
    R = np.repeat(np.eye(3)[None, :, :], n, axis=0)

    # precompute rest edge vectors per vertex
    rest_edges = [None] * n
    for i in range(n):
        j_idx, w = neigh[i]
        rest_edges[i] = (j_idx, w, V[i] - V[j_idx])

    # pin handle rows of L
    Lpin = L.tolil()
    for h in handles:
        Lpin[h, :] = 0
        Lpin[h, h] = 1.0
    Lpin = Lpin.tocsc()
    try:
        solve = sparse.linalg.factorized(Lpin)
    except Exception:  # singular -> use spsolve fallback
        solve = None

    for _ in range(iterations):
        # --- local step: per-vertex rotation via SVD ---
        for i in range(n):
            j_idx, w, e = rest_edges[i]
            S = np.zeros((3, 3))
            for k, j in enumerate(j_idx):
                if j == i:
                    continue
                S += w[k] * np.outer(e[k], Vp[i] - Vp[j])
            U, _, Vt = np.linalg.svd(S)
            Rik = U @ Vt
            if np.linalg.det(Rik) < 0:
                U[:, -1] *= -1
                Rik = U @ Vt
            R[i] = Rik
        # --- global step: solve L V' = b ---
        b = np.zeros_like(Vp)
        for i in range(n):
            j_idx, w, e = rest_edges[i]
            acc = np.zeros(3)
            for k, j in enumerate(j_idx):
                acc += w[k] * (R[i] + R[j]) @ e[k]
            b[i] = 0.5 * acc
        for h, t in zip(handles, target_pos):
            b[h] = t
        if solve is not None:
            Vnew = np.column_stack([
                solve(b[:, 0]), solve(b[:, 1]), solve(b[:, 2]),
            ])
        else:
            Vnew = np.column_stack([
                spsolve(Lpin, b[:, 0]), spsolve(Lpin, b[:, 1]), spsolve(Lpin, b[:, 2]),
            ])
        change = np.max(np.abs(Vnew - Vp))
        Vp = Vnew
        if change < tol:
            break
    return Vp


# ---------------------------------------------------------------------------
# Sparse deformation transfer (Sumner & Popovic 2004)
# ---------------------------------------------------------------------------

def _v4(rest_V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face third vertex: a + cross(c-a, b-a)/sqrt(|cross|) (S&P, as in
    the vasiliskatr reference)."""
    e1 = rest_V[F[:, 1]] - rest_V[F[:, 0]]
    e2 = rest_V[F[:, 2]] - rest_V[F[:, 0]]
    cross = np.cross(e2, e1)
    norm = np.linalg.norm(cross, axis=1)
    norm[norm == 0] = 1.0
    return rest_V[F[:, 0]] + cross / np.sqrt(norm)[:, None]


def _valid_faces(rest_V: np.ndarray, F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Mask of non-degenerate triangles (non-zero rest area)."""
    e1 = rest_V[F[:, 1]] - rest_V[F[:, 0]]
    e2 = rest_V[F[:, 2]] - rest_V[F[:, 0]]
    return np.linalg.norm(np.cross(e1, e2), axis=1) > eps


def _face_inverses(rest_V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face inverse 3x3 matrices V_inv (m,3,3): E_t = [b-a, c-a, v4-a]."""
    v4 = _v4(rest_V, F)
    a = rest_V[F[:, 0]]
    E = np.stack([
        rest_V[F[:, 1]] - a,
        rest_V[F[:, 2]] - a,
        v4 - a,
    ], axis=-1)  # (m,3,3): columns are the 3 edge vectors
    return np.linalg.inv(E)


def _source_transforms(rest_V: np.ndarray, def_V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Per-face affine transforms T_t = E'_t E_t^{-1} (m,3,3) mapping source
    rest triangles to the deformed source triangles."""
    E_inv = _face_inverses(rest_V, F)
    a = def_V[F[:, 0]]
    E = np.stack([
        def_V[F[:, 1]] - a,
        def_V[F[:, 2]] - a,
        _v4(def_V, F) - a,
    ], axis=-1)
    return E @ E_inv


def _face_adjacency(F: np.ndarray) -> list[list[int]]:
    """Adjacent face lists (faces sharing an edge)."""
    m = len(F)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.repeat(np.arange(m), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, fid = key[order], fid[order]
    adj: list[list[int]] = [[] for _ in range(m)]
    i = 0
    while i < len(key):
        j = i + 1
        while j < len(key) and (key[j] == key[i]).all():
            j += 1
        ids = fid[i:j]
        if len(ids) == 2:
            adj[ids[0]].append(int(ids[1]))
            adj[ids[1]].append(int(ids[0]))
        i = j
    return adj


def _build_data_terms(
    target_V: np.ndarray, F: np.ndarray, T_rows: np.ndarray, faces: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Rows: E'_t(x, f_t) Vinv_t = T_row  (9 rows per target face).

    (E' Vinv)_{j,i} = sum_k E'_{j,k} Vinv_{k,i}; E' columns are the deformed
    edge vectors [x_b-x_a, x_c-x_a, f_t-x_a]. Matches the vasiliskatr
    reference layout: row 9r+3i+j carries
        -sum_k Vinv[k,i] at x_a[j], +Vinv[0,i] at x_b[j],
        +Vinv[1,i] at x_c[j], +Vinv[2,i] at f_t[j],
    rhs = T_rows[r, j, i]   (E' Vinv = T  <=>  E' = T E, the S&P objective).
    """
    m = len(F)
    n = len(target_V)
    Vinv = _face_inverses(target_V, F)
    rows_9 = len(faces) * 9
    ncols = 3 * (n + m)
    r_i, c_i, v_i = [], [], []
    rhs = np.zeros(rows_9)

    for r, t in enumerate(faces):
        a, b, c = F[t]
        Vt = Vinv[t]
        for i in range(3):          # column index of Vinv
            for j in range(3):      # component index (row of E')
                row = 9 * r + 3 * i + j
                rhs[row] = T_rows[r, j, i]
                r_i.append(row); c_i.append(3 * a + j)
                v_i.append(-(Vt[0, i] + Vt[1, i] + Vt[2, i]))
                r_i.append(row); c_i.append(3 * b + j); v_i.append(Vt[0, i])
                r_i.append(row); c_i.append(3 * c + j); v_i.append(Vt[1, i])
                r_i.append(row); c_i.append(3 * (n + t) + j); v_i.append(Vt[2, i])
    A = sparse.coo_matrix((v_i, (r_i, c_i)), shape=(rows_9, ncols)).tocsr()
    return A, rhs


def _build_identity_terms(target_V, F) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Rows: E'_t Vinv_t = I (regularizer pulling transforms to identity)."""
    ok = _valid_faces(target_V, F)
    Fok = F[ok]
    m = len(Fok)
    return _build_data_terms(
        target_V, Fok, np.repeat(np.eye(3)[None], m, axis=0), np.arange(m)
    )


def _build_smoothness_terms(target_V, F) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Rows: Vinv_i^T E'_i - Vinv_j^T E'_j = 0 for adjacent faces (i,j)."""
    ok = _valid_faces(target_V, F)
    Fok = F[ok]
    adj = _face_adjacency(Fok)
    pairs = [(i, j) for i, js in enumerate(adj) for j in js if i < j]
    m = len(Fok)
    n = len(target_V)
    Vinv = _face_inverses(target_V, Fok)
    rows_9 = len(pairs) * 9
    ncols = 3 * (n + m)
    r_i, c_i, v_i = [], [], []

    def emit(face, sign, row, i, j):
        a, b, c = F[face]
        Vt = Vinv[face]
        # (E' Vinv)_{j,i}: -sum_k Vt[k,i] at x_a[j], +Vt[0,i] at x_b[j], ...
        r_i.append(row); c_i.append(3 * a + j)
        v_i.append(sign * -(Vt[0, i] + Vt[1, i] + Vt[2, i]))
        r_i.append(row); c_i.append(3 * b + j); v_i.append(sign * Vt[0, i])
        r_i.append(row); c_i.append(3 * c + j); v_i.append(sign * Vt[1, i])
        r_i.append(row); c_i.append(3 * (n + face) + j); v_i.append(sign * Vt[2, i])

    for r, (i, j) in enumerate(pairs):
        for ii in range(3):
            for jj in range(3):
                row = 9 * r + 3 * ii + jj
                emit(i, +1.0, row, ii, jj)
                emit(j, -1.0, row, ii, jj)
    A = sparse.coo_matrix((v_i, (r_i, c_i)), shape=(rows_9, ncols)).tocsr()
    return A, np.zeros(rows_9)


def deformation_transfer(
    src_rest: np.ndarray,
    src_def: np.ndarray,
    F: np.ndarray,
    tgt_rest: np.ndarray,
    corr: np.ndarray | None = None,
    src_F: np.ndarray | None = None,
    wd: float = 1.0,
    ws: float = 1.0,
    wi: float = 0.1,
) -> np.ndarray:
    """Sparse deformation transfer (Sumner & Popovic 2004).

    Transfers the per-triangle affine deformation (src_rest -> src_def) onto
    the target mesh (tgt_rest) and returns the deformed target vertices.

    F:     target face indices (m_t,3).
    src_F: source face indices; default None = same topology as the target.
    corr:  optional (k,2) [src_face, tgt_face] correspondences; None means
           same-topology (face i of source <-> face i of target).
    Degenerate (zero-area rest) triangles are dropped from all terms.
    """
    src_F = F if src_F is None else src_F
    # ---- valid faces (non-degenerate rest triangles) ----
    ok_t = _valid_faces(tgt_rest, F)
    ok_s = _valid_faces(src_rest, src_F)
    if corr is None:
        assert len(src_F) == len(F), "corr required when source/target topology differ"
        keep = ok_t & ok_s
        src_faces = np.arange(len(F))[keep]
        faces = src_faces
    else:
        corr = np.asarray(corr, dtype=int)
        keep = ok_t[corr[:, 1]] & ok_s[corr[:, 0]]
        src_faces = corr[keep, 0]
        faces = corr[keep, 1]

    Fok = F[ok_t]
    new_idx = np.cumsum(ok_t) - 1  # original face -> index in Fok
    T_full = _source_transforms(src_rest, src_def, src_F)
    T_rows = T_full[src_faces]
    faces = new_idx[faces]

    Ad, bd = _build_data_terms(tgt_rest, Fok, T_rows, faces)
    Ai, bi = _build_identity_terms(tgt_rest, Fok)
    As, _bs = _build_smoothness_terms(tgt_rest, Fok)
    A = (wd * (Ad.T @ Ad) + wi * (Ai.T @ Ai) + ws * (As.T @ As)).tocsr()
    rhs = wd * (Ad.T @ bd) + wi * (Ai.T @ bi)
    n = len(tgt_rest)
    # The S&P objective is translation-invariant: pin vertex 0 to its rest
    # position to kill the nullspace (rotation/scale are fixed by T).
    w_pin = 1e4
    pin = sparse.lil_matrix((3, A.shape[1]))
    for k in range(3):
        pin[k, 3 * 0 + k] = w_pin
    A = (A + pin.T @ pin).tocsr()  # adds w_pin^2 on the diagonal
    for k in range(3):
        rhs[3 * 0 + k] += w_pin * w_pin * tgt_rest[0, k]
    # vertices untouched by any term stay at their rest position (pinned)
    ncols = A.shape[1]
    touched = np.diff(A.tocsr().indptr)  # per-column nnz of A (symmetric)
    if len(touched) < ncols:
        touched = np.concatenate([touched, np.zeros(ncols - len(touched), dtype=int)])
    free_cols = np.where(touched[: 3 * n] == 0)[0]
    if len(free_cols):
        A = A.tolil()
        for c in free_cols:
            A[c, c] = 1.0
            rhs[c] = tgt_rest.ravel()[c]
        A = A.tocsr()
    x = spsolve(A, rhs)
    return x[: 3 * n].reshape(n, 3)


def face_correspondences_by_proximity(
    src_V, src_F, tgt_V, tgt_F, threshold: float | None = None,
    normal_threshold: float = 0.0,
) -> np.ndarray:
    """Bidirectional centroid-proximity + normal-dot face correspondences
    (as in the vasiliskatr reference). Returns (k,2) array [src_face, tgt_face]."""
    from scipy.spatial import cKDTree

    def centroids(V, F):
        return (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

    def normals(V, F):
        fn = face_normals(V, F)
        return fn / np.linalg.norm(fn, axis=1, keepdims=True)

    sc, tc = centroids(src_V, src_F), centroids(tgt_V, tgt_F)
    sn, tn = normals(src_V, src_F), normals(tgt_V, tgt_F)
    if threshold is None:
        threshold = 2.0 * max(np.ptp(src_V), np.ptp(tgt_V)) / 100.0
    out = []
    tree = cKDTree(sc)
    d, idx = tree.query(tc, k=1)
    for t in range(len(tc)):
        if d[t] < threshold and np.dot(tn[t], sn[idx[t]]) > normal_threshold:
            out.append((int(idx[t]), t))
    tree = cKDTree(tc)
    d, idx = tree.query(sc, k=1)
    for s in range(len(sc)):
        if d[s] < threshold and np.dot(sn[s], tn[idx[s]]) > normal_threshold:
            out.append((s, int(idx[s])))
    return np.array(out, dtype=int).reshape(-1, 2)


# ---------------------------------------------------------------------------
# Delta Mush
# ---------------------------------------------------------------------------

def delta_mush(
    rest_V: np.ndarray,
    F: np.ndarray,
    def_V: np.ndarray,
    iterations: int = 4,
    alpha: float = 1.0,
    S=None,
) -> np.ndarray:
    """Detail-preserving smoothing (Delta Mush, canonical formulation).

    Detail vectors are computed ONCE from the rest pose (rest - smooth(rest),
    rest topology weights) and added back smoothed onto the smoothed deformed
    pose at every iteration:

        X <- S@X + alpha * S@delta,   delta = rest - S@rest

    This keeps the transferred-expression detail while blending into adjacent
    non-template regions (paper Sec. 3.6.4).
    S: optional prebuilt smoothing matrix (smoothing_matrix(F, n)) to avoid
    rebuilding it per call on large meshes.
    """
    if S is None:
        S = smoothing_matrix(F, len(rest_V), normalize=True)
    delta = np.asarray(rest_V, dtype=float) - S @ np.asarray(rest_V, dtype=float)
    X = np.asarray(def_V, dtype=float).copy()
    for _ in range(iterations):
        X = S @ X + alpha * (S @ delta)
    return X


# ---------------------------------------------------------------------------
# SDF helpers (trimesh) - inner-mouth collision refinement (paper §3.6.2)
# ---------------------------------------------------------------------------

def signed_distance_to_mesh(points: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> np.ndarray:
    """Signed distance of points to a mesh (trimesh; negative = inside)."""
    import trimesh

    m = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
    return m.proximity.signed_distance(points)


def push_out_of_mesh(
    points: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    margin: float = 1e-3,
    steps: int = 10,
) -> np.ndarray:
    """Push points out of a mesh along the SDF gradient until outside by
    `margin` (paper: teeth convex hull SDF vs face vertices)."""
    import trimesh

    m = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
    pts = np.asarray(points, dtype=float).copy()
    for _ in range(steps):
        sd = m.proximity.signed_distance(pts)
        inside = sd < margin
        if not inside.any():
            break
        # finite-difference gradient of the SDF
        eps = 1e-4
        grad = np.zeros_like(pts)
        for axis in range(3):
            d = np.zeros_like(pts)
            d[:, axis] = eps
            grad[:, axis] = (m.proximity.signed_distance(pts + d) -
                             m.proximity.signed_distance(pts - d)) / (2 * eps)
        nrm = np.linalg.norm(grad, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        pts[inside] += grad[inside] / nrm[inside] * (margin - sd[inside, None])
    return pts
