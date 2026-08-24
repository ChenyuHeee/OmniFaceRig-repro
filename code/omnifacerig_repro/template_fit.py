"""Stage 1: face template registration (paper Sec. 3.5.3).

Two-stage optimization, implemented exactly as in the paper:

  Stage 1 (Eq. 1): global rigid alignment
      E_rigid = sum_{i in K} || s R v_i + t - p_i ||^2          (Umeyama)

  Stage 2 (Eq. 2): per-vertex non-rigid deformation
      E(D) = l1 E_corr + l2 E_smooth + l3 E_edge + l4 E_tri + l5 E_flip + l6 E_reg

    E_corr   = sum_i H(||v'_i - p_i||)            Huber loss       (Eq. 3)
    E_smooth = sum_{(i,j) in E} ||d_i - d_j||^2                   (Eq. 4)
    E_edge   = sum_{(i,j) in E} (||v'_i-v'_j|| - ||v_i-v_j||)^2   (Eq. 5)
    E_tri    = sum_t ||G_t - G'_t||_F^2           edge matrices    (Eq. 6)
    E_flip   = sum_t max(0, -n_t . n'_t)^3        unnormalized    (Eq. 7)
    E_reg    = sum_i ||d_i||^2                                    (Eq. 8)

The energy and its analytic gradient are computed vectorized with numpy and
minimized with L-BFGS-B (scipy). Works on CPU for template meshes up to a few
thousand vertices.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# Default regularization weights (initial values; the paper does not publish
# its weights, these are tuned for unit-normalized meshes).
DEFAULT_LAMBDAS = {
    "corr": 1.0,    # l1 data term
    "smooth": 0.5,  # l2 displacement smoothness
    "edge": 1.0,    # l3 edge-length preservation
    "tri": 1.0,     # l4 triangle shape preservation
    "flip": 5.0,    # l5 normal-flip penalty
    "reg": 0.01,    # l6 offset regularization
}
DEFAULT_HUBER_DELTA = 0.05


# ---------------------------------------------------------------------------
# Stage 1: rigid alignment (Eq. 1) -- Umeyama/Kabsch with scale
# ---------------------------------------------------------------------------

def rigid_align(src: np.ndarray, tgt: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Best rigid+uniform-scale transform mapping src to tgt:

        min_{s,R,t} sum || s R src_i + t - tgt_i ||^2        (paper Eq. 1)

    Returns (s, R, t) with R in SO(3), s > 0, t (3,).
    """
    src = np.asarray(src, dtype=float)
    tgt = np.asarray(tgt, dtype=float)
    assert src.shape == tgt.shape and src.shape[1] == 3
    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    X = src - mu_s
    Y = tgt - mu_t
    cov = X.T @ Y
    U, S, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) if np.linalg.det(Vt.T @ U.T) != 0 else 1.0
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    var_s = np.sum(X ** 2)
    s = (np.trace(np.diag(S) @ D) / var_s) if var_s > 0 else 1.0
    t = mu_t - s * R @ mu_s
    return float(s), R, t


def apply_rigid(V: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * (V @ R.T) + t


# ---------------------------------------------------------------------------
# Stage 2: non-rigid deformation (Eq. 2)
# ---------------------------------------------------------------------------

def _edge_index(F: np.ndarray) -> np.ndarray:
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def _huber(x: np.ndarray, delta: float) -> np.ndarray:
    """Huber penalty H(x) (Eq. 3): 0.5 x^2 for |x|<=delta else delta(|x|-delta/2)."""
    x = np.abs(x)
    out = np.where(x <= delta, 0.5 * x ** 2, delta * (x - 0.5 * delta))
    return out


def _huber_deriv(x: np.ndarray, delta: float) -> np.ndarray:
    x = np.abs(x)
    return np.where(x <= delta, x, delta)


def fit_energy_and_grad(
    d: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    corr_idx: np.ndarray,
    corr_pos: np.ndarray,
    lambdas: dict[str, float] | None = None,
    huber_delta: float = DEFAULT_HUBER_DELTA,
) -> tuple[float, np.ndarray]:
    """Value and analytic gradient of E(D) (Eq. 2), vectorized.

    d:        (n,3) per-vertex displacements (flat or (n,3))
    corr_idx: (k,) template vertex indices with target keypoints
    corr_pos: (k,3) target 3D keypoint positions
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = len(V)
    d = np.asarray(d, dtype=float).reshape(n, 3)
    lm = {**DEFAULT_LAMBDAS, **(lambdas or {})}
    E_idx = _edge_index(F)
    e0, e1 = E_idx[:, 0], E_idx[:, 1]

    Vp = V + d
    grad = np.zeros_like(d)
    energy = 0.0

    # ---- E_corr (Eq. 3): Huber on keypoint residuals ----
    if len(corr_idx):
        r = Vp[corr_idx] - corr_pos
        x = np.linalg.norm(r, axis=1)
        energy += lm["corr"] * float(np.sum(_huber(x, huber_delta)))
        w = _huber_deriv(x, huber_delta) / np.maximum(x, 1e-12)
        np.add.at(grad, corr_idx, lm["corr"] * w[:, None] * r)

    # ---- E_smooth (Eq. 4) ----
    dd = d[e0] - d[e1]
    energy += lm["smooth"] * float(np.sum(dd ** 2))
    g = 2.0 * dd
    np.add.at(grad, e0, lm["smooth"] * g)
    np.add.at(grad, e1, -lm["smooth"] * g)

    # ---- E_edge (Eq. 5) ----
    ev = Vp[e0] - Vp[e1]
    ln = np.linalg.norm(ev, axis=1)
    rest_l = np.linalg.norm(V[e0] - V[e1], axis=1)
    diff = ln - rest_l
    energy += lm["edge"] * float(np.sum(diff ** 2))
    g = 2.0 * diff[:, None] * ev / np.maximum(ln[:, None], 1e-12)
    np.add.at(grad, e0, lm["edge"] * g)
    np.add.at(grad, e1, -lm["edge"] * g)

    # ---- E_tri (Eq. 6): 2x3 edge matrices ----
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    e1r = V[b] - V[a]
    e2r = V[c] - V[a]
    e1p = Vp[b] - Vp[a]
    e2p = Vp[c] - Vp[a]
    D1 = e1p - e1r
    D2 = e2p - e2r
    energy += lm["tri"] * float(np.sum(D1 ** 2) + np.sum(D2 ** 2))
    ga = -2.0 * (D1 + D2)
    gb = 2.0 * D1
    gc = 2.0 * D2
    np.add.at(grad, a, lm["tri"] * ga)
    np.add.at(grad, b, lm["tri"] * gb)
    np.add.at(grad, c, lm["tri"] * gc)

    # ---- E_flip (Eq. 7): max(0, -n . n')^3, unnormalized normals ----
    nr = np.cross(e1r, e2r)
    np_ = np.cross(e1p, e2p)
    dp = np.einsum("ij,ij->i", nr, np_)  # n_t . n'_t
    active = dp < 0
    if active.any():
        energy += lm["flip"] * float(np.sum((-dp[active]) ** 3))
        f = 3.0 * (-dp[active]) ** 2
        nra = nr[active]
        e1pa, e2pa = e1p[active], e2p[active]
        ga = f[:, None] * np.cross(nra, e2pa - e1pa)
        gb = f[:, None] * np.cross(e2pa, nra)
        gc = f[:, None] * (-np.cross(e1pa, nra))
        np.add.at(grad, a[active], lm["flip"] * ga)
        np.add.at(grad, b[active], lm["flip"] * gb)
        np.add.at(grad, c[active], lm["flip"] * gc)

    # ---- E_reg (Eq. 8) ----
    energy += lm["reg"] * float(np.sum(d ** 2))
    grad += 2.0 * lm["reg"] * d

    return float(energy), grad.ravel()


def nonrigid_fit(
    V: np.ndarray,
    F: np.ndarray,
    corr_idx: np.ndarray,
    corr_pos: np.ndarray,
    lambdas: dict[str, float] | None = None,
    huber_delta: float = DEFAULT_HUBER_DELTA,
    max_iter: int = 400,
    ftol: float = 1e-12,
    x0: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Minimize Eq. 2 (L-BFGS-B) starting from x0 (default: zeros).

    Returns (deformed vertices (n,3), info dict).
    """
    V = np.asarray(V, dtype=float)
    F = np.asarray(F, dtype=int)
    n = len(V)
    corr_idx = np.asarray(corr_idx, dtype=int)
    corr_pos = np.asarray(corr_pos, dtype=float)
    if lambdas is None:
        lambdas = {}
    if "corr" not in lambdas:
        # E_corr sums over K keypoints while E_edge/E_tri sum over all
        # edges/triangles; scale the data weight accordingly so the fit is
        # not drowned by the regularizers (tuned for unit-normalized meshes).
        n_edges = len(_edge_index(F))
        lambdas = {**lambdas, "corr": 10.0 * max(1.0, n_edges / max(1, len(corr_idx)))}
    if x0 is None:
        x0 = np.zeros(3 * n)

    def fun(d):
        return fit_energy_and_grad(d, V, F, corr_idx, corr_pos, lambdas, huber_delta)

    res = minimize(fun, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": max_iter, "ftol": ftol, "maxls": 40})
    d = res.x.reshape(n, 3)
    info = {
        "success": bool(res.success),
        "nit": int(getattr(res, "nit", -1)),
        "fun": float(res.fun),
        "message": str(res.message),
    }
    return V + d, info


def fit_template(
    V: np.ndarray,
    F: np.ndarray,
    keypoint_idx: np.ndarray,
    keypoint_pos: np.ndarray,
    lambdas: dict[str, float] | None = None,
    huber_delta: float = DEFAULT_HUBER_DELTA,
) -> tuple[np.ndarray, dict]:
    """Full Stage 1: rigid (Eq. 1) then non-rigid (Eq. 2) fitting.

    Returns (fitted vertices, info) with info['rigid'] = (s, R, t).
    """
    keypoint_idx = np.asarray(keypoint_idx, dtype=int)
    keypoint_pos = np.asarray(keypoint_pos, dtype=float)
    s, R, t = rigid_align(V[keypoint_idx], keypoint_pos)
    V_rigid = apply_rigid(V, s, R, t)
    V_fit, info = nonrigid_fit(V_rigid, F, keypoint_idx, keypoint_pos,
                               lambdas=lambdas, huber_delta=huber_delta)
    info["rigid"] = (s, R, t)
    return V_fit, info
