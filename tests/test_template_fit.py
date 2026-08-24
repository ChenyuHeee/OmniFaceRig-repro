"""Tests for Stage 1 template fitting (paper Eq. 1 / Eq. 2)."""

import numpy as np
import pytest

from omnifacerig_repro.template_fit import (
    rigid_align, apply_rigid, fit_energy_and_grad, nonrigid_fit, fit_template,
)
from omnifacerig_repro.pipeline import ellipsoid_mesh, head_keypoints
from omnifacerig_repro.geometry import face_normals


def test_rigid_align_recovers_transform():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    s, R, t = 1.7, np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.array([0.5, -2.0, 1.0])
    tgt = s * (src @ R.T) + t
    s2, R2, t2 = rigid_align(src, tgt)
    assert np.allclose(s2, s, atol=1e-10)
    assert np.allclose(R2, R, atol=1e-10)
    assert np.allclose(t2, t, atol=1e-10)
    assert np.allclose(apply_rigid(src, s2, R2, t2), tgt, atol=1e-10)


def test_energy_gradient_finite_difference():
    rng = np.random.default_rng(3)
    V, F = ellipsoid_mesh(1.0, 1.2, 1.0, n_lat=6, n_lon=12)
    corr_idx = np.array([0, 5, 12, 33, 40])
    corr_pos = V[corr_idx] + rng.normal(0, 0.1, (5, 3))
    d = rng.normal(0, 0.02, (len(V), 3))
    E, g = fit_energy_and_grad(d, V, F, corr_idx, corr_pos)
    eps = 1e-6
    g_num = np.zeros_like(g)
    dflat = d.ravel()
    for i in range(0, len(dflat), 7):  # sample a subset of coordinates
        dflat[i] += eps
        Ep, _ = fit_energy_and_grad(dflat, V, F, corr_idx, corr_pos)
        dflat[i] -= 2 * eps
        Em, _ = fit_energy_and_grad(dflat, V, F, corr_idx, corr_pos)
        dflat[i] += eps
        g_num[i] = (Ep - Em) / (2 * eps)
    mask = np.zeros(len(g), dtype=bool)
    mask[::7] = True
    assert np.allclose(g[mask], g_num[mask], atol=1e-4)


def test_nonrigid_fit_converges_no_flips():
    V, F = ellipsoid_mesh(1.0, 1.2, 1.0, n_lat=10, n_lon=20)
    key_idx, key_pos = head_keypoints(V, 1.0, 1.2, 1.0)
    target = key_pos + np.array([0.0, -0.05, 0.03])  # slight displacement
    Vfit, info = nonrigid_fit(V, F, key_idx, target)
    assert info["success"]
    err = np.linalg.norm(Vfit[key_idx] - target, axis=1).mean()
    assert err < 0.02
    # no triangle flips after fitting
    n0 = face_normals(V, F)
    n1 = face_normals(Vfit, F)
    assert (np.einsum("ij,ij->i", n0, n1) > 0).all()


def test_fit_template_end_to_end():
    V, F = ellipsoid_mesh(1.0, 1.15, 1.0, n_lat=10, n_lon=20)
    key_idx, key_pos = head_keypoints(V, 1.0, 1.15, 1.0)
    rng = np.random.default_rng(5)
    noisy = key_pos + rng.normal(0, 0.02, key_pos.shape)
    Vfit, info = fit_template(V, F, key_idx, noisy)
    err = np.linalg.norm(Vfit[key_idx] - noisy, axis=1).mean()
    assert err < 0.03
    assert "rigid" in info
    s, R, t = info["rigid"]
    assert s > 0
