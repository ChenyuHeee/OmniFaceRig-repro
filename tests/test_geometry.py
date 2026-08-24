"""Tests for the geometry primitives (Laplacian / ARAP / DT / Delta Mush)."""

import numpy as np
import pytest

from omnifacerig_repro.geometry import (
    cotangent_laplacian, smoothing_matrix, arap_deform,
    deformation_transfer, delta_mush, face_normals,
    face_correspondences_by_proximity,
)
from omnifacerig_repro.pipeline import ellipsoid_mesh


@pytest.fixture(scope="module")
def mesh():
    return ellipsoid_mesh(1.0, 1.2, 1.0, n_lat=10, n_lon=20)


def test_cotangent_laplacian(mesh):
    V, F = mesh
    L = cotangent_laplacian(V, F)
    assert abs(L - L.T).max() < 1e-10          # symmetric
    assert np.abs(np.asarray(L.sum(axis=1)).ravel()).max() < 1e-10  # zero row sums


def test_smoothing_matrix_row_stochastic(mesh):
    V, F = mesh
    S = smoothing_matrix(F, len(V))
    assert np.allclose(np.asarray(S.sum(axis=1)).ravel(), 1.0)
    assert S.shape == (len(V), len(V))


def test_arap_pins_handles(mesh):
    V, F = mesh
    handles = np.array([0, 17, 88, 150], dtype=int)
    target = V[handles] + np.array([0.2, -0.1, 0.05])
    out = arap_deform(V, F, handles, target, iterations=30)
    assert np.allclose(out[handles], target, atol=1e-6)
    # non-handle vertices should have moved (rigidity preserved but deformed)
    assert np.linalg.norm(out - V) > 1e-3


def test_deformation_transfer_identity(mesh):
    V, F = mesh
    out = deformation_transfer(V, V, F, V)
    assert np.allclose(out, V, atol=1e-4)


def test_deformation_transfer_rigid(mesh):
    """A similarity transform on the source must be recovered by the target
    (up to translation, which the transfer cannot determine)."""
    V, F = mesh
    rng = np.random.default_rng(0)
    angle = 0.3
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1.0]])
    s = 1.4
    src_def = s * (V @ R.T)
    # exact transfer: no identity/smoothness regularization; tiny rest noise
    tgt_def = deformation_transfer(V, src_def, F, V + rng.normal(0, 1e-4, V.shape),
                                   wi=0.0, ws=0.0)
    # compare shapes up to translation: subtract centroids
    a = tgt_def - tgt_def.mean(axis=0)
    b = s * (V @ R.T)
    b = b - b.mean(axis=0)
    assert np.allclose(a, b, atol=1e-3)


def test_delta_mush_smooths_noise(mesh):
    V, F = mesh
    rng = np.random.default_rng(1)
    noisy = V + rng.normal(0, 0.05, V.shape)
    S = smoothing_matrix(F, len(V))
    rough = lambda X: np.linalg.norm(X - S @ X)  # high-frequency content
    assert rough(delta_mush(V, F, noisy, iterations=4)) < rough(noisy) * 0.7


def test_delta_mush_preserves_shape(mesh):
    """DM smooths but keeps the mesh close to the input (detail preserved)."""
    V, F = mesh
    rng = np.random.default_rng(1)
    noisy = V + rng.normal(0, 0.05, V.shape)
    out = delta_mush(V, F, noisy, iterations=4)
    assert np.abs(out - noisy).max() < 0.25


def test_face_correspondences(mesh):
    V, F = mesh
    corr = face_correspondences_by_proximity(V, F, V + 0.01, F)
    assert len(corr) > 0
    assert corr.shape[1] == 2


def test_face_normals_no_degenerate(mesh):
    V, F = mesh
    n = face_normals(V, F)
    assert np.all(np.linalg.norm(n, axis=1) > 1e-8)
