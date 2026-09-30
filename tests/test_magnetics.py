import numpy as np
import pytest
from scipy import integrate, special

from loophole.constants import MU0
from loophole.magnetics import CylinderMagnet, cel, cylinder_axis_field


def _loop_field(radius, z0, current, rho, z):
    """Exact field of a circular current loop (textbook elliptic-integral form)."""
    dz = z - z0
    a2 = (radius + rho) ** 2 + dz**2
    m = 4 * radius * rho / a2
    K, E = special.ellipk(m), special.ellipe(m)
    den = (radius - rho) ** 2 + dz**2
    bz = MU0 * current / (2 * np.pi * np.sqrt(a2)) * (K + (radius**2 - rho**2 - dz**2) / den * E)
    safe = np.where(rho > 0, rho, 1.0)
    br = np.where(rho > 0, MU0 * current / (2 * np.pi) * dz / (safe * np.sqrt(a2))
                  * (-K + (radius**2 + rho**2 + dz**2) / den * E), 0.0)
    return br, bz


def test_cel_matches_quadrature():
    rng = np.random.default_rng(1)
    for _ in range(100):
        kc, c, s = rng.uniform(0.02, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)
        p = 1.0 if rng.random() < 0.5 else rng.uniform(0.0, 1.0)
        f = lambda t: (c * np.cos(t) ** 2 + s * np.sin(t) ** 2) / (
            (np.cos(t) ** 2 + p * np.sin(t) ** 2) * np.sqrt(np.cos(t) ** 2 + kc**2 * np.sin(t) ** 2))
        ref = integrate.quad(f, 0, np.pi / 2, epsabs=1e-13, limit=200)[0]
        assert float(cel(kc, p, c, s)) == pytest.approx(ref, rel=1e-9, abs=1e-12)


def test_cel_complete_elliptic_K():
    # cel(kc, 1, 1, 1) = K(k) with k^2 = 1 - kc^2
    kc = np.linspace(0.05, 1.0, 20)
    assert np.allclose(cel(kc, 1.0, 1.0, 1.0), special.ellipk(1 - kc**2), rtol=1e-12)


def test_cylinder_matches_biot_savart():
    mag = CylinderMagnet(radius=3e-3, length=4e-3, br=1.4)
    k_sheet = mag.br / MU0
    xg, wg = np.polynomial.legendre.leggauss(400)
    zs, ws = xg * mag.length / 2, wg * mag.length / 2
    rho = np.array([0.0, 1e-3, 2.9e-3, 3.1e-3, 5e-3, 1e-2, 4e-3])
    z = np.array([2.5e-3, 2.3e-3, 2.4e-3, 0.0, 3e-3, -7e-3, 2.1e-3])
    br_ref, bz_ref = np.zeros_like(rho), np.zeros_like(rho)
    for zi, wi in zip(zs, ws):
        a, b = _loop_field(mag.radius, zi, k_sheet * wi, rho, z)
        br_ref += a
        bz_ref += b
    br, bz = mag.field_local(rho, z)
    assert np.allclose(br, br_ref, atol=2e-6)
    assert np.allclose(bz, bz_ref, atol=2e-6)


def test_on_axis_formula():
    z = np.linspace(2.1e-3, 5e-2, 25)
    mag = CylinderMagnet(3e-3, 4e-3, 1.4)
    assert np.allclose(mag.field_local(np.zeros_like(z), z)[1],
                       cylinder_axis_field(3e-3, 4e-3, 1.4, z), atol=1e-12)


def test_far_field_is_dipole():
    mag = CylinderMagnet(3e-3, 4e-3, 1.4)
    r, th = 0.3, 0.7
    p = np.array([r * np.sin(th), 0.0, r * np.cos(th)])
    B = mag.field(p[None, :])[0]
    m, rh = mag.moment, p / r
    Bd = MU0 / (4 * np.pi) * (3 * rh * m * rh[2] - np.array([0, 0, m])) / r**3
    assert np.linalg.norm(B - Bd) / np.linalg.norm(Bd) < 1e-4


def test_field_is_curl_and_divergence_free():
    mag = CylinderMagnet(5e-3, 3e-3, 1.3, inner_radius=2e-3)
    h = 1e-6
    for p in ([1e-3, 2e-3, 4e-3], [7e-3, -1e-3, 0.5e-3], [0.0, 3e-3, -6e-3]):
        p = np.array(p)
        J = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = h
            J[:, j] = (mag.field((p + e)[None])[0] - mag.field((p - e)[None])[0]) / (2 * h)
        scale = np.abs(J).max()
        assert abs(np.trace(J)) < 1e-6 * scale                 # div B = 0
        assert np.allclose(J, J.T, atol=1e-6 * scale)          # curl B = 0 outside sources


def test_rotation_consistency():
    mag = CylinderMagnet(2e-3, 3e-3, 1.2)
    # Rotating the magnet by 90 deg about y must match evaluating a rotated point.
    R = np.array([[0.0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    p = np.array([[4e-3, 1e-3, 2e-3]])
    B_rot = mag.field(p, rotation=R)[0]
    B_ref = R @ mag.field((R.T @ p[0])[None])[0]
    assert np.allclose(B_rot, B_ref, atol=1e-12)


def test_ring_is_difference_of_cylinders():
    ring = CylinderMagnet(10e-3, 5e-3, 1.3, inner_radius=4e-3)
    outer, inner = CylinderMagnet(10e-3, 5e-3, 1.3), CylinderMagnet(4e-3, 5e-3, 1.3)
    p = np.array([[3e-3, 2e-3, 8e-3]])
    assert np.allclose(ring.field(p), outer.field(p) - inner.field(p))
    assert ring.moment == pytest.approx(outer.moment - inner.moment)
