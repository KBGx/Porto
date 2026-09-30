"""Exact magnetostatics of uniformly magnetised cylinders.

A uniformly magnetised cylinder (remanence Br along its axis) has the same
external field as an ideal finite solenoid with sheet current K = Br / mu0.
Its off-axis field has a closed form in terms of Bulirsch's generalised
complete elliptic integral (Derby & Olbert, Am. J. Phys. 78, 229 (2010)):

    B_rho = (Br/pi) [a+ C(k+, 1, 1, -1) - a- C(k-, 1, 1, -1)]
    B_z   = (Br/pi) R/(R+rho) [b+ C(k+, g^2, 1, g) - b- C(k-, g^2, 1, g)]

with z+- = z +- h/2 (z measured from the magnet's centre; the two terms are the
bottom and top end faces),
a+- = R / sqrt(z+-^2 + (rho+R)^2), b+- = z+- / sqrt(z+-^2 + (rho+R)^2),
g = (R - rho)/(R + rho), k+- = sqrt((z+-^2 + (R-rho)^2) / (z+-^2 + (R+rho)^2)).

Everything here is vectorised with numpy and validated in tests/ against
the textbook on-axis formula, direct Biot-Savart integration and the
far-field dipole limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import MU0


def cel(kc, p, c, s, tol: float = 1e-12, max_iter: int = 60):
    """Bulirsch's generalised complete elliptic integral, vectorised.

        cel(kc,p,c,s) = int_0^{pi/2} (c cos^2 t + s sin^2 t) /
                        ((cos^2 t + p sin^2 t) sqrt(cos^2 t + kc^2 sin^2 t)) dt

    Port of the algorithm printed in Derby & Olbert (2010), extended to
    arrays: converged elements are frozen while the rest keep iterating.
    """
    kc, p, c, s = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (kc, p, c, s)))
    k = np.abs(kc)
    em = np.ones_like(k)

    pos = p > 0
    # p > 0 branch
    pp = np.where(pos, np.sqrt(np.where(pos, p, 1.0)), 0.0)
    ss = np.where(pos, s / np.where(pos, pp, 1.0), 0.0)
    cc = c.copy()
    # p <= 0 branch
    if not np.all(pos):
        f = kc * kc
        q = 1.0 - f
        g = 1.0 - p
        f = f - p
        q = q * (s - c * p)
        pp_n = np.sqrt(f / g)
        cc_n = (c - s) / g
        ss_n = -q / (g * g * pp_n) + cc_n * pp_n
        pp = np.where(pos, pp, pp_n)
        cc = np.where(pos, cc, cc_n)
        ss = np.where(pos, ss, ss_n)

    f = cc
    cc = cc + ss / pp
    g = k / pp
    ss = 2 * (ss + f * g)
    pp = g + pp
    g = em
    em = k + em
    kk = k

    active = np.abs(g - k) > g * tol
    for _ in range(max_iter):
        if not active.any():
            break
        k_n = 2 * np.sqrt(kk)
        kk_n = k_n * em
        f = cc
        cc_n = cc + ss / pp
        g_n = kk_n / pp
        ss_n = 2 * (ss + f * g_n)
        pp_n = g_n + pp
        g_old = em
        em_n = k_n + em
        k = np.where(active, k_n, k)
        kk = np.where(active, kk_n, kk)
        cc = np.where(active, cc_n, cc)
        ss = np.where(active, ss_n, ss)
        pp = np.where(active, pp_n, pp)
        g = np.where(active, g_old, g)
        em = np.where(active, em_n, em)
        active = active & (np.abs(g - k) > g * tol)
    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))


def cylinder_field_local(radius: float, length: float, br: float, rho, z):
    """Field (B_rho, B_z) [T] of a cylinder magnet in its own frame.

    The magnet is centred at the origin with magnetisation along +z. ``rho``
    and ``z`` are arrays of cylindrical coordinates [m]. Valid everywhere
    except exactly on the curved surface's edges.
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    R = radius
    b = length / 2
    B0 = br / np.pi
    out_rho = np.zeros(np.broadcast(rho, z).shape)
    out_z = np.zeros_like(out_rho)
    gamma = (R - rho) / (R + rho)
    for sign, zz in ((+1.0, z + b), (-1.0, z - b)):
        # z + b: distance above the bottom face; z - b: above the top face.
        denom = np.sqrt(zz**2 + (rho + R) ** 2)
        alpha = R / denom
        beta = zz / denom
        kcs = np.sqrt((zz**2 + (R - rho) ** 2) / (zz**2 + (R + rho) ** 2))
        out_rho += sign * alpha * cel(kcs, 1.0, 1.0, -1.0)
        out_z += sign * beta * cel(kcs, gamma**2, 1.0, gamma)
    return B0 * out_rho, B0 * R / (R + rho) * out_z


def cylinder_axis_field(radius: float, length: float, br: float, z):
    """Textbook on-axis field B_z(z) of a cylinder centred at the origin."""
    z = np.asarray(z, dtype=float)
    b = length / 2
    R = radius
    return br / 2 * ((z + b) / np.sqrt((z + b) ** 2 + R**2) - (z - b) / np.sqrt((z - b) ** 2 + R**2))


@dataclass(frozen=True)
class CylinderMagnet:
    """Axially magnetised cylinder (or ring, via ``inner_radius``) magnet.

    A ring is modelled as an outer cylinder plus an inner cylinder of opposite
    magnetisation, which is exact for uniform magnetisation.
    """

    radius: float
    length: float
    br: float
    inner_radius: float = 0.0

    @property
    def volume(self) -> float:
        return np.pi * (self.radius**2 - self.inner_radius**2) * self.length

    @property
    def moment(self) -> float:
        """Magnetic dipole moment [A m^2]."""
        return self.br * self.volume / MU0

    def field_local(self, rho, z):
        br, bz = cylinder_field_local(self.radius, self.length, self.br, rho, z)
        if self.inner_radius > 0:
            br_i, bz_i = cylinder_field_local(self.inner_radius, self.length, self.br, rho, z)
            br, bz = br - br_i, bz - bz_i
        return br, bz

    def field(self, points, center=(0.0, 0.0, 0.0), rotation=None):
        """Cartesian field [T] at ``points`` (..., 3) in the world frame.

        ``rotation`` maps body to world coordinates (3x3); the magnetisation
        is along the body +z axis.
        """
        pts = np.asarray(points, dtype=float) - np.asarray(center, dtype=float)
        if rotation is not None:
            pts = pts @ np.asarray(rotation)          # world -> body: R^T x, row-vector form
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        rho = np.hypot(x, y)
        safe = np.where(rho > 0, rho, 1.0)
        brho, bz = self.field_local(rho, z)
        bx = np.where(rho > 0, brho * x / safe, 0.0)
        by = np.where(rho > 0, brho * y / safe, 0.0)
        B = np.stack([bx, by, bz], axis=-1)
        if rotation is not None:
            B = B @ np.asarray(rotation).T            # body -> world
        return B

    def axis_field(self, z):
        bz = cylinder_axis_field(self.radius, self.length, self.br, z)
        if self.inner_radius > 0:
            bz = bz - cylinder_axis_field(self.inner_radius, self.length, self.br, z)
        return bz
