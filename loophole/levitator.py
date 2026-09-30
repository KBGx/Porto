"""Diamagnetically stabilised magnet levitation: the device model.

Geometry (world frame, z up, floater's nominal centre at the origin):

            ┌──────────────┐   lifter magnet (fixed), magnetised +z,
            │    LIFTER    │   centre at z = lifter_height
            └──────────────┘
                  ...           free space
    ═══════════════════════════ upper pyrolytic-graphite plate
                  gap
               ┌──────┐         floater magnet (free, 5 DOF), magnetised +z,
               │FLOATR│         carries the payload
               └──────┘
                  gap
    ═══════════════════════════ lower pyrolytic-graphite plate

The lifter pulls the floater up with exactly its weight. By Earnshaw's
theorem that equilibrium is unstable in at least one direction: with the
field curvature chosen to hold it laterally, it is unstable vertically.
The diamagnetic plates add the missing vertical stiffness: a magnet
approaching either plate induces an opposing magnetisation and is pushed
back. No power, no cryogenics, no moving parts.

Physics, all from first principles:

* magnet-magnet energy   U_m = -(Br_f/mu0) * integral over the floater of n.B_lifter dV
  (rigid, uniformly magnetised floater; exact volume integral, not a dipole approx.)
* diamagnetic energy     U_d = -(1/2mu0) * integral over the plates of B.chi.B dV,
  B = B_floater + B_lifter, chi = diag(chi_par, chi_par, chi_perp) for pyrolytic
  graphite with its c-axis normal to the plate (first order in chi ~ 1e-4)
* gravity                U_g = m_total g z

Stability is judged from the full 5x5 Hessian in (x, y, z, tilt_x, tilt_y):
all eigenvalues positive means a true energy minimum, i.e. stable levitation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from scipy import linalg, optimize

from .constants import CHI_PG_PAR, CHI_PG_PERP, G_EARTH, MU0, RHO_NDFEB
from .magnetics import CylinderMagnet

DOF_NAMES = ("x", "y", "z", "tilt_x", "tilt_y")


def rotation(tilt_x: float, tilt_y: float) -> np.ndarray:
    """Body-to-world rotation: tilt about x, then about y."""
    cx, sx = math.cos(tilt_x), math.sin(tilt_x)
    cy, sy = math.cos(tilt_y), math.sin(tilt_y)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return ry @ rx


def _gl_panels(edges, n):
    """Composite Gauss-Legendre nodes/weights over consecutive panels."""
    x0, w0 = np.polynomial.legendre.leggauss(n)
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(0.5 * (b - a) * x0 + 0.5 * (a + b))
        ws.append(0.5 * (b - a) * w0)
    return np.concatenate(xs), np.concatenate(ws)


def _geometric_edges(start, stop, first, growth=1.5):
    edges = [start]
    step = first
    while edges[-1] + step < stop:
        edges.append(edges[-1] + step)
        step *= growth
    edges.append(stop)
    return edges


@dataclass
class Quadrature:
    points: np.ndarray    # (N, 3) world coordinates
    weights: np.ndarray   # (N,)


@dataclass
class Levitator:
    lifter: CylinderMagnet
    floater: CylinderMagnet
    lifter_height: float                 # z of lifter centre [m]; floater nominal centre at z=0
    gap: float                           # clearance floater face <-> plate face [m]
    plate_thickness: float = 2e-3
    plate_radius: float | None = None    # default: 6 floater radii
    chi_perp: float = CHI_PG_PERP
    chi_par: float = CHI_PG_PAR
    payload_mass: float = 0.0            # extra mass with CoM at the floater centre [kg]
    payload_inertia: float = 0.0         # its transverse moment of inertia [kg m^2]
    floater_density: float = RHO_NDFEB
    n_phi: int = 32                      # azimuthal nodes for the full 3D plate integral
    n_gl: int = 6                        # Gauss-Legendre nodes per panel

    # ----------------------------------------------------------------- geometry
    @property
    def plate_r(self) -> float:
        return self.plate_radius if self.plate_radius is not None else 6 * self.floater.radius

    @property
    def plate_face(self) -> float:
        """|z| of the plate faces nearest the floater."""
        return self.floater.length / 2 + self.gap

    @property
    def floater_mass(self) -> float:
        return self.floater_density * self.floater.volume

    @property
    def mass(self) -> float:
        return self.floater_mass + self.payload_mass

    @property
    def weight(self) -> float:
        return self.mass * G_EARTH

    @property
    def lifter_clearance(self) -> float:
        """Free space between the upper plate's top and the lifter's bottom face."""
        return (self.lifter_height - self.lifter.length / 2) - (self.plate_face + self.plate_thickness)

    def inertia_matrix(self) -> np.ndarray:
        a, L = self.floater.radius, self.floater.length
        i_t = self.floater_mass * (3 * a**2 + L**2) / 12 + self.payload_inertia
        m = self.mass
        return np.diag([m, m, m, i_t, i_t])

    # --------------------------------------------------------------- quadrature
    @cached_property
    def _floater_body_quad(self) -> Quadrature:
        """Volume quadrature of the floater in its body frame (for U_m)."""
        a, L = self.floater.radius, self.floater.length
        r, wr = _gl_panels([self.floater.inner_radius, a], 5)
        zz, wz = _gl_panels([-L / 2, L / 2], 5)
        nphi = 12
        phi = 2 * np.pi * np.arange(nphi) / nphi
        R_, P_, Z_ = np.meshgrid(r, phi, zz, indexing="ij")
        W_ = (wr * r)[:, None, None] * (2 * np.pi / nphi) * wz[None, None, :] * np.ones_like(R_)
        pts = np.stack([R_ * np.cos(P_), R_ * np.sin(P_), Z_], axis=-1).reshape(-1, 3)
        return Quadrature(pts, W_.reshape(-1))

    def _plate_quad(self, n_phi: int) -> Quadrature:
        """Quadrature over both plates; n_phi=1 gives the axisymmetric reduction."""
        a, gap, t, Rp = self.floater.radius, self.gap, self.plate_thickness, self.plate_r
        d = max(gap, 0.15 * a)
        lo, hi = max(0.0, a - 4 * d), min(Rp, a + 4 * d)
        edges = list(np.linspace(0.0, lo, max(2, int(math.ceil(lo / (1.5 * d))) + 1)))
        edges += list(np.linspace(lo, hi, 9))[1:]
        edges += _geometric_edges(hi, Rp, d)[1:]
        rho, w_rho = _gl_panels(sorted(set(edges)), self.n_gl)
        depth_edges = _geometric_edges(0.0, t, min(d / 2, t / 2), growth=1.8)
        dep, w_dep = _gl_panels(depth_edges, self.n_gl)
        phi = 2 * np.pi * np.arange(n_phi) / n_phi
        R_, P_, D_ = np.meshgrid(rho, phi, dep, indexing="ij")
        W_ = (w_rho * rho)[:, None, None] * (2 * np.pi / n_phi) * w_dep[None, None, :] * np.ones_like(R_)
        x, y = R_ * np.cos(P_), R_ * np.sin(P_)
        pts, wts = [], []
        for side in (-1.0, +1.0):
            z = side * (self.plate_face + D_)
            pts.append(np.stack([x, y, z], axis=-1).reshape(-1, 3))
            wts.append(W_.reshape(-1))
        return Quadrature(np.concatenate(pts), np.concatenate(wts))

    @cached_property
    def _plates_axial(self):
        q = self._plate_quad(1)
        return q, self._lifter_field(q.points)

    @cached_property
    def _plates_full(self):
        q = self._plate_quad(self.n_phi)
        return q, self._lifter_field(q.points)

    def _lifter_field(self, pts):
        return self.lifter.field(pts, center=(0.0, 0.0, self.lifter_height))

    # ------------------------------------------------------------------ energy
    def energy_parts(self, q, full: bool = True, plates: bool = True) -> dict[str, float]:
        """Potential energy terms [J] at pose q = (x, y, z, tilt_x, tilt_y)."""
        x, y, z, tx, ty = q
        Rm = rotation(tx, ty)
        c = np.array([x, y, z])

        u_grav = self.mass * G_EARTH * z

        fq = self._floater_body_quad
        pts = fq.points @ Rm.T + c
        B_L = self._lifter_field(pts)
        u_mag = -(self.floater.br / MU0) * np.sum(fq.weights * (B_L @ Rm[:, 2]))

        u_dia = 0.0
        if plates and (self.chi_perp != 0.0 or self.chi_par != 0.0):
            pq, BL_P = self._plates_full if full else self._plates_axial
            B_f = self.floater.field(pq.points, center=c, rotation=Rm)
            chi = np.array([self.chi_par, self.chi_par, self.chi_perp])
            integrand = np.sum(chi * (B_f * B_f + 2.0 * B_f * BL_P), axis=-1)
            u_dia = -np.sum(pq.weights * integrand) / (2 * MU0)
        return {"gravity": u_grav, "magnet": u_mag, "diamagnetic": u_dia}

    def energy(self, q, full: bool = True) -> float:
        return sum(self.energy_parts(q, full).values())

    def axial_energy(self, z: float) -> float:
        """Energy on the symmetry axis, untilted (cheap axisymmetric integral)."""
        return self.energy((0.0, 0.0, z, 0.0, 0.0), full=False)

    def axial_force(self, z: float, h: float = 1e-6) -> float:
        """Net upward force on the floater at height z on axis [N]."""
        return -(self.axial_energy(z + h) - self.axial_energy(z - h)) / (2 * h)

    # ------------------------------------------------------------- equilibrium
    def equilibria(self, n_scan: int = 161, margin: float = 0.03):
        """Stationary points of the axial energy strictly inside the gap.

        Returns a list of (z, kind) with kind 'min' or 'max'.
        """
        zmax = self.gap * (1 - margin)
        zs = np.linspace(-zmax, zmax, n_scan)
        f = np.array([self.axial_force(z) for z in zs])
        out = []
        for i in range(n_scan - 1):
            if f[i] == 0 or f[i] * f[i + 1] < 0:
                z0 = optimize.brentq(self.axial_force, zs[i], zs[i + 1], xtol=1e-10)
                kind = "min" if f[i] > 0 else "max"      # force up below, down above -> restoring
                out.append((z0, kind))
        return out

    def hessian(self, z0: float = 0.0, full: bool = True, h_lin: float | None = None,
                h_ang: float = 2e-3) -> np.ndarray:
        """5x5 Hessian of the energy at the on-axis pose (0, 0, z0, 0, 0).

        ``full=True`` integrates the plates in 3D (slow, rigorous).
        ``full=False`` is the fast screening model: the plates' vertical
        stiffness comes from the exact axisymmetric integral, and their
        lateral/tilt contributions (zero for infinite plates) are dropped.
        """
        h_lin = h_lin if h_lin is not None else min(2e-5, self.gap / 50)
        if not full:
            H = replace(self, chi_perp=0.0, chi_par=0.0).hessian(z0, True, h_lin, h_ang)
            dia = lambda z: self.energy_parts((0.0, 0.0, z, 0.0, 0.0), full=False)["diamagnetic"]
            H[2, 2] += (dia(z0 + h_lin) - 2 * dia(z0) + dia(z0 - h_lin)) / h_lin**2
            return H
        steps = np.array([h_lin, h_lin, h_lin, h_ang, h_ang])
        q0 = np.array([0.0, 0.0, z0, 0.0, 0.0])
        E = lambda q: self.energy(q, full=True)
        e0 = E(q0)
        H = np.zeros((5, 5))
        for i in range(5):
            ei = np.zeros(5); ei[i] = steps[i]
            H[i, i] = (E(q0 + ei) - 2 * e0 + E(q0 - ei)) / steps[i] ** 2
            for j in range(i + 1, 5):
                ej = np.zeros(5); ej[j] = steps[j]
                H[i, j] = H[j, i] = (E(q0 + ei + ej) - E(q0 + ei - ej)
                                     - E(q0 - ei + ej) + E(q0 - ei - ej)) / (4 * steps[i] * steps[j])
        return H

    def stability(self, z0: float = 0.0, full: bool = True) -> "Stability":
        H = self.hessian(z0, full=full)
        M = self.inertia_matrix()
        w2, modes = linalg.eigh(H, M)
        return Stability(z0=z0, hessian=H, omega2=w2, modes=modes, inertia=M)

    # ------------------------------------------------ lifter-field diagnostics
    def lifter_axis_derivatives(self, z: float = 0.0, h: float = 1e-5):
        """B, dB/dz, d2B/dz2 of the lifter field on axis at height z."""
        f = lambda zz: float(self.lifter.axis_field(zz - self.lifter_height))
        b0, bp, bm = f(z), f(z + h), f(z - h)
        return b0, (bp - bm) / (2 * h), (bp - 2 * b0 + bm) / h**2

    def point_dipole_estimates(self, z: float = 0.0) -> dict[str, float]:
        """Simon-Heflinger-Geim point-dipole stability functions (for cross-checks)."""
        M = self.floater.moment
        B, B1, B2 = self.lifter_axis_derivatives(z)
        k_lat = 0.5 * M * (B2 - B1**2 / (2 * B))
        return {"B": B, "dBdz": B1, "d2Bdz2": B2, "k_lateral_dipole": k_lat,
                "k_vertical_lifter_only": -M * B2,
                "radial_ok": B2 > B1**2 / (2 * B)}


@dataclass
class Stability:
    z0: float
    hessian: np.ndarray
    omega2: np.ndarray       # generalized eigenvalues of (H, M) [rad^2/s^2]
    modes: np.ndarray
    inertia: np.ndarray

    @property
    def stable(self) -> bool:
        return bool(np.all(self.omega2 > 0))

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Signed natural frequencies: negative values mark unstable modes (growth rate/2pi)."""
        return np.sign(self.omega2) * np.sqrt(np.abs(self.omega2)) / (2 * np.pi)

    def mode_labels(self) -> list[str]:
        labels = []
        w = np.sqrt(np.diag(self.inertia))       # kinetic-energy weighting
        for v in self.modes.T:
            i = int(np.argmax(np.abs(v * w)))
            labels.append(DOF_NAMES[i])
        return labels

    def summary(self) -> str:
        parts = [f"{lab}:{f:+.3f}Hz" for lab, f in zip(self.mode_labels(), self.frequencies_hz)]
        return ("STABLE " if self.stable else "UNSTABLE ") + " ".join(parts)


# --------------------------------------------------------------------- design
def balance(lev: Levitator, h_lo: float | None = None, h_hi: float = 1.0, n_scan: int = 400):
    """All lifter heights that put the floater in force balance at z = 0.

    Scans the lifter height (logarithmically) and returns the roots of the net
    axial force at the centre, nearest first. The magnetic gradient on axis is
    non-monotonic, so there are generally two: an 'upper' and a 'lower' point.
    """
    min_h = lev.plate_face + lev.plate_thickness + lev.lifter.length / 2 + 1e-4
    h_lo = max(h_lo if h_lo is not None else min_h, min_h)

    def f(h):
        return replace(lev, lifter_height=h).axial_force(0.0)

    hs = np.geomspace(h_lo, h_hi, n_scan)
    fs = np.array([f(h) for h in hs])
    roots = []
    for i in range(n_scan - 1):
        if fs[i] * fs[i + 1] < 0:
            roots.append(optimize.brentq(f, hs[i], hs[i + 1], xtol=1e-9))
    return roots


def balanced(lev: Levitator, which: str = "lower", **kw) -> Levitator:
    """Return a copy of ``lev`` with the lifter placed at a balance height."""
    roots = balance(lev, **kw)
    if not roots:
        raise ValueError("lifter too weak: no height balances the floater's weight")
    h = roots[-1] if which == "lower" else roots[0]
    return replace(lev, lifter_height=h)


# ------------------------------------------------------------------- dynamics
class FastEnergy:
    """5-DOF energy with the plates treated as laterally infinite.

    Equals ``lev.energy(q, full=True)`` up to the (verified negligible)
    lateral/tilt terms of the finite plates. The diamagnetic term then
    depends on z only and is tabulated once as a spline.
    """

    def __init__(self, lev: Levitator, n: int = 241):
        from scipy.interpolate import CubicSpline

        self.lev = lev
        zs = np.linspace(-lev.gap, lev.gap, n)
        dia = [lev.energy_parts((0.0, 0.0, z, 0.0, 0.0), full=False)["diamagnetic"] for z in zs]
        self._dia = CubicSpline(zs, dia)

    def __call__(self, q) -> float:
        p = self.lev.energy_parts(tuple(q), plates=False)
        return p["gravity"] + p["magnet"] + float(self._dia(q[2]))


def plate_clearance(lev: Levitator, q) -> np.ndarray:
    """Closest approach of the (tilted) floater's rim to either plate [m]; <= 0 is contact."""
    q = np.atleast_2d(q)
    tilt = np.max(np.abs(q[:, 3:5]), axis=1)
    reach = (np.abs(q[:, 2]) + lev.floater.radius * np.sin(tilt)
             + lev.floater.length / 2 * np.cos(tilt))
    return lev.plate_face - reach


def simulate(lev: Levitator, q0, v0=(0, 0, 0, 0, 0), t_end: float = 3.0, dt: float = 5e-4,
             damping: float = 0.0, record_every: int = 4):
    """Integrate the rigid floater's motion (velocity Verlet, 5 DOF).

    ``damping`` is a linear damping rate [1/s] applied to every DOF (a crude
    stand-in for eddy currents in the graphite); 0 gives the conservative,
    worst-case motion. Stops early if the floater touches a plate.
    Returns dict with t, q (n,5), energy (n,), touched (bool).
    """
    M = np.diag(lev.inertia_matrix())
    steps = np.array([2e-6, 2e-6, 2e-6, 1e-4, 1e-4])
    energy = FastEnergy(lev)

    def grad(q):
        g = np.empty(5)
        for i in range(5):
            e = np.zeros(5); e[i] = steps[i]
            g[i] = (energy(q + e) - energy(q - e)) / (2 * steps[i])
        return g

    q = np.array(q0, dtype=float)
    v = np.array(v0, dtype=float)
    a = -grad(q) / M
    ts, qs, es = [], [], []
    touched = False
    n = int(round(t_end / dt))
    for k in range(n + 1):
        if k % record_every == 0:
            ts.append(k * dt)
            qs.append(q.copy())
            es.append(energy(q) + 0.5 * np.sum(M * v * v))
        if plate_clearance(lev, q)[0] <= 0:
            touched = True
            break
        v_half = v + 0.5 * dt * a
        q = q + dt * v_half
        a = -grad(q) / M
        v = (v_half + 0.5 * dt * a) * (1 - damping * dt)
    return {"t": np.array(ts), "q": np.array(qs), "energy": np.array(es), "touched": touched}
