"""Design search: how much can a passive, room-temperature levitator carry?

A levitator that only works on a vibration-isolated optical table at a
fixed temperature isn't a device. Every candidate here must also pass:

* thermal: stay levitated, clear of both plates, over +-3 K without
  re-adjustment (NdFeB loses ~0.12 %/K of Br, so the lift changes ~0.24 %/K);
* levelling: a 0.5 deg unlevel table may push the floater sideways by at most
  half its radius;
* clearance: the floater may use at most 75 % of each gap;
* stiffness: every normal mode at least 0.5 Hz, so no design sits on the
  edge of stability.

``max_payload`` bisects the payload mass against these constraints for one
geometry. ``search`` scans a catalogue of stock magnet sizes.
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import linalg, optimize

from .constants import BR_N42, BR_N52, G_EARTH, NDFEB_DBR_DT
from .levitator import Levitator
from .magnetics import CylinderMagnet

INCH = 25.4e-3


@dataclass(frozen=True)
class Requirements:
    clearance_frac: float = 0.25      # keep >= 25 % of the gap free on each side
    thermal_band_K: float = 3.0       # survive +-3 K without touching a plate
    base_tilt_deg: float = 0.5        # table levelling error
    max_shift_radii: float = 0.5      # allowed lateral drift under that tilt, in floater radii
    min_frequency_hz: float = 0.5     # every mode must be at least this stiff (no marginal wells)
    min_lifter_clearance: float = 1e-3


@dataclass
class Assessment:
    lev: Levitator
    stable: bool
    omega2: np.ndarray
    k_vertical: float
    k_lateral: float                  # tilt-condensed lateral stiffness [N/m]
    force_up: float                   # restoring force at the lower travel limit [N]
    force_down: float                 # restoring force at the upper travel limit [N]
    thermal_K: float                  # +-K the device tolerates
    lifter_window: float              # range of lifter heights that levitate [m]
    tilt_shift: float                 # lateral drift for the specified table tilt [m]
    feasible: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def payload(self) -> float:
        return self.lev.payload_mass

    def row(self) -> dict:
        lev = self.lev
        f = np.sqrt(np.abs(self.omega2)) / (2 * np.pi)
        return {
            "lifter": describe_magnet(lev.lifter),
            "floater": describe_magnet(lev.floater),
            "gap_mm": lev.gap * 1e3,
            "payload_g": lev.payload_mass * 1e3,
            "floater_g": lev.floater_mass * 1e3,
            "lifter_to_floater_mm": (lev.lifter_height - lev.lifter.length / 2) * 1e3,
            "thermal_K": self.thermal_K,
            "lifter_window_mm": self.lifter_window * 1e3,
            "tilt_shift_mm": self.tilt_shift * 1e3,
            "f_vertical_Hz": math.sqrt(max(self.k_vertical, 0) / lev.mass) / (2 * math.pi),
            "f_lateral_Hz": math.sqrt(max(self.k_lateral, 0) / lev.mass) / (2 * math.pi),
            "f_min_Hz": float(f.min()),
            "feasible": self.feasible,
        }


def describe_magnet(m: CylinderMagnet) -> str:
    grade = {BR_N42: "N42", BR_N52: "N52"}.get(m.br, f"Br={m.br:.2f}T")
    ring = f"/{2 * m.inner_radius * 1e3:.1f}" if m.inner_radius else ""
    return f"{grade} D{2 * m.radius * 1e3:.2f}{ring}x{m.length * 1e3:.2f}mm"


def assess(lev: Levitator, req: Requirements = Requirements(), n_z: int = 25) -> Assessment:
    reasons = []
    H = lev.hessian(full=False)
    w2 = linalg.eigh(H, lev.inertia_matrix(), eigvals_only=True)
    stable = bool(np.all(w2 > 0))
    k_lat = H[0, 0] - H[0, 4] ** 2 / H[4, 4]
    k_v = H[2, 2]
    if not stable:
        reasons.append("unstable")
    elif np.sqrt(w2.min()) / (2 * np.pi) < req.min_frequency_hz:
        reasons.append(f"soft mode {np.sqrt(w2.min()) / (2 * np.pi):.2f} Hz")

    # Vertical travel: the connected stretch around z=0 where the well is restoring.
    zc = (1 - req.clearance_frac) * lev.gap
    zs = np.linspace(-zc, zc, n_z)
    F = np.array([lev.axial_force(z) for z in zs])
    mid = n_z // 2
    lo = mid
    while lo > 0 and F[lo - 1] > F[lo]:
        lo -= 1
    hi = mid
    while hi < n_z - 1 and F[hi + 1] < F[hi]:
        hi += 1
    force_up, force_down = max(F[lo], 0.0), max(-F[hi], 0.0)
    lateral_ok = all(lev.point_dipole_estimates(z)["radial_ok"] for z in (zs[lo], zs[hi]))
    if not lateral_ok:
        reasons.append("laterally unstable within travel")

    dlift_dT = 2 * abs(NDFEB_DBR_DT) * lev.weight          # lift ~ Br_lifter * Br_floater
    thermal = min(force_up, force_down) / dlift_dT
    if thermal < req.thermal_band_K:
        reasons.append(f"thermal +-{thermal:.2f} K")

    dh = 1e-5
    dF_dh = (replace(lev, lifter_height=lev.lifter_height + dh).axial_force(0.0)
             - replace(lev, lifter_height=lev.lifter_height - dh).axial_force(0.0)) / (2 * dh)
    window = (force_up + force_down) / abs(dF_dh) if dF_dh else 0.0

    tilt = math.radians(req.base_tilt_deg)
    shift = lev.weight * math.sin(tilt) / k_lat if k_lat > 0 else math.inf
    if shift > req.max_shift_radii * lev.floater.radius:
        reasons.append(f"tilt shift {shift * 1e3:.2f} mm")
    if lev.lifter_clearance < req.min_lifter_clearance:
        reasons.append("lifter collides with plate")

    return Assessment(lev, stable, w2, k_v, k_lat, force_up, force_down, thermal, window,
                      shift, feasible=not reasons, reasons=reasons)


# ----------------------------------------------------------------- payload
def _lift(lev: Levitator, h: float) -> float:
    """Upward magnetic + diamagnetic force on the centred floater [N]."""
    probe = replace(lev, lifter_height=h)
    return probe.axial_force(0.0) + probe.weight


class Balancer:
    """Places the lifter for any payload: all balance heights, each assessed."""

    def __init__(self, template: Levitator, req: Requirements = Requirements(), n_heights: int = 90):
        self.req = req
        self.lev0 = lev0 = replace(template, payload_mass=0.0)
        h_min = (lev0.plate_face + lev0.plate_thickness + lev0.lifter.length / 2
                 + req.min_lifter_clearance)
        h_max = h_min + 25 * max(lev0.lifter.radius, lev0.lifter.length)
        self.hs = np.geomspace(h_min, h_max, n_heights)
        self.lift = np.array([_lift(lev0, h) for h in self.hs])

    @property
    def max_liftable_payload(self) -> float:
        return max(0.0, self.lift.max() / G_EARTH - self.lev0.floater_mass)

    def candidates(self, payload: float) -> list[Assessment]:
        lev = replace(self.lev0, payload_mass=payload)
        W = lev.weight
        g = self.lift - W
        out = []
        for i in range(len(self.hs) - 1):
            if g[i] * g[i + 1] < 0:
                h = optimize.brentq(lambda hh: _lift(self.lev0, hh) - W, self.hs[i], self.hs[i + 1],
                                    xtol=1e-8)
                out.append(assess(replace(lev, lifter_height=h), self.req))
        return out

    def best(self, payload: float) -> Assessment | None:
        ok = [c for c in self.candidates(payload) if c.feasible]
        return max(ok, key=lambda c: c.thermal_K) if ok else None


def max_payload(template: Levitator, req: Requirements = Requirements(),
                n_heights: int = 90, rtol: float = 0.005, max_iter: int = 40) -> Assessment | None:
    """Largest payload (CoM at the floater centre) meeting ``req``, with the
    lifter height re-balanced for every trial mass. None if even zero payload fails."""
    bal = Balancer(template, req, n_heights)
    base = bal.best(0.0)
    if base is None:
        return None
    lo_m, hi_m = 0.0, bal.max_liftable_payload
    top = bal.best(hi_m * 0.999)
    if top is not None:
        return top
    best_ok = base
    for _ in range(max_iter):
        if hi_m - lo_m <= rtol * max(lo_m, 1e-4):
            break
        mid = 0.5 * (lo_m + hi_m)
        a = bal.best(mid)
        if a is None:
            hi_m = mid
        else:
            lo_m, best_ok = mid, a
    return best_ok


# ---------------------------------------------------------------- catalogue
def lifter_catalogue() -> list[CylinderMagnet]:
    """Common stock NdFeB (N42) lifter sizes (diameter x thickness, inches)."""
    sizes = [(1, 0.5), (1.5, 0.5), (2, 1), (3, 1), (4, 1)]
    return [CylinderMagnet(d * INCH / 2, t * INCH, BR_N42) for d, t in sizes]


def floater_catalogue() -> list[CylinderMagnet]:
    """Common stock N52 disc/cylinder floater sizes."""
    ds = [1 / 8, 3 / 16, 1 / 4, 3 / 8, 1 / 2]
    ts = [1 / 16, 1 / 8, 1 / 4, 3 / 8, 1 / 2]
    out = []
    for d in ds:
        for t in ts:
            if d / 8 <= t <= 1.5 * d:
                out.append(CylinderMagnet(d * INCH / 2, t * INCH, BR_N52))
    return out


GAPS = (0.5e-3, 0.75e-3, 1.0e-3, 1.5e-3)


def make_template(lifter: CylinderMagnet, floater: CylinderMagnet, gap: float,
                  plate_thickness: float = 2e-3) -> Levitator:
    return Levitator(lifter=lifter, floater=floater, lifter_height=1.0, gap=gap,
                     plate_thickness=plate_thickness,
                     plate_radius=max(5 * floater.radius, floater.radius + 10e-3))


def _job(args):
    lifter, floater, gap, req = args
    a = max_payload(make_template(lifter, floater, gap), req)
    return None if a is None else a.row() | {"_assessment": a}


def search(lifters=None, floaters=None, gaps=GAPS, req: Requirements = Requirements(),
           workers: int = 4) -> list[dict]:
    lifters = lifters if lifters is not None else lifter_catalogue()
    floaters = floaters if floaters is not None else floater_catalogue()
    jobs = [(l, f, g, req) for l in lifters for f in floaters for g in gaps]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_job, jobs, chunksize=2) if r is not None]
    rows.sort(key=lambda r: -r["payload_g"])
    return rows


THERMAL_BANDS = (3.0, 1.0, 0.3)


def scaling_study(lifter_diameters=(1, 2, 4, 8),
                  floater_diameters=(1 / 4, 3 / 8, 1 / 2, 3 / 4, 1, 1.5, 2),
                  floater_aspects=(0.25, 0.5), gaps=(0.5e-3, 0.75e-3),
                  bands=THERMAL_BANDS, workers: int = 4) -> dict:
    """Best payload vs lifter size (N42 discs, diameter = 2 x thickness), for
    several thermal-tolerance requirements. Diameters in inches."""
    from .constants import RHO_NDFEB

    lifters = [CylinderMagnet(d * INCH / 2, d * INCH / 2, BR_N42) for d in lifter_diameters]
    floaters = [CylinderMagnet(d * INCH / 2, a * d * INCH, BR_N52)
                for d in floater_diameters for a in floater_aspects]
    jobs = [(l, f, g, Requirements(thermal_band_K=b))
            for b in bands for l in lifters for f in floaters for g in gaps]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_job, jobs, chunksize=2))
    out = {}
    for (l, f, g, req), r in zip(jobs, results):
        key = (req.thermal_band_K, l.radius)
        if r is not None and (key not in out or r["payload_g"] > out[key]["payload_g"]):
            out[key] = r
    study = {}
    for b in bands:
        pts = []
        for l in lifters:
            r = out.get((b, l.radius))
            mass = RHO_NDFEB * l.volume
            pts.append({"lifter_kg": mass, "lifter": describe_magnet(l),
                        "payload_g": r["payload_g"] if r else 0.0,
                        "best": {k: v for k, v in r.items() if not k.startswith("_")} if r else None})
        study[b] = pts
    return study


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if not str(k).startswith("_")}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main(out_path: str = "results/catalogue_search.json",
         scaling_path: str = "results/scaling_study.json") -> None:
    import json
    import os
    import time

    t0 = time.time()
    rows = search()
    clean = [_jsonable(r) for r in rows]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"requirements": Requirements().__dict__, "designs": clean}, fh, indent=1)
    print(f"{len(rows)} feasible geometries in {time.time() - t0:.0f} s -> {out_path}")
    for r in clean[:15]:
        print(f"{r['payload_g']:7.2f} g payload | {r['lifter']:<22} {r['floater']:<20} "
              f"gap {r['gap_mm']:.2f} mm | +-{r['thermal_K']:.1f} K | window {r['lifter_window_mm']:.2f} mm")

    t0 = time.time()
    study = scaling_study()
    with open(scaling_path, "w") as fh:
        json.dump(_jsonable(study), fh, indent=1)
    print(f"scaling study in {time.time() - t0:.0f} s -> {scaling_path}")
    for band, pts in study.items():
        print(f"+-{band} K: " + ", ".join(f"{p['lifter_kg']:.2g} kg -> {p['payload_g']:.1f} g" for p in pts))


if __name__ == "__main__":
    main()
