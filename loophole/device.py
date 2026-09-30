"""The invention: a passive, zero-power, room-temperature levitation cell.

``python -m loophole.device`` re-derives the design (payload bisection),
verifies it with the full 3D model, runs a nonlinear disturbance test and
redraws the figures.
"""

from __future__ import annotations

import numpy as np

from .constants import BR_N42, BR_N52
from .design import INCH, Assessment, Requirements, describe_magnet, make_template, max_payload
from .magnetics import CylinderMagnet

# ---- the spec (chosen from results/catalogue_search.json; see README) -------
LIFTER = CylinderMagnet(radius=1.0 * INCH, length=1.0 * INCH, br=BR_N42)        # 2" x 1" N42 disc
FLOATER = CylinderMagnet(radius=0.25 * INCH, length=0.25 * INCH, br=BR_N52)     # 1/2" x 1/4" N52 disc
GAP = 0.5e-3                                                                     # face-to-plate clearance
PLATE_THICKNESS = 2e-3                                                           # pyrolytic graphite, c-axis vertical


def design(req: Requirements = Requirements()) -> Assessment:
    a = max_payload(make_template(LIFTER, FLOATER, GAP, PLATE_THICKNESS), req)
    if a is None:
        raise RuntimeError("spec is infeasible")
    return a


def verify(a: Assessment) -> dict:
    """Full-3D stability check plus the numbers a builder needs."""
    lev = a.lev
    st = lev.stability(full=True)
    return {
        "full_3d_stable": st.stable,
        "modes": st.summary(),
        "lifter_face_to_floater_centre_mm": (lev.lifter_height - lev.lifter.length / 2) * 1e3,
        "upper_plate_to_lifter_clearance_mm": lev.lifter_clearance * 1e3,
        "field_at_floater_mT": lev.lifter_axis_derivatives(0.0)[0] * 1e3,
        "gradient_T_per_m": lev.lifter_axis_derivatives(0.0)[1],
        "weight_supported_N": lev.weight,
        "payload_g": lev.payload_mass * 1e3,
        "floater_g": lev.floater_mass * 1e3,
        "thermal_tolerance_K": a.thermal_K,
        "lifter_adjust_window_mm": a.lifter_window * 1e3,
        "tilt_shift_mm_per_0.5deg": a.tilt_shift * 1e3,
    }


def report() -> str:
    a = design()
    v = verify(a)
    lines = [
        "LOOPHOLE LEVITATOR - predicted performance",
        f"  lifter  : {describe_magnet(LIFTER)}",
        f"  floater : {describe_magnet(FLOATER)}",
        f"  plates  : 2 x pyrolytic graphite, {PLATE_THICKNESS * 1e3:.1f} mm thick, "
        f"{2 * a.lev.plate_r * 1e3:.0f} mm across, gap {GAP * 1e3:.2f} mm",
    ]
    for k, val in v.items():
        lines.append(f"  {k:<36s} {val if isinstance(val, (str, bool)) else round(float(val), 4)}")
    return "\n".join(lines)


def make_figures(outdir: str = "figures") -> None:
    import os

    from . import figures

    os.makedirs(outdir, exist_ok=True)
    a = design()
    lev = a.lev
    figures.earnshaw_landscape(lev, f"{outdir}/earnshaw_landscape.png")
    figures.field_cross_section(lev, f"{outdir}/cross_section.png")
    q0 = [1.5e-3, 0.0, 0.15e-3, 0.0, 0.0]
    res = figures.disturbance_response(lev, f"{outdir}/disturbance.png", q0=q0, t_end=4.0)
    print(f"disturbance test: touched plate = {res['touched']}, "
          f"min clearance = {res['min_clearance'] * 1e3:.3f} mm, "
          f"energy drift = {np.ptp(res['energy']):.2e} J")

    from .benchmark import simon_geim_device

    figures.benchmark_stiffness(simon_geim_device, f"{outdir}/benchmark_validation.png")

    scaling = os.path.join(os.path.dirname(__file__), "..", "results", "scaling_study.json")
    if os.path.exists(scaling):
        import json

        with open(scaling) as fh:
            study = json.load(fh)
        series = {f"±{float(b):g} K": [(p["lifter_kg"], p["payload_g"]) for p in pts]
                  for b, pts in study.items()}
        figures.payload_scaling(series, f"{outdir}/payload_scaling.png")


if __name__ == "__main__":
    print(report())
    make_figures()
