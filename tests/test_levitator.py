"""Physics tests for the levitator model (benchmark: antigravity/benchmark.py)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from antigravity.benchmark import (MEASURED_DISTANCE_BELOW_LIFTER, MEASURED_STABILITY_LIMIT_GAP,
                                   simon_geim_device)
from antigravity.constants import MU0
from antigravity.levitator import Levitator, balance
from antigravity.magnetics import CylinderMagnet


@pytest.fixture(scope="module")
def device():
    return simon_geim_device()


def test_levitation_height_matches_experiment(device):
    below_lifter_face = device.lifter_height - device.lifter.length / 2
    assert below_lifter_face == pytest.approx(MEASURED_DISTANCE_BELOW_LIFTER, abs=0.003)


def test_force_balance(device):
    assert abs(device.axial_force(0.0)) < 1e-6 * device.weight


def test_stable_at_1mm_gap(device):
    st = device.stability(full=False)
    assert st.stable
    assert st.omega2.min() > 0


def test_stability_lost_near_1p4mm_gap():
    """Vertical stiffness crosses zero between 1.2 and 1.45 mm (paper: ~1.4 mm)."""
    assert 1.2e-3 < MEASURED_STABILITY_LIMIT_GAP < 1.45e-3
    k = {}
    for gap in (1.2e-3, 1.45e-3):
        lev = simon_geim_device(gap)
        k[gap] = lev.hessian(full=False)[2, 2]
    assert k[1.2e-3] > 0 > k[1.45e-3]
    wide = simon_geim_device(1.6e-3)
    kinds = [kind for _, kind in wide.equilibria()]
    assert kinds == ["min", "max", "min"]                        # double well


def test_earnshaw_without_diamagnet(device):
    """Remove the graphite: every force-balance point must be unstable."""
    bare = replace(device, chi_perp=0.0, chi_par=0.0)
    roots = balance(bare, h_lo=0.06, h_hi=0.5, n_scan=120)
    assert roots
    for h in roots:
        st = replace(bare, lifter_height=h).stability(full=True)
        assert not st.stable
        assert st.omega2.min() < 0


def test_trace_of_translational_hessian_is_zero_without_diamagnet(device):
    """Earnshaw in its sharpest form: laplacian of a static magnetic energy is 0."""
    bare = replace(device, chi_perp=0.0, chi_par=0.0)
    H = bare.hessian(full=True)
    assert abs(np.trace(H[:3, :3])) < 1e-3 * np.abs(np.diag(H[:3, :3])).max()


def test_full_and_fast_models_agree(device):
    full = device.hessian(full=True)
    fast = device.hessian(full=False)
    assert np.allclose(full, fast, rtol=5e-3, atol=1e-6 * np.abs(full).max())


def test_tilt_condensation_reproduces_aligned_dipole_formula(device):
    """Eliminating tilt from the rigid-body Hessian gives Simon et al.'s lateral stiffness."""
    H = device.hessian(full=False)
    k_eff = H[0, 0] - H[0, 4] ** 2 / H[4, 4]
    k_dip = device.point_dipole_estimates()["k_lateral_dipole"]
    assert k_eff == pytest.approx(k_dip, rel=0.02)


def test_plate_energy_matches_image_dipole_far_away():
    """Far from a thick plate, the exact plate integral equals the image-dipole energy."""
    fl = CylinderMagnet(1e-3, 2e-3, 1.3)
    lev = Levitator(lifter=CylinderMagnet(1e-3, 1e-3, 0.0), floater=fl, lifter_height=1.0,
                    gap=15e-3, plate_thickness=0.6, plate_radius=1.5,
                    chi_perp=-1e-4, chi_par=-1e-4)
    u = lev.energy_parts((0, 0, 0, 0, 0), full=False)["diamagnetic"]
    d = fl.length / 2 + lev.gap
    t = lev.plate_thickness
    u_img = 2 * 1e-4 * MU0 * fl.moment**2 / (64 * math.pi) * (1 / d**3 - 1 / (d + t) ** 3)
    assert u == pytest.approx(u_img, rel=2e-3)


def test_simulation_conserves_energy_and_stays_trapped(device):
    from antigravity.levitator import FastEnergy, simulate
    q0 = [3e-4, 0.0, 2e-4, 0.0, 0.0]
    r = simulate(device, q0=q0, t_end=0.5)
    assert not r["touched"]
    excitation = FastEnergy(device)(q0) - FastEnergy(device)([0, 0, 0, 0, 0])
    assert np.ptp(r["energy"]) < 0.05 * excitation
    assert np.abs(r["q"][:, 2]).max() < device.gap
