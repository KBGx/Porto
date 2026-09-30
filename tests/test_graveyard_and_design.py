import pytest

from loophole import graveyard
from loophole.constants import BR_N42, BR_N52, C, G_EARTH
from loophole.design import INCH, Balancer, Requirements, make_template, max_payload
from loophole.magnetics import CylinderMagnet


def test_only_the_loophole_survives():
    routes = graveyard.all_routes()
    assert [r.alive for r in routes] == [False] * (len(routes) - 1) + [True]


def test_graveyard_numbers():
    sh = graveyard.gravitational_shielding().numbers
    assert sh["...in Earth-diameters of rock"] > 1e7
    nm = graveyard.negative_mass().numbers
    assert nm["sigmas excluding 'falls up' (-1 g), errors in quadrature"] > 5
    cas = graveyard.casimir_negative_energy().numbers
    assert cas["positive plate mass / negative Casimir mass"] > 1e10     # net mass stays positive
    gm = graveyard.gravitomagnetic_rotor().numbers
    assert gm["...as a fraction of g"] < 1e-15
    hp = graveyard.hover_power().numbers
    assert hp["photon rocket (legit, reaction via light) [W/kg]"] == pytest.approx(G_EARTH * C)
    # the EmDrive's experimental bound is below a photon rocket's thrust per watt
    assert hp["EmDrive at the experimental upper bound [W/kg]"] > hp[
        "photon rocket (legit, reaction via light) [W/kg]"]


@pytest.fixture(scope="module")
def small_design():
    lifter = CylinderMagnet(1 * INCH, 1 * INCH, BR_N42)              # 2" x 1" N42 disc
    floater = CylinderMagnet(0.125 * INCH, INCH / 8, BR_N52)         # 1/4" x 1/8" N52
    return max_payload(make_template(lifter, floater, 0.75e-3))


def test_max_payload_meets_requirements(small_design):
    a = small_design
    req = Requirements()
    assert a is not None and a.feasible and a.stable
    assert a.payload > 0
    assert a.thermal_K >= req.thermal_band_K
    assert a.tilt_shift <= req.max_shift_radii * a.lev.floater.radius
    assert a.lifter_window > 0


def test_payload_is_maximal(small_design):
    """5 % more payload breaks a requirement at every possible lifter height."""
    bal = Balancer(small_design.lev)
    assert bal.best(small_design.payload) is not None
    assert bal.best(small_design.payload * 1.05) is None
