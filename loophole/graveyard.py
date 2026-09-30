"""The graveyard: order-of-magnitude checks of every "true" anti-gravity route.

Each function sizes one idea using published experimental limits and returns a
Route with the key numbers. Where a choice exists, the numbers are set to be
*generous* to the idea, so each dead end holds with room to spare.

Run ``python -m loophole.graveyard`` for the summary table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import C, G, G_EARTH, HBAR, R_EARTH, M_EARTH, RHO_AIR


@dataclass
class Route:
    name: str
    idea: str
    numbers: dict[str, float] = field(default_factory=dict)
    verdict: str = ""
    alive: bool = False


# ---------------------------------------------------------------------------
# 1. Gravitational shielding (Majorana absorption, Podkletnov's claim)
# ---------------------------------------------------------------------------

# Majorana (1920) modelled shielding as exp(-h * column density).
H_MAJORANA_CLAIM = 6.7e-13   # [m^2/kg]  his claimed value (6.7e-12 cm^2/g)
H_LLR_BOUND = 1e-21          # [m^2/kg]  lunar-laser-ranging bound; the weakest quoted
                             # value is used on purpose (other analyses give ~1e-22)
RHO_LEAD = 11_340.0


def gravitational_shielding(weight_reduction: float = 0.01) -> Route:
    """Column density of shield needed to cut a weight by ``weight_reduction``."""
    sigma_needed = -math.log(1.0 - weight_reduction) / H_LLR_BOUND   # [kg/m^2]
    rho_earth = M_EARTH / (4 / 3 * math.pi * R_EARTH**3)
    earth_column = rho_earth * 2 * R_EARTH                             # through the diameter
    lead_1m = -math.expm1(-H_LLR_BOUND * RHO_LEAD * 1.0)
    return Route(
        name="Gravity shielding",
        idea="Block gravity with a material layer, as a Faraday cage blocks E-fields.",
        numbers={
            "shield column density for 1% lift [kg/m^2]": sigma_needed,
            "...in Earth-diameters of rock": sigma_needed / earth_column,
            "weight reduction behind 1 m of lead (upper bound)": lead_1m,
            "Podkletnov claimed reduction": 0.02,
            "Hathaway et al. replication, no effect down to": 1e-5,
        },
        verdict=(
            "Dead. Gravity has one sign of charge, so there is nothing to polarise "
            "and no cage. Lunar ranging bounds any absorption so tightly that a 1% "
            "shield would need a stack of ~1e8 Earths. The 2% superconductor "
            "claim did not replicate (null to 0.001%)."
        ),
    )


# ---------------------------------------------------------------------------
# 2. Negative mass / antimatter / "anti-gravity materials"
# ---------------------------------------------------------------------------

ALPHA_G_ACCEL = 0.75          # antihydrogen free-fall, in units of g (Nature 621, 716, 2023)
ALPHA_G_SIGMA = math.hypot(0.13, 0.16)
MICROSCOPE_ETA = 1.5e-15      # |Eotvos parameter| central value, Ti vs Pt (PRL 129, 121102, 2022)
MICROSCOPE_ETA_ERR = math.hypot(2.3e-15, 1.5e-15)


def negative_mass(distance: float = 1.0) -> Route:
    """How much negative mass a 'repulsor' would need, and why materials can't self-levitate."""
    sigmas_from_repulsive = (ALPHA_G_ACCEL - (-1.0)) / ALPHA_G_SIGMA
    # By the equivalence principle even a negative mass *falls down* (Bondi 1957):
    # a body can't opt out of the field. To push something *up* you need a
    # source whose repulsion at `distance` equals g:
    m_negative_source = G_EARTH * distance**2 / G
    return Route(
        name="Negative mass / antimatter",
        idea="Find or make matter that falls up, or that repels ordinary matter.",
        numbers={
            "antihydrogen free-fall [g]": ALPHA_G_ACCEL,
            "sigmas excluding 'falls up' (-1 g), errors in quadrature": sigmas_from_repulsive,
            "equivalence-principle bound |eta| (1 sigma)": MICROSCOPE_ETA + MICROSCOPE_ETA_ERR,
            f"negative source mass to repel at 1 g from {distance:g} m [kg]": m_negative_source,
        },
        verdict=(
            "Dead. Antimatter falls down. No composition of ordinary matter changes "
            "its free-fall rate by more than ~4 parts in 1e15. Even hypothetical "
            "negative mass would itself fall down; a repulsor needs ~1.5e11 kg of it "
            "within a metre, and none has ever been observed."
        ),
    )


# ---------------------------------------------------------------------------
# 3. Casimir negative energy as that source
# ---------------------------------------------------------------------------

GRAPHENE_AREAL_DENSITY = 7.7e-7   # [kg/m^2], thinnest conceivable plate


def casimir_negative_energy(gap: float = 10e-9) -> Route:
    """Negative energy density between ideal plates vs the positive mass of the plates."""
    u = -math.pi**2 * HBAR * C / (720 * gap**4)          # [J/m^3]
    rho_neg = u / C**2                                     # [kg/m^3]
    target = negative_mass().numbers[
        "negative source mass to repel at 1 g from 1 m [kg]"]
    volume = target / abs(rho_neg)
    earth_volume = 4 / 3 * math.pi * R_EARTH**3
    rho_plates = GRAPHENE_AREAL_DENSITY / gap              # one graphene sheet per cavity
    return Route(
        name="Casimir negative energy",
        idea="Use the vacuum's negative energy between plates as the repulsive source.",
        numbers={
            f"Casimir energy density at {gap * 1e9:g} nm [J/m^3]": u,
            "equivalent mass density [kg/m^3]": rho_neg,
            "cavity volume for a 1 g repulsor [Earth volumes]": volume / earth_volume,
            "positive plate mass / negative Casimir mass": rho_plates / abs(rho_neg),
        },
        verdict=(
            "Dead. The negative energy is real but ~1e-13 kg/m^3. A repulsor would "
            "need ~300 Earth volumes of cavities, and even one-atom-thick plates "
            "outweigh the negative energy by ~1e14. The net mass is always positive."
        ),
    )


# ---------------------------------------------------------------------------
# 4. Gravitomagnetism (frame dragging) from a spinning rotor
# ---------------------------------------------------------------------------

def gravitomagnetic_rotor(mass: float = 1000.0, radius: float = 0.5,
                          rim_speed: float = 1000.0, test_speed: float = 1000.0) -> Route:
    """Lorentz-like gravitomagnetic acceleration near an extreme lab flywheel.

    Uses the on-axis GEM field B_g = 2 G J / (c r^3) and |a| <= 2 (v/c) B_g c,
    i.e. a <= 4 G J v / (c^2 r^3). Order-of-magnitude, generous prefactors.
    """
    omega = rim_speed / radius
    J = 0.5 * mass * radius**2 * omega
    r = radius
    a_gm = 4 * G * J * test_speed / (C**2 * r**3)
    a_newton = G * mass / r**2
    return Route(
        name="Gravitomagnetism",
        idea="Spin a massive rotor so its 'gravitational magnetic field' lifts things.",
        numbers={
            "rotor angular momentum [kg m^2/s]": J,
            "gravitomagnetic accel. on a 1 km/s test mass [m/s^2]": a_gm,
            "...as a fraction of g": a_gm / G_EARTH,
            "rotor's ordinary Newtonian pull [m/s^2]": a_newton,
        },
        verdict=(
            "Dead. Frame dragging is real (Gravity Probe B measured Earth's), but a "
            "1-tonne rotor at burst speed gives ~1e-18 m/s^2, about 1e-19 g. The "
            "claimed superconductor enhancement (Tajmar 2006) was null in an "
            "independent ring-laser test (Graham et al. 2008)."
        ),
    )


# ---------------------------------------------------------------------------
# 5. Field propulsion: reactionless drives vs honest reaction
# ---------------------------------------------------------------------------

EMDRIVE_CLAIM_N_PER_W = 1.2e-6   # White et al., J. Propul. Power 33, 830 (2017): 1.2 mN/kW
EMDRIVE_REFUTATION_FACTOR = 1e3  # Tajmar et al. 2021: claims refuted by >= 3 orders of magnitude


def hover_power(mass: float = 1.0, rotor_area: float = 0.5) -> Route:
    """Power to hover a mass by various 'propulsion' routes."""
    W = mass * G_EARTH
    p_photon = W * C
    p_emdrive_claim = W / EMDRIVE_CLAIM_N_PER_W
    p_emdrive_bound = p_emdrive_claim * EMDRIVE_REFUTATION_FACTOR
    p_rotor = math.sqrt(W**3 / (2 * RHO_AIR * rotor_area))   # ideal actuator disk
    return Route(
        name="Reactionless / field drives",
        idea="EmDrive, Mach-effect, Biefeld-Brown 'lifters': thrust without throwing mass.",
        numbers={
            "photon rocket (legit, reaction via light) [W/kg]": p_photon / mass,
            "EmDrive at its *claimed* 1.2 mN/kW [W/kg]": p_emdrive_claim / mass,
            "EmDrive at the experimental upper bound [W/kg]": p_emdrive_bound / mass,
            f"ideal propeller, {rotor_area} m^2 disk [W/kg]": p_rotor / mass,
        },
        verdict=(
            "Dead as anti-gravity. Momentum conservation forbids reactionless thrust. "
            "The EmDrive signal was a thermal artefact (refuted by >= 1000x); the "
            "bound left is below a flashlight's thrust per watt. "
            "Lifters are ion-wind fans with no thrust in vacuum (Tajmar 2004). "
            "Anything that hovers must push on something, and a propeller pushing "
            "air beats a photon rocket by ~1e8."
        ),
    )


# ---------------------------------------------------------------------------
# 6. What survives
# ---------------------------------------------------------------------------

def the_loophole() -> Route:
    return Route(
        name="Field-supported levitation",
        idea=("Keep gravity as it is and cancel an object's weight with a force from "
              "a field anchored to the ground, with nothing touching the object."),
        numbers={},
        verdict=(
            "Alive, with a catch. Earnshaw (1842): no static arrangement of magnets "
            "can hold a magnet stably. Loopholes: superconductors (need cryogenics), "
            "spin (Levitron, needs spin-up), feedback (needs power), and diamagnetism "
            "(Braunbek 1939), the only one that is passive, room-temperature and "
            "zero-power. The device in loophole/levitator.py builds on that one."
        ),
        alive=True,
    )


def all_routes() -> list[Route]:
    return [
        gravitational_shielding(),
        negative_mass(),
        casimir_negative_energy(),
        gravitomagnetic_rotor(),
        hover_power(),
        the_loophole(),
    ]


def format_report(routes: list[Route] | None = None) -> str:
    routes = routes if routes is not None else all_routes()
    out = []
    for r in routes:
        tag = "ALIVE" if r.alive else "DEAD "
        out.append(f"[{tag}] {r.name}: {r.idea}")
        for k, v in r.numbers.items():
            out.append(f"    {k:<62s} {v: .3g}")
        out.append(f"    -> {r.verdict}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(format_report())
