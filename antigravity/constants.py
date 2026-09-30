"""Physical constants and material data (SI units throughout)."""

import math

G = 6.67430e-11          # gravitational constant [m^3 kg^-1 s^-2]
C = 299_792_458.0        # speed of light [m/s]
HBAR = 1.054571817e-34   # reduced Planck constant [J s]
MU0 = 4e-7 * math.pi     # vacuum permeability [T m / A] (1.25663706e-6; the 2019 SI value differs by 1e-10)
G_EARTH = 9.80665        # standard gravity [m/s^2]

R_EARTH = 6.371e6        # mean Earth radius [m]
M_EARTH = 5.972e24       # Earth mass [kg]
RHO_AIR = 1.225          # sea-level air density [kg/m^3]

# Permanent magnets: remanence Br [T] (typical datasheet mid values) and density.
BR_N35 = 1.19
BR_N42 = 1.30
BR_N52 = 1.45
RHO_NDFEB = 7500.0       # [kg/m^3]
# Reversible temperature coefficient of Br for NdFeB, per kelvin (typ. -0.11 .. -0.12 %/K).
NDFEB_DBR_DT = -0.0012

# Diamagnetic volume susceptibilities (SI, dimensionless), from
# Simon, Heflinger & Geim, Am. J. Phys. 69, 702 (2001), Table I.
CHI_PG_PERP = -450e-6    # pyrolytic graphite, field along the c-axis (normal to the sheets)
CHI_PG_PAR = -85e-6      # pyrolytic graphite, field in the plane of the sheets
CHI_BISMUTH = -170e-6
CHI_GRAPHITE_ROD = -160e-6
CHI_WATER = -8.8e-6
