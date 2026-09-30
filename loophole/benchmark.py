"""The measured levitator used to validate the model.

Simon, Heflinger & Geim, Am. J. Phys. 69, 702 (2001), Sec. IX.C: a
2.5 cm x 10 cm cylindrical lifter (moment 25 A m^2), a 4.7 mm x 1.6 mm
NdFeB floater (0.024 A m^2, 0.22 g) and graphite-rod plates
(chi = -170e-6). Reported: the floater levitates 8 cm below the lifter,
and the vertical well becomes double-humped once the face-to-plate gap
exceeds ~1.4 mm.
"""

import math

from .constants import MU0
from .levitator import Levitator, balanced
from .magnetics import CylinderMagnet

MEASURED_DISTANCE_BELOW_LIFTER = 0.08    # [m]
MEASURED_STABILITY_LIMIT_GAP = 1.4e-3    # [m]


def simon_geim_device(gap: float = 1.0e-3, **kw) -> Levitator:
    v_l = math.pi * 0.0125**2 * 0.10
    v_f = math.pi * 2.35e-3**2 * 1.6e-3
    lev = Levitator(
        lifter=CylinderMagnet(0.0125, 0.10, 25 * MU0 / v_l),
        floater=CylinderMagnet(2.35e-3, 1.6e-3, 0.024 * MU0 / v_f),
        lifter_height=0.13, gap=gap, plate_thickness=5e-3, plate_radius=15e-3,
        chi_perp=-170e-6, chi_par=-170e-6, floater_density=0.22e-3 / v_f, **kw)
    return balanced(lev, h_lo=0.1, h_hi=0.2, n_scan=30)
