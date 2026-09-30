# Physics notes

Derivations behind `loophole/levitator.py`, in the order they constrain the design.

## 1. Why "anti-gravity" has to mean "push on something"

The weak equivalence principle says every body falls with the same acceleration, whatever it is
made of. MICROSCOPE tested it to |η| ≈ 10⁻¹⁵ (Ti vs Pt, PRL 129, 121102, 2022). So no material,
arrangement or internal state lets an object opt out of gravity. Even hypothetical negative mass
would *fall down* in Earth's field (Bondi 1957). Only its own field would be repulsive.

So there are two ways left to keep something up:

1. **Change the field.** You need a source comparable to what you're cancelling. Repelling at 1 g from
   1 m takes |M| = g r²/G ≈ 1.5 × 10¹¹ kg of negative mass. `graveyard.py` shows that no known
   source (Casimir energy, frame dragging, shielding) comes within ten orders of magnitude.
2. **Push on something**: air (wings, rotors), exhaust (rockets), or the ground *through a field*.
   Only the last is contactless and needs no power. That's magnetic levitation.

## 2. Earnshaw's theorem

A rigid magnet with fixed moment **m** in a static field has energy U = −**m**·**B** + m_tot g z. In free
space, ∇²B_i = 0 for each Cartesian component, so ∇²U = 0 wherever the moment is fixed. A function
with zero Laplacian has no local minimum. Any force-balance point is a saddle, and the sum of the three
translational stiffnesses is exactly zero. `tests/test_levitator.py::test_trace_of_translational_hessian_is_zero_without_diamagnet`
checks this identity in the numerical model.

For an axisymmetric lifter field, expand about the levitation point (B′ = ∂B_z/∂z, B″ = ∂²B_z/∂z² on axis):

    B_z = B₀ + B′z + ½B″z² − ¼B″r² + …,     B_r = −½B′r − ½B″rz + …

A floater that aligns with the local field has U = −M|B| + mgz. Then

    force balance:        M B′ = m g
    vertical stiffness:   k_z = −M B″
    lateral stiffness:    k_r = ½ M (B″ − B′²/2B₀)

Lateral stability needs B″ > B′²/2B₀ > 0, which forces k_z < 0: the floater falls up or down.
(Simon, Heflinger & Geim, Am. J. Phys. 69, 702, 2001, eqs. 11-14.)

## 3. The loophole: diamagnetism

Earnshaw assumes the only field sources are fixed. A diamagnet (χ < 0) magnetises *against* the
applied field. Its energy −(χ/2μ₀)∫B² dV is positive and grows where the field is strong, and
∇²(B²) ≥ 0, so B² *can* have minima. Two plates of pyrolytic graphite (χ⊥ = −4.5 × 10⁻⁴ along the
c-axis, χ∥ = −0.85 × 10⁻⁴ in-plane) above and below the floater push it back toward the centre when it
moves toward either plate. That adds vertical stiffness C and turns the condition into

    C / M  >  B″  >  B′² / 2B₀

This window exists only if the diamagnetic stiffness beats the lifter's unavoidable curvature.
Everything in the design search is about keeping the window open while adding weight.

## 4. The model (no dipole approximations)

* **Lifter and floater fields.** A uniformly magnetised cylinder equals a finite solenoid with sheet current
  K = B_r/μ₀. Its field anywhere is closed-form in Bulirsch's generalised complete elliptic integral
  `cel(kc, p, c, s)` (Derby & Olbert, Am. J. Phys. 78, 229, 2010). A ring is an outer cylinder minus an inner
  one. Tests check it against Biot–Savart integration (to 2 µT), the on-axis formula, the far-field dipole,
  and ∇·B = 0, ∇×B = 0.
* **Magnet–magnet energy.** U_m = −(B_r/μ₀)∫_floater n̂·B_lifter dV, a Gauss–Legendre volume quadrature
  over the rotated and translated floater. The floater is not treated as a point dipole.
* **Diamagnetic energy.** U_d = −(1/2μ₀)∫_plates B·χ·B dV with B = B_floater + B_lifter (first order in
  |χ| ~ 10⁻⁴). The floater's self-term gives the centring stiffness. The cross term is the lifter-magnetised
  plate acting on the floater. Composite Gauss–Legendre quadrature is refined around the floater's rim,
  where the field changes fastest.
* **Stability.** A central-difference Hessian in the five rigid-body coordinates (x, y, z, tilt_x, tilt_y);
  spin about the axis is a symmetry. Mode frequencies come from the generalised eigenproblem H v = ω² M v.
  All ω² > 0 means a strict energy minimum. With eddy-current damping in the graphite it is also
  asymptotically stable; without damping it is Lyapunov stable.
* **Fast vs full.** The full model integrates the plates in 3D. The fast model integrates them
  axisymmetrically (valid for z) and drops their lateral and tilt terms, which vanish for infinite plates.
  They agree to <0.5 % on every Hessian entry (tested), and the design search uses the fast one.

Point-dipole limits are recovered as tests. Far from a thick plate the plate integral equals the image-dipole
energy μ₀|χ|m²/(64π d³). Eliminating tilt from the rigid-body Hessian reproduces Simon et al.'s
"aligned-dipole" lateral stiffness above.

## 5. Validation against a real device

Simon, Heflinger & Geim (2001, §IX.C) report a measured levitator. The lifter is a 2.5 cm × 10 cm cylinder
(25 A m²). The floater is a 4.7 mm × 1.6 mm NdFeB disc (0.024 A m², 0.22 g). The plates are graphite
(χ = −170 × 10⁻⁶). They observed:

| quantity | measured | this model |
|---|---|---|
| floater distance below lifter | 8 cm | 7.87 cm |
| gap where vertical stability is lost | ≈ 1.4 mm (double well beyond) | 1.2–1.4 mm (k_z crosses 0 at ≈ 1.35 mm) |

The model uses only the paper's measured moments, masses and susceptibility, with nothing fitted.

## 6. What limits the payload

For a given floater (moment M, plate stiffness C), the two stability inequalities meet when
C = M B″ = M B′²/2B₀. With M B′ = W this gives a hard ceiling

    W_max = √(2 B₀ M C)

The floater should sit in a *strong* field B₀ with a *gentle* gradient, which calls for a large lifter far away.
Long before that ceiling, a more mundane limit bites. NdFeB loses ≈0.12 %/K of remanence, so the lift
changes by ≈0.24 % of the weight per kelvin. The plates can absorb that only up to C × (usable travel). A
device that must survive ±ΔT without re-adjustment needs

    C · z_travel ≳ 0.0024 · ΔT · W

`design.py` enforces this together with a 0.5° levelling tolerance. With ±3 K it is the binding constraint
for 204 of the 315 feasible stock-part geometries in the catalogue, including every design that carries
more than ~3.9 g. The other 111 (all with gaps of 0.75 mm or more) first hit the 0.5 Hz stiffness floor
or the stability edge itself, where the vertical well goes flat.
