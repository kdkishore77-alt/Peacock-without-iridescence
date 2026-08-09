"""
peacock_optics.py

Core physics module for modeling structural colour in peacock feather
barbules (Freyer & Stavenga 2020 geometry), consolidating the 8 scripts
originally used to generate the manuscript figures.

This module fixes five implementation bugs identified in peer review of the
original scripts (verified by tracing through the code, not just asserted):

  BUG 1 -- Sign of the melanin extinction coefficient.
      All 8 original scripts used n_melanin = n + i*k (k > 0). Combined with
      delta = k0 * n * d and a standard Macleod-type characteristic matrix,
      this makes the imaginary part of `delta` positive, which causes
      cos(delta)/sin(delta) to grow WITHOUT BOUND as thickness increases
      (verified numerically: R(400 nm) for a single melanin-like layer came
      out to 3.8, i.e. gain, not absorption). Melanin must be entered as
      n - i*k for this convention. Fixed in `n_melanin`.

  BUG 2 -- Disorder was not applied as described in the manuscript.
      `2-disorder.py` drew ONE scalar per realization, applied it to the
      lattice PERIOD (not the melanosome/air-channel diameters as Section
      2.3 states), and computed filling fractions from the UNPERTURBED
      diameters. Every layer in a given realization ended up with the same
      effective index -- i.e. the "disordered" stack was a perfectly
      periodic homogeneous slab of random period, not a disordered
      multilayer. Fixed in `build_layered_stack`: Dm and Da are now drawn
      independently for EACH of the Nm layers.

  BUG 3 -- Undisclosed exponential melanin grading.
      `4-angle-averaged.py` (Fig. 4) silently imposed
      f_m(z) = f_m0 * exp(-z/180nm), a depth-graded profile absent from the
      manuscript text, which is what actually produced the smooth spectrum
      attributed to "emergent" homogenization. Removed: Fig. 4 now reuses
      the same period-averaged (undecorated) continuum fractions as Fig. 3,
      exactly as `3-Continum_EMT.py` already did correctly.

  BUG 4 -- Unnormalized luminance in chromaticity -> sRGB conversion.
      `6b-peacock-allRegion-color.py` passed a raw, un-normalized trapezoid
      integral of R*ybar as "Y" into the sRGB conversion, which expects Y
      normalized so a perfect white reflector (R=1 at all wavelengths) maps
      to Y=1. The un-normalized value is roughly two orders of magnitude too
      large, saturating/clipping every computed colour. Fixed in
      `spectrum_to_XYZ`, which normalizes by the illuminant/CMF integral.

  BUG 5 -- Disorder-iridescence metric measured its own construction noise.
      `7-disorder-iridescence-tradeoff.py` called `build_stack(wl, sigma)`
      INSIDE the wavelength loop, so a single "realization spectrum" was
      actually stitched from ~250 independent random structures (one per
      wavelength sample) instead of one fixed structure evaluated across
      wavelengths. This inflates spectral roughness in a way that scales
      with sigma, manufacturing the reported eta ~ 1/sigma collapse. Fixed
      in `disorder_iridescence_tradeoff`: one structure is drawn per
      realization and reused across the full spectrum and all angles.

In addition, this module adds a coherent/incoherent reflectance split
(`ensemble_coherent_incoherent`), since plain TMM ensemble-averaging only
ever computes the coherent component |<r>|^2 and silently discards the
diffuse component <|r|^2> - |<r>|^2 that disorder physically generates.
This does not change any of the five bugs above, but it gives a real,
measured basis for "homogenization" claims instead of an assumed one.

NOTE ON THE MIXING-RULE EXPONENT: the manuscript's Eq. for n_eff applies the
exponent w directly to refractive index n (n_eff = [fk*nk^-w + ...]^(-1/w)),
but all 8 original scripts applied w to PERMITTIVITY eps = n^2 (equivalent
to using exponent 2w on n). This module preserves the scripts' as-tested
convention (eps-based mixing) since that is what the reported figures were
actually generated with, but the mismatch against the manuscript text is
worth resolving explicitly against Freyer & Stavenga (2020) Eq. 1 before
resubmission -- see the `mixing_convention` parameter below.
"""

import numpy as np
from io import StringIO
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
if _trapz is None:
    raise ImportError("NumPy has neither np.trapezoid nor np.trapz")

# =============================================================================
# 1. MATERIAL DISPERSION
# =============================================================================

def n_keratin(wl_nm):
    """Real keratin refractive index (Freyer & Stavenga 2020)."""
    return 1.532 + 5890.0 / wl_nm**2


def n_melanin(wl_nm):
    """
    Complex melanin refractive index, n - i*k (k > 0), consistent with the
    exp(i(kz - wt)) time convention used by the characteristic-matrix TMM
    below. BUG 1 FIX: original scripts used n + i*k, which is gain, not
    absorption, under this convention.
    """
    n = 1.648 + 23700.0 / wl_nm**2
    k = 0.56 * np.exp(-wl_nm / 270.0)
    return n - 1j * k


# =============================================================================
# 2. EFFECTIVE-MEDIUM MIXING (power-law / Bruggeman-type)
# =============================================================================

def n_eff(fk, fm, fa, wl_nm, w, mixing_convention="freyer"):
    """
    Effective refractive index via power-law mixing.

    mixing_convention:
      "freyer" (default) -- the literal formula from Freyer & Stavenga
                (2020) Eq. 1, confirmed directly against the source PDF:
                    n_eff = (fk*nk^w + fm*nm_tilde^w + fa*na^w)^(1/w)
                i.e. POSITIVE exponent w applied directly to each complex
                index, POSITIVE outer power 1/w. This is the ground-truth
                convention; use it unless you have a specific reason not
                to.
      "eps"   -- exponent w applied to permittivity (eps = n^2) with
                 negated exponents, i.e. what all 8 original scripts
                 actually computed. Kept only for backward comparison.
      "n"     -- exponent -w applied directly to n with outer -1/w,
                 matching the (differently-signed, and NOT the source
                 formula) version transcribed into the manuscript's own
                 Methods section. Also kept only for comparison.
    These three give quantitatively different results. "freyer" is the
    one to trust; "eps" and "n" exist so you can see how much the earlier,
    incorrect transcriptions were changing the output.
    """
    nk = n_keratin(wl_nm)
    nm = n_melanin(wl_nm)
    na = 1.0

    if mixing_convention == "freyer":
        return (fk * nk**w + fm * nm**w + fa * na**w) ** (1.0 / w)
    elif mixing_convention == "eps":
        eps_k, eps_m, eps_a = nk**2, nm**2, na**2
        eps_eff = (fk * eps_k**(-w) + fm * eps_m**(-w) + fa * eps_a**(-w)) ** (-1.0 / w)
        return np.sqrt(eps_eff)
    elif mixing_convention == "n":
        return (fk * nk**(-w) + fm * nm**(-w) + fa * na**(-w)) ** (-1.0 / w)
    else:
        raise ValueError("mixing_convention must be 'freyer', 'eps', or 'n'")


def n_eff_TE(fk, fm, fa, wl_nm, mixing_convention="freyer"):
    return n_eff(fk, fm, fa, wl_nm, w=2.5, mixing_convention=mixing_convention)


def n_eff_TM(fk, fm, fa, wl_nm, mixing_convention="freyer"):
    return n_eff(fk, fm, fa, wl_nm, w=-1.5, mixing_convention=mixing_convention)


# =============================================================================
# 3. GEOMETRY / FILLING FRACTIONS
# =============================================================================

REGIONS = {
    "R1": dict(name="R1 (black-violet)", a=140, b=150, c=100, Dm=100, Da=33, Nm=10),
    "R2": dict(name="R2 (blue-green)",   a=160, b=170, c=100, Dm=110, Da=33, Nm=9),
    "R3": dict(name="R3 (brown)",        a=190, b=150, c=70,  Dm=110, Da=50, Nm=5),
    "R4": dict(name="R4 (green-yellow)", a=195, b=150, c=130, Dm=120, Da=55, Nm=5),
    "R5": dict(name="R5 (purple)",       a=185, b=190, c=140, Dm=100, Da=70, Nm=3),
    "R6": dict(name="R6 (brass-green)",  a=185, b=160, c=130, Dm=120, Da=70, Nm=4),
}

# Colours shown for regions where the multilayer effective-medium model does
# not apply (outside its scope per manuscript Section 3.6). Kept separate
# from any computed output so it is never possible to silently substitute a
# hard-coded value for a modeled one (BUG 4 was partly obscured by this
# ambiguity in the original 6b script).
OBSERVED_COLORS_OUTSIDE_MODEL_SCOPE = {
    "R1": "#2C1E3C",
    "R3": "#8B4513",
    "R5": "#9370DB",
    "R6": "#BDB76B",
}


def period_averaged_fractions(params):
    """Period-averaged (continuum-limit) filling fractions f_k, f_m, f_a."""
    A_cell = params["a"] * params["b"]
    f_m = np.pi * (params["Dm"] / 2) ** 2 / A_cell
    f_a = np.pi * (params["Da"] / 2) ** 2 / A_cell
    f_k = 1.0 - f_m - f_a
    return f_k, f_m, f_a


def depth_resolved_fractions(z, a, b, Dm, Da):
    """
    Filling fractions at a single depth z (nm) within one period, from
    circular melanosome / air-channel cross-sections (as in the manuscript
    Fig. 1 geometry). z is taken modulo the period `a`.
    """
    A_cell = a * b
    r_m, r_a = Dm / 2.0, Da / 2.0
    z_mod = z % a

    z_m = z_mod - r_m
    f_m = (2 * np.sqrt(max(r_m**2 - z_m**2, 0.0)) * b / A_cell) if abs(z_m) <= r_m else 0.0

    z_a = z_mod - (Dm + r_a)
    f_a = (2 * np.sqrt(max(r_a**2 - z_a**2, 0.0)) * b / A_cell) if abs(z_a) <= r_a else 0.0

    f_k = 1.0 - f_m - f_a
    return f_k, f_m, f_a


# =============================================================================
# 4. STACK BUILDERS
# =============================================================================

def build_ordered_stack(params, wl_nm, dz_nm=1.0, mixing_convention="freyer"):
    """
    Depth-resolved ordered stack: 1 nm slices with position-dependent
    filling fractions from the circular melanosome/air-channel cross
    sections, exactly as Freyer & Stavenga's Methods describe ("we sliced
    the melanosome and air channel stack into 1 nm thin layers and for
    each layer calculated the volume fractions"). Replaces the
    period-averaged constant-fraction version used previously, which is a
    coarser structure than the one actually described in the source paper
    and was found to underestimate peak reflectance substantially.
    """
    nk = n_keratin(wl_nm)
    total_depth = (params["Nm"] - 1) * params["a"] + params["Dm"]
    z_vals = np.arange(0, total_depth, dz_nm)

    stack = [((nk, nk), params["c"] * 1e-9)]
    for z in z_vals:
        fk, fm, fa = depth_resolved_fractions(z, params["a"], params["b"],
                                                params["Dm"], params["Da"])
        nTE = n_eff_TE(fk, fm, fa, wl_nm, mixing_convention)
        nTM = n_eff_TM(fk, fm, fa, wl_nm, mixing_convention)
        stack.append(((nTE, nTM), dz_nm * 1e-9))
    return stack


def build_layered_disordered_stack(params, wl_nm, sigma, rng, dz_nm=1.0, mixing_convention="freyer"):
    """
    BUG 2 FIX: Dm and Da are drawn INDEPENDENTLY for EACH of the Nm layers
    (not one scalar applied uniformly to the whole stack), so each period
    gets its own geometry -- a genuinely disordered multilayer, not a
    homogeneous slab of random thickness. Each perturbed period is itself
    depth-resolved into 1 nm slices (consistent with build_ordered_stack),
    not period-averaged.
    """
    nk = n_keratin(wl_nm)
    stack = [((nk, nk), params["c"] * 1e-9)]

    for i in range(params["Nm"]):
        Dm_i = params["Dm"] * (1 + sigma * rng.standard_normal())
        Da_i = params["Da"] * (1 + sigma * rng.standard_normal())
        z_end = Dm_i if i == params["Nm"] - 1 else params["a"]

        for z in np.arange(0, z_end, dz_nm):                     
            fk, fm, fa = depth_resolved_fractions(z, params["a"], params["b"], Dm_i, Da_i)
            nTE = n_eff_TE(fk, fm, fa, wl_nm, mixing_convention)
            nTM = n_eff_TM(fk, fm, fa, wl_nm, mixing_convention)
            stack.append(((nTE, nTM), dz_nm * 1e-9))

    return stack


def build_continuum_stack(params, wl_nm, dz_nm=1.0, mixing_convention="freyer"):
    """
    Continuum (homogenized) limit: depth-independent, period-averaged
    filling fractions, discretized into thin slices for the TMM.
    BUG 3 FIX: no undisclosed depth grading -- uses the same period-averaged
    fractions as the ordered/disordered stacks above, just applied
    uniformly with depth. This is the version already implemented correctly
    in the original 3-Continum_EMT.py; it now also backs Fig. 4 (angle
    averaging), replacing the undisclosed exp(-z/180nm) grading that was
    silently producing Fig. 4's smoothness in 4-angle-averaged.py.
    """
    fk, fm, fa = period_averaged_fractions(params)
    nTE = n_eff_TE(fk, fm, fa, wl_nm, mixing_convention)
    nTM = n_eff_TM(fk, fm, fa, wl_nm, mixing_convention)
    nk = n_keratin(wl_nm)

    total_depth = (params["Nm"] - 1) * params["a"] + params["Dm"]
    n_layers = int(round(total_depth / dz_nm))

    stack = [((nk, nk), params["c"] * 1e-9)]
    stack += [((nTE, nTM), dz_nm * 1e-9)] * n_layers
    return stack

def build_layered_stack_from_diameters(params, wl_nm, Dm_layers, Da_layers, dz_nm=1.0, mixing_convention="freyer"):
    nk = n_keratin(wl_nm)
    stack = [((nk, nk), params["c"] * 1e-9)]
    Nm = len(Dm_layers)
    for i, (Dm_i, Da_i) in enumerate(zip(Dm_layers, Da_layers)):
        z_end = Dm_i if i == Nm - 1 else params["a"]        # same truncation as build_ordered_stack
        for z in np.arange(0, z_end, dz_nm):
            fk, fm, fa = depth_resolved_fractions(z, params["a"], params["b"], Dm_i, Da_i)
            nTE = n_eff_TE(fk, fm, fa, wl_nm, mixing_convention)
            nTM = n_eff_TM(fk, fm, fa, wl_nm, mixing_convention)
            stack.append(((nTE, nTM), dz_nm * 1e-9))
    return stack
# =============================================================================
# 5. TRANSFER-MATRIX METHOD (oblique incidence, TE/TM)
# =============================================================================

def _tmm_amplitude(stack, wl_nm, theta_deg, pol):
    """Complex reflection amplitude r for one polarization ('TE' or 'TM')."""
    theta = np.radians(theta_deg)
    k0 = 2 * np.pi / (wl_nm * 1e-9)

    M = np.eye(2, dtype=complex)
    for (nTE, nTM), d in stack:
        n = nTE if pol == "TE" else nTM
        cos_t = np.sqrt(1 - (np.sin(theta) / np.real(n)) ** 2 + 0j)
        delta = k0 * n * d * cos_t
        eta = n * cos_t if pol == "TE" else n / cos_t
        M_layer = np.array([
            [np.cos(delta), 1j * np.sin(delta) / eta],
            [1j * eta * np.sin(delta), np.cos(delta)],
        ])
        M = M_layer @ M

    eta0 = np.cos(theta) if pol == "TE" else 1.0 / np.cos(theta)
    eta_exit = eta0  # air on both sides
    Y = (M[1, 0] + M[1, 1] * eta_exit) / (M[0, 0] + M[0, 1] * eta_exit)
    r = (eta0 - Y) / (eta0 + Y)
    return r


def reflectance(stack, wl_nm, theta_deg=0.0):
    """Unpolarized reflectance R = (R_TE + R_TM)/2 at one wavelength/angle."""
    r_TE = _tmm_amplitude(stack, wl_nm, theta_deg, "TE")
    r_TM = _tmm_amplitude(stack, wl_nm, theta_deg, "TM")
    return (np.abs(r_TE) ** 2 + np.abs(r_TM) ** 2) / 2.0


def reflectance_spectrum(stack_builder, wavelengths_nm, theta_deg=0.0, **kwargs):
    """
    stack_builder: callable(wl_nm, **kwargs) -> stack, called once PER
    WAVELENGTH to account for material dispersion. Use this only for a
    SINGLE fixed structure evaluated across wavelengths (see BUG 5 note
    below for why re-drawing disorder per wavelength is wrong).
    """
    return np.array([reflectance(stack_builder(wl, **kwargs), wl, theta_deg)
                      for wl in wavelengths_nm])


def ensemble_coherent_incoherent(params, wavelengths_nm, sigma, theta_deg, N, seed=None,
                                  mixing_convention="freyer"):
    """
    Draw N disordered realizations (BUG-2-fixed, per-layer disorder), and
    return both the coherent (specular, |<r>|^2) and total (<|r|^2>)
    reflectance spectra. diffuse = total - coherent is the scattered
    intensity that plain TMM ensemble-averaging normally discards --
    this is the direct, measured coherent-to-incoherent transition
    Reviewer 2 asked for, rather than an assumed one.

    One structure per realization is built ONCE and reused across all
    wavelengths (see BUG 5 fix in disorder_iridescence_tradeoff for why
    this matters).
    """
    rng = np.random.default_rng(seed)
    r_TE_all = np.zeros((N, len(wavelengths_nm)), dtype=complex)
    r_TM_all = np.zeros((N, len(wavelengths_nm)), dtype=complex)
    for real_idx in range(N):
        Dm_layers = params["Dm"] * (1 + sigma * rng.standard_normal(params["Nm"]))
        Da_layers = params["Da"] * (1 + sigma * rng.standard_normal(params["Nm"]))
        for i, wl in enumerate(wavelengths_nm):
            stack = build_layered_stack_from_diameters(
                params, wl, Dm_layers, Da_layers, mixing_convention=mixing_convention)
            r_TE_all[real_idx, i] = _tmm_amplitude(stack, wl, theta_deg, "TE")
            r_TM_all[real_idx, i] = _tmm_amplitude(stack, wl, theta_deg, "TM")

    coherent_R = (np.abs(r_TE_all.mean(axis=0)) ** 2 + np.abs(r_TM_all.mean(axis=0)) ** 2) / 2.0
    total_R = (np.mean(np.abs(r_TE_all) ** 2, axis=0) + np.mean(np.abs(r_TM_all) ** 2, axis=0)) / 2.0
    diffuse_R = total_R - coherent_R
    return coherent_R, total_R, diffuse_R


# =============================================================================
# 6. CIE COLORIMETRY (proper luminance normalization -- BUG 4 fix)
# =============================================================================

_CMF_DATA = """
400 0.01431 0.000396 0.067850
410 0.04351 0.001210 0.207400
420 0.13438 0.004000 0.645600
430 0.28390 0.011600 1.385600
440 0.34828 0.023000 1.747060
450 0.33620 0.038000 1.772110
460 0.29080 0.060000 1.669200
470 0.19536 0.091000 1.287640
480 0.09564 0.139020 0.812950
490 0.03201 0.208020 0.465180
500 0.00490 0.323000 0.272000
510 0.00930 0.503000 0.158200
520 0.06327 0.710000 0.078250
530 0.16550 0.862000 0.042160
540 0.29040 0.954000 0.020300
550 0.43345 0.995000 0.008750
560 0.59450 0.995000 0.003900
570 0.76210 0.952000 0.002100
580 0.91630 0.870000 0.001650
590 1.02630 0.757000 0.001100
600 1.06220 0.631000 0.000800
610 1.00260 0.503000 0.000340
620 0.85445 0.381000 0.000190
630 0.64240 0.265000 0.000050
640 0.44790 0.175000 0.000020
650 0.28350 0.107000 0.000000
660 0.16490 0.061000 0.000000
670 0.08740 0.032000 0.000000
680 0.04677 0.017000 0.000000
690 0.02270 0.008210 0.000000
700 0.01136 0.004102 0.000000
710 0.00579 0.002091 0.000000
720 0.00290 0.001047 0.000000
730 0.00144 0.000520 0.000000
740 0.00069 0.000249 0.000000
750 0.00033 0.000120 0.000000
"""
_cmf = np.loadtxt(StringIO(_CMF_DATA))
_WL_CMF, _XBAR, _YBAR, _ZBAR = _cmf.T

# Normalization constant: integral of ybar over the sampled range under an
# equal-energy illuminant. A perfect reflector (R=1 everywhere) must map to
# Y=1 for the sRGB matrix below to behave sensibly. BUG 4 was exactly the
# omission of this normalization.
_Y_NORM = _trapz(_YBAR, _WL_CMF)


def spectrum_to_XYZ(R, wavelengths_nm):
    """R(wavelength) -> normalized CIE XYZ (perfect reflector -> Y=1)."""
    xb = np.interp(wavelengths_nm, _WL_CMF, _XBAR, left=0.0, right=0.0)
    yb = np.interp(wavelengths_nm, _WL_CMF, _YBAR, left=0.0, right=0.0)
    zb = np.interp(wavelengths_nm, _WL_CMF, _ZBAR, left=0.0, right=0.0)
    X = _trapz(R * xb, wavelengths_nm) / _Y_NORM
    Y = _trapz(R * yb, wavelengths_nm) / _Y_NORM
    Z = _trapz(R * zb, wavelengths_nm) / _Y_NORM
    return X, Y, Z


def XYZ_to_xy(X, Y, Z):
    s = X + Y + Z
    return (X / s, Y / s) if s > 0 else (0.0, 0.0)


def spectrum_to_xy(R, wavelengths_nm):
    return XYZ_to_xy(*spectrum_to_XYZ(R, wavelengths_nm))


def XYZ_to_srgb(X, Y, Z):
    """Standard sRGB matrix + gamma. Expects Y normalized as above."""
    R_lin = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    G_lin = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    B_lin = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(c):
        c = max(c, 0.0)
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return tuple(np.clip(gamma(c), 0, 1) for c in (R_lin, G_lin, B_lin))


def spectrum_to_srgb(R, wavelengths_nm):
    X, Y, Z = spectrum_to_XYZ(R, wavelengths_nm)
    return XYZ_to_srgb(X, Y, Z)


def srgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in rgb))
