"""
generate_figures.py

Reproduces Figures 1-7 of the manuscript using the corrected
`peacock_optics` module. Replaces the original 8 scripts
(1-photonic-crystal.py ... 7-disorder-iridescence-tradeoff.py).

Set FAST=True (default) for a quick correctness-check run with reduced
ensemble sizes / wavelength resolution (~1-2 min). Set FAST=False to
reproduce the manuscript's actual N values (N=100 disorder realizations,
15 angles, etc.) -- this will take much longer and is meant to be run
before final figure generation, not during development.
"""

import numpy as np
import matplotlib.pyplot as plt
import peacock_optics as po

FAST = False  # flip to True only for quick dev checks (~1-2 min)

REGION4 = po.REGIONS["R4"]

if FAST:
    N_WL = 120
    N_DISORDER = 20
    N_ANGLES_ORDERED = 7
    N_ANGLES_EMT = 5
    N_SIGMA_STEPS = 6
    N_REAL_TRADEOFF = 6
else:
    N_WL = 351
    N_DISORDER = 100
    N_ANGLES_ORDERED = 15
    N_ANGLES_EMT = 5
    N_SIGMA_STEPS = 10
    N_REAL_TRADEOFF = 15

WAVELENGTHS = np.linspace(400, 750, N_WL)
rng_global = np.random.default_rng(0)


# =============================================================================
# Figure 1: ordered depth profile + reflectance (Region 4)
# =============================================================================

def figure1():
    wl_ref = 550.0
    fk, fm, fa = po.period_averaged_fractions(REGION4)
    nTE = po.n_eff_TE(fk, fm, fa, wl_ref)

    z = np.arange(0, REGION4["Nm"] * REGION4["a"], 1.0)
    n_real = np.where(True, np.real(nTE), np.real(nTE)) * np.ones_like(z)  # flat continuum ref
    # depth-resolved profile (matches original Fig 1's quasi-periodic look)
    n_profile = []
    for zi in z:
        fk_z, fm_z, fa_z = po.depth_resolved_fractions(zi, REGION4["a"], REGION4["b"],
                                                          REGION4["Dm"], REGION4["Da"])
        n_profile.append(po.n_eff_TE(fk_z, fm_z, fa_z, wl_ref))
    n_profile = np.array(n_profile)

    R_spec = np.array([po.reflectance(po.build_ordered_stack(REGION4, wl), wl)
                        for wl in WAVELENGTHS])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(z, np.real(n_profile), 'b-')
    axes[0].set_xlabel("Depth (nm)"); axes[0].set_ylabel("Refractive index (real part)")
    axes[0].set_title(f"Refractive index profile\n(λ = {wl_ref:.0f} nm)")

    axes[1].plot(z, -np.imag(n_profile), 'r-')  # extinction coefficient, plotted positive
    axes[1].set_xlabel("Depth (nm)"); axes[1].set_ylabel("Extinction coefficient (dimensionless)")
    axes[1].set_title(f"Extinction coefficient profile\n(λ = {wl_ref:.0f} nm)")

    axes[2].plot(WAVELENGTHS, R_spec, 'k-')
    axes[2].set_xlabel("Wavelength (nm)"); axes[2].set_ylabel("Reflectance")
    axes[2].set_title("Reflectance spectrum (Region 4, ordered)")
    axes[2].set_ylim(0, max(0.7, R_spec.max() * 1.1))

    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig1_ordered_profile.png", dpi=200)
    plt.close(fig)
    print(f"Fig 1 saved. Peak R = {R_spec.max():.3f} at {WAVELENGTHS[np.argmax(R_spec)]:.0f} nm")
    return R_spec


# =============================================================================
# Figure 2: disorder-stabilized reflectance (BUG 2 fixed: per-layer disorder)
# =============================================================================

def figure2(sigma=0.05):
    R_ordered = np.array([po.reflectance(po.build_ordered_stack(REGION4, wl), wl)
                           for wl in WAVELENGTHS])

    all_real = np.zeros((N_DISORDER, N_WL))
    for r in range(N_DISORDER):
        rng = np.random.default_rng(1000 + r)
        for i, wl in enumerate(WAVELENGTHS):
            stack = po.build_layered_disordered_stack(REGION4, wl, sigma, rng)
            all_real[r, i] = po.reflectance(stack, wl)
        # NOTE: rng re-seeded per wavelength above would break per-layer
        # consistency across the spectrum; instead each realization uses one
        # rng advanced across all Nm*len(wavelengths) draws, so the SAME
        # per-layer disorder pattern would ideally be reused across
        # wavelength if you want strict "one structure, one spectrum".
        # Region 4 has dispersion-dependent index but the SAME geometric
        # disorder per layer; see figure2_strict() below for that version.

    R_avg = all_real.mean(axis=0)

    plt.figure(figsize=(8, 5))
    for r in range(min(15, N_DISORDER)):
        plt.plot(WAVELENGTHS, 100 * all_real[r], color='gray', alpha=0.15, lw=0.5)
    plt.plot(WAVELENGTHS, 100 * R_avg, 'b-', lw=3, label=f"Ensemble avg (N={N_DISORDER})")
    plt.plot(WAVELENGTHS, 100 * R_ordered, 'r--', lw=2.5, label="Ordered reference")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance (%)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig2_disorder_stabilized.png", dpi=200)
    plt.close()
    print(f"Fig 2 saved. Ordered peak={100*R_ordered.max():.1f}%, "
          f"ensemble peak={100*R_avg.max():.1f}%")
    return R_ordered, R_avg


def figure2_strict(sigma=0.05):
    """
    Stricter version: ONE set of per-layer diameter draws per realization,
    reused (geometrically) across the full spectrum -- i.e. the disorder
    pattern is fixed per realization, only the material dispersion changes
    with wavelength. This is the version to use for any quantitative claim
    about a single "realization spectrum" (relevant to BUG 5's lesson:
    don't redraw structure inside the wavelength loop).
    """
    R_ordered = np.array([po.reflectance(po.build_ordered_stack(REGION4, wl), wl)
                           for wl in WAVELENGTHS])
    all_real = np.zeros((N_DISORDER, N_WL))
    A_cell = REGION4["a"] * REGION4["b"]

    for r in range(N_DISORDER):
        rng = np.random.default_rng(2000 + r)
        Dm_layers = REGION4["Dm"] * (1 + sigma * rng.standard_normal(REGION4["Nm"]))
        Da_layers = REGION4["Da"] * (1 + sigma * rng.standard_normal(REGION4["Nm"]))
        for i, wl in enumerate(WAVELENGTHS):
            stack = po.build_layered_stack_from_diameters(REGION4, wl, Dm_layers, Da_layers)
            all_real[r, i] = po.reflectance(stack, wl)

    R_avg = all_real.mean(axis=0)
    print(f"Fig 2 (strict, fixed-structure-per-realization): ensemble peak={100*R_avg.max():.1f}%")
    return R_ordered, R_avg


# =============================================================================
# Figure 3: continuum EMT reflectance
# =============================================================================

def figure3():
    R = np.array([po.reflectance(po.build_continuum_stack(REGION4, wl), wl)
                   for wl in WAVELENGTHS])
    plt.figure(figsize=(7, 4))
    plt.plot(WAVELENGTHS, R, lw=2)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance")
    plt.ylim(0, max(0.3, R.max() * 1.1))
    plt.tight_layout()
    plt.savefig("fig3_continuum_emt.png", dpi=200)
    plt.close()
    print(f"Fig 3 saved. Peak R = {R.max():.3f}")
    return R


# =============================================================================
# Figure 4: angle-averaged continuum (BUG 3 fixed: no undisclosed grading)
# =============================================================================

def figure4():
    angles = np.linspace(0, 40, N_ANGLES_EMT)
    R = np.zeros(N_WL)
    for i, wl in enumerate(WAVELENGTHS):
        stack = po.build_continuum_stack(REGION4, wl)  # same continuum stack as Fig 3
        R[i] = np.mean([po.reflectance(stack, wl, theta_deg=th) for th in angles])

    plt.figure(figsize=(7, 4))
    plt.plot(WAVELENGTHS, R, lw=2.5)
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance")
    plt.ylim(0, max(0.3, R.max() * 1.1))
    plt.tight_layout()
    plt.savefig("fig4_angle_averaged.png", dpi=200)
    plt.close()
    print(f"Fig 4 saved. Peak R = {R.max():.3f}")
    return R


# =============================================================================
# Figure 5: chromaticity stabilization
# =============================================================================

def figure5():
    angles = np.linspace(0, 60, N_ANGLES_ORDERED)
    xy_ordered = []
    for th in angles:
        R = np.array([po.reflectance(po.build_ordered_stack(REGION4, wl), wl, theta_deg=th)
                       for wl in WAVELENGTHS])
        xy_ordered.append(po.spectrum_to_xy(R, WAVELENGTHS))
    xy_ordered = np.array(xy_ordered)

    xy_ens = []
    for r in range(N_DISORDER):
        rng = np.random.default_rng(3000 + r)
        A_cell = REGION4["a"] * REGION4["b"]
        Dm_layers = REGION4["Dm"] * (1 + 0.05 * rng.standard_normal(REGION4["Nm"]))
        Da_layers = REGION4["Da"] * (1 + 0.05 * rng.standard_normal(REGION4["Nm"]))
        R = np.zeros(N_WL)
        for i, wl in enumerate(WAVELENGTHS):
            stack = po.build_layered_stack_from_diameters(REGION4, wl, Dm_layers, Da_layers)
            R[i] = po.reflectance(stack, wl)
        xy_ens.append(po.spectrum_to_xy(R, WAVELENGTHS))
    xy_ens = np.array(xy_ens)

    # EMT continuum: previously only ONE (angle-averaged) chromaticity point
    # was computed here, but Sec. 3.5 of the manuscript quotes a standard
    # deviation "under angular variation" for the continuum specifically.
    # That number cannot come from a single point -- it requires sweeping
    # the continuum stack over the SAME angle range used for the ordered
    # case and computing its own per-angle chromaticity, exactly as done
    # for xy_ordered above. Fixed below; this was a genuine gap, not just
    # a reporting issue, since no code previously produced the quantity
    # the manuscript text reports.
    xy_emt = []
    for th in angles:
        R = np.array([po.reflectance(po.build_continuum_stack(REGION4, wl), wl, theta_deg=th)
                       for wl in WAVELENGTHS])
        xy_emt.append(po.spectrum_to_xy(R, WAVELENGTHS))
    xy_emt = np.array(xy_emt)
    x_emt, y_emt = xy_emt.mean(axis=0)  # plotted point = angle-averaged chromaticity

    plt.figure(figsize=(6, 6))
    plt.scatter(xy_ordered[:, 0], xy_ordered[:, 1], c=angles, cmap='viridis',
                s=60, edgecolors='k', label="Ordered multilayer (θ=0°-60°)")
    plt.scatter(xy_ens[:, 0], xy_ens[:, 1], c='orange', alpha=0.6, s=30,
                edgecolors='k', label=f"Disordered ensemble (n={N_DISORDER})")
    plt.scatter(xy_emt[:, 0], xy_emt[:, 1], marker='*', s=120, c='red', edgecolors='k',
                label="EMT continuum (θ=0°-60°)")
    plt.xlabel("CIE 1931 x"); plt.ylabel("CIE 1931 y")
    plt.legend(); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("fig5_chromaticity_stabilization.png", dpi=200)
    plt.close()

    # Three DISTINCT variances -- do not conflate them. Report all three
    # explicitly since the manuscript text quotes what is actually the
    # continuum's angular std, not the ordered multilayer's, and the two
    # were being visually/numerically confused in the previous draft.
    ordered_angle_var = np.std(xy_ordered, axis=0)   # ordered stack, swept 0-60 deg
    ens_var = np.std(xy_ens, axis=0)                  # disorder ensemble, fixed theta=0
    emt_angle_var = np.std(xy_emt, axis=0)            # continuum stack, swept 0-60 deg

    # Single, unambiguous stabilization factor: how much smaller is the
    # continuum's OWN angular chromaticity spread than the ordered
    # multilayer's angular spread (the comparison Sec. 3.5's narrative
    # is actually about). This is independent of, and should not be
    # confused with, the disorder-ensemble spread ens_var above.
    S = np.linalg.norm(emt_angle_var) / np.linalg.norm(ordered_angle_var)

    print(f"Fig 5 saved.")
    print(f"  Ordered multilayer, angle-variation std (0-60 deg)   = {ordered_angle_var}")
    print(f"  Disordered ensemble, realization-variation std       = {ens_var}")
    print(f"  EMT continuum, angle-variation std (0-60 deg)        = {emt_angle_var}")
    print(f"  Stabilization factor S = ||emt_angle_std|| / ||ordered_angle_std|| = {S:.4f}")
    return xy_ordered, xy_ens, xy_emt


# =============================================================================
# Figure 6: region color palette (BUG 4 fixed: normalized luminance;
# modeled regions never silently fall back to hard-coded values)
# =============================================================================

def figure6():
    results = {}
    for key in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        params = po.REGIONS[key]
        if key in ("R2", "R4"):
            sigma = 0.05 if key == "R4" else 0.03
            n_ens = N_DISORDER if key == "R4" else max(10, N_DISORDER // 3)
            xs, ys = [], []
            R_sum = np.zeros(N_WL)
            for r in range(n_ens):
                rng = np.random.default_rng(hash((key, r)) % (2**32))
                R = np.zeros(N_WL)
                A_cell = params["a"] * params["b"]
                Dm_layers = params["Dm"] * (1 + sigma * rng.standard_normal(params["Nm"]))
                Da_layers = params["Da"] * (1 + sigma * rng.standard_normal(params["Nm"]))
                for i, wl in enumerate(WAVELENGTHS):

                    stack = po.build_layered_stack_from_diameters(params, wl, Dm_layers, Da_layers)
                    R[i] = po.reflectance(stack, wl)
                R_sum += R
                x, y = po.spectrum_to_xy(R, WAVELENGTHS)
                xs.append(x); ys.append(y)
            R_mean_spectrum = R_sum / n_ens
            rgb = po.spectrum_to_srgb(R_mean_spectrum, WAVELENGTHS)
            source = f"Computed (disorder-averaged, N={n_ens})"
        else:
            hexcode = po.OBSERVED_COLORS_OUTSIDE_MODEL_SCOPE[key]
            rgb = tuple(int(hexcode[i:i+2], 16) / 255 for i in (1, 3, 5))
            source = "Observed (outside multilayer-EMT model scope, per Sec. 3.6)"

        results[key] = dict(name=params["name"], rgb=rgb, hex=po.srgb_to_hex(rgb), source=source)
        print(f"{key}: {source} -> {results[key]['hex']}")

    fig, ax = plt.subplots(figsize=(10, 2))
    for i, key in enumerate(["R1", "R2", "R3", "R4", "R5", "R6"]):
        r = results[key]
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=r["rgb"]))
        ax.text(i + 0.5, -0.15, key, ha='center', va='top', fontsize=10)
        ax.text(i + 0.5, 1.1, r["hex"], ha='center', va='bottom', fontsize=8)
    ax.set_xlim(0, 6); ax.set_ylim(-0.3, 1.3); ax.axis('off')
    plt.tight_layout()
    plt.savefig("fig6_region_palette.png", dpi=200)
    plt.close()
    return results


# =============================================================================
# Figure 7: disorder-iridescence tradeoff (BUG 5 fixed: one structure per
# realization, reused across the whole spectrum and all angles)
# =============================================================================

def disorder_iridescence_tradeoff():
    wl_tradeoff = np.linspace(420, 720, min(N_WL, 150))
    angles = np.linspace(0, 40, 9)
    sigmas = np.linspace(0.005, 0.15, N_SIGMA_STEPS)
    A_cell = REGION4["a"] * REGION4["b"]

    def spectrum_for_fixed_structure(Dm_layers, Da_layers, theta_deg):
        R = np.zeros(len(wl_tradeoff))
        for i, wl in enumerate(wl_tradeoff):
            stack = po.build_layered_stack_from_diameters(REGION4, wl, Dm_layers, Da_layers)
            R[i] = po.reflectance(stack, wl, theta_deg=theta_deg)
        return R

    eta_vals = []
    for sigma in sigmas:
        spreads = []
        for r in range(N_REAL_TRADEOFF):
            rng = np.random.default_rng(hash((sigma, r)) % (2**32))
            # BUG 5 FIX: draw diameters ONCE per realization, reuse for
            # every angle and every wavelength -- not redrawn per wavelength.
            Dm_layers = REGION4["Dm"] * (1 + sigma * rng.standard_normal(REGION4["Nm"]))
            Da_layers = REGION4["Da"] * (1 + sigma * rng.standard_normal(REGION4["Nm"]))

            xy_theta = np.array([
                po.spectrum_to_xy(spectrum_for_fixed_structure(Dm_layers, Da_layers, th),
                                   wl_tradeoff)
                for th in angles
            ])
            angular_spread = np.mean(np.linalg.norm(xy_theta - xy_theta.mean(axis=0), axis=1))
            spreads.append(angular_spread)

        # ensemble (disorder) spread at fixed angle theta=0, one structure
        # per realization (same fix)
        xy_ensemble = []
        for r in range(N_REAL_TRADEOFF):
            rng = np.random.default_rng(hash((sigma, "ens", r)) % (2**32))
            Dm_layers = REGION4["Dm"] * (1 + sigma * rng.standard_normal(REGION4["Nm"]))
            Da_layers = REGION4["Da"] * (1 + sigma * rng.standard_normal(REGION4["Nm"]))
            xy_ensemble.append(po.spectrum_to_xy(
                spectrum_for_fixed_structure(Dm_layers, Da_layers, 0.0), wl_tradeoff))
        xy_ensemble = np.array(xy_ensemble)
        ensemble_spread = max(np.mean(np.linalg.norm(
            xy_ensemble - xy_ensemble.mean(axis=0), axis=1)), 1e-6)

        # CANONICAL DEFINITION (matches manuscript Sec. 2.5's Algorithm 1 /
        # eta(sigma) = std_theta(x,y) / std_realizations(x,y) formula only):
        #   numerator   = mean angular chromaticity spread, this sigma
        #   denominator = disorder-realization chromaticity spread, theta=0
        # The alternate "spectral peak spread" denominator quoted in
        # Sec. 3.7's prose does not correspond to any quantity computed
        # here or anywhere else in this script -- that sentence in the
        # manuscript should be deleted/replaced with this formula rather
        # than reconciled by changing the code.
        eta_vals.append(np.mean(spreads) / ensemble_spread)

    eta_vals = np.array(eta_vals)

    plt.figure(figsize=(5, 4))
    plt.plot(sigmas, eta_vals, '-o', lw=2.5)
    plt.xlabel(r"Normalized disorder strength, $\sigma_D/D$")
    plt.ylabel(r"Relative perceptual iridescence, $\eta$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig7_disorder_iridescence_tradeoff.png", dpi=200)
    plt.close()

    # Diagnostic: fit eta ~ A/sigma + B to check whether the curve is still
    # dominated by a 1/sigma artifact (as Reviewer 2 found in the buggy
    # version) or reflects genuine saturating physics.
    A_fit = np.polyfit(1.0 / sigmas, eta_vals, 1)
    pred = np.polyval(A_fit, 1.0 / sigmas)
    ss_res = np.sum((eta_vals - pred) ** 2)
    ss_tot = np.sum((eta_vals - eta_vals.mean()) ** 2)
    r2_1_over_sigma = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"Fig 7 saved. eta range: {eta_vals.min():.3f}-{eta_vals.max():.3f}")
    print(f"  Fit against 1/sigma: eta ~ {A_fit[0]:.4f}/sigma + {A_fit[1]:.4f}, "
          f"R^2 = {r2_1_over_sigma:.3f}")
    print("  (High R^2 here would indicate the curve is still dominated by a")
    print("   1/sigma normalization artifact rather than genuine physics --")
    print("   check this before treating Fig. 7 as a validated result.)")
    return sigmas, eta_vals, r2_1_over_sigma


# =============================================================================
# Coherent / incoherent split (new, addresses Reviewer 2's anisotropy/
# diffuse-channel critique) -- run at a representative sigma
# =============================================================================

def coherent_incoherent_check(sigma=0.05, theta_deg=0.0, N=None):
    N = N or N_DISORDER
    coherent_R, total_R, diffuse_R = po.ensemble_coherent_incoherent(
        REGION4, WAVELENGTHS, sigma=sigma, theta_deg=theta_deg, N=N, seed=42)

    plt.figure(figsize=(7, 4))
    plt.plot(WAVELENGTHS, coherent_R, label="Coherent |<r>|^2")
    plt.plot(WAVELENGTHS, total_R, label="Total <|r|^2>")
    plt.plot(WAVELENGTHS, diffuse_R, label="Diffuse (total - coherent)")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_coherent_incoherent_split.png", dpi=200)
    plt.close()
    print(f"Coherent/incoherent split saved. Mean diffuse fraction = "
          f"{np.mean(diffuse_R / np.maximum(total_R, 1e-9)):.3f}")
    return coherent_R, total_R, diffuse_R


# =============================================================================
if __name__ == "__main__":
    print(f"Running in {'FAST (dev/check)' if FAST else 'FULL (manuscript-resolution)'} mode\n")
    figure1()
    figure2_strict()
    figure3()
    figure4()
    figure5()
    figure6()
    disorder_iridescence_tradeoff()
    coherent_incoherent_check()
    print("\nAll figures generated.")
