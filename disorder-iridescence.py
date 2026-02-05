import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ============================================================
# 1. MATERIAL MODELS (Freyer 2020)
# ============================================================

def n_keratin(wl_nm):
    return 1.532 + 5890.0 / wl_nm**2

def n_melanin_complex(wl_nm):
    n = 1.648 + 23700.0 / wl_nm**2
    k = 0.56 * np.exp(-wl_nm / 270.0)
    return n + 1j * k

def n_eff_TE(fk, fm, fa, wl_nm):
    w = 2.5
    eps = (
        fk * n_keratin(wl_nm)**(-2*w) +
        fm * n_melanin_complex(wl_nm)**(-2*w) +
        fa * 1.0
    ) ** (-1/w)
    return np.sqrt(eps)

def n_eff_TM(fk, fm, fa, wl_nm):
    w = -1.5
    eps = (
        fk * n_keratin(wl_nm)**(-2*w) +
        fm * n_melanin_complex(wl_nm)**(-2*w) +
        fa * 1.0
    ) ** (-1/w)
    return np.sqrt(eps)

# ============================================================
# 2. GEOMETRY (Region 4)
# ============================================================

params = dict(
    a=195, b=150,
    Dm=120, Da=55,
    Nm=5, cortex=130
)

A_cell = params["a"] * params["b"]

# ============================================================
# 3. CIE 1931 COLOUR MATCHING FUNCTIONS
# ============================================================
cmf_data = """
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


cmf = np.loadtxt(StringIO(cmf_data))
wl_cmf, xbar, ybar, zbar = cmf.T

def spectrum_to_xy(R, wl):
    xb = np.interp(wl, wl_cmf, xbar)
    yb = np.interp(wl, wl_cmf, ybar)
    zb = np.interp(wl, wl_cmf, zbar)
    X = np.trapezoid(R * xb, wl)
    Y = np.trapezoid(R * yb, wl)
    Z = np.trapezoid(R * zb, wl)
    return X/(X+Y+Z), Y/(X+Y+Z)

# ============================================================
# 4. TRANSFER MATRIX (OBLIQUE, TE+TM)
# ============================================================

def reflectance(stack, wl_nm, theta_deg):
    theta = np.radians(theta_deg)
    k0 = 2*np.pi / (wl_nm*1e-9)

    R_pol = []
    for pol in ["TE","TM"]:
        M = np.eye(2, dtype=complex)
        for (nTE,nTM), d in stack:
            n = nTE if pol=="TE" else nTM
            ct = np.sqrt(1 - (np.sin(theta)/np.real(n))**2 + 0j)
            delta = k0 * n * d * ct
            eta = n*ct if pol=="TE" else n/ct
            M_layer = np.array([
                [np.cos(delta), 1j*np.sin(delta)/eta],
                [1j*eta*np.sin(delta), np.cos(delta)]
            ])
            M = M_layer @ M
        Y = (M[1,0]+M[1,1])/(M[0,0]+M[0,1])
        r = (1-Y)/(1+Y)
        R_pol.append(np.abs(r)**2)
    return np.mean(R_pol)

# ============================================================
# 5. BUILD DISORDERED STACK (REALIZATION)
# ============================================================

def build_stack(wl, sigma):
    Dm = params["Dm"] * (1 + sigma*np.random.randn())
    Da = params["Da"] * (1 + sigma*np.random.randn())

    fm = np.pi*(Dm/2)**2 / A_cell
    fa = np.pi*(Da/2)**2 / A_cell
    fk = 1 - fm - fa

    nTE = n_eff_TE(fk,fm,fa,wl)
    nTM = n_eff_TM(fk,fm,fa,wl)

    stack = [((n_keratin(wl),)*2, params["cortex"]*1e-9)]
    for _ in range(params["Nm"]):
        stack.append(((nTE,nTM), params["a"]*1e-9))
    return stack

# ============================================================
# 6. DISORDER–IRIDESCENCE TRADEOFF
# ============================================================

wavelengths = np.linspace(420,720,250)
angles = np.linspace(0,40,9)
sigmas = np.linspace(0.005,0.15,10)
N_real = 15

Iridescence = []

for sigma in sigmas:
    xy_all = []
    for _ in range(N_real):
        xy_theta = []
        for th in angles:
            R = []
            for wl in wavelengths:
                stack = build_stack(wl, sigma)
                R.append(reflectance(stack, wl, th))
            R = np.array(R)
            xy_theta.append(spectrum_to_xy(R, wavelengths))
        xy_theta = np.array(xy_theta)
        #ensemble (disorder-induced) chromaticity spread 
        xy_ensemble = []
        for _ in range(len(xy_theta)):
            R = []
            for wl in wavelengths:
                stack = build_stack(wl, sigma)
                R.append(reflectance(stack, wl, 0))  # fixed angle
            xy_ensemble.append(spectrum_to_xy(np.array(R), wavelengths))
        xy_ensemble = np.array(xy_ensemble)

        mean_xy = xy_theta.mean(axis=0)
        angular_spread = np.mean(np.linalg.norm(xy_theta - mean_xy, axis=1))
        ensemble_spread = np.mean(
            np.linalg.norm(xy_ensemble - xy_ensemble.mean(axis=0), axis=1)
        )
        ensemble_spread = max(ensemble_spread, 1e-6)

        spread = angular_spread / ensemble_spread
        xy_all.append(spread)
    Iridescence.append(np.mean(xy_all))

Iridescence = np.array(Iridescence)


# ============================================================
# 7. PUBLICATION-READY PLOT 
# ============================================================

plt.figure(figsize=(4.5,3.5))
plt.plot(sigmas, Iridescence, '-o', lw=2.8, ms=7, color='#1f77b4')
plt.xlabel(r"Normalized disorder strength, $\sigma_D/D$", fontsize=12)
plt.ylabel(r"Relative perceptual iridescence, $\eta$", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("disorder-iridescence-tradeoff.png", dpi=300)
plt.show()


# ============================================================
# 8. SENSITIVITY ANALYSIS (ROBUSTNESS CHECK)
# ============================================================

print("\nRunning sensitivity analysis for disorder–iridescence trade-off...")

# --- Sensitivity configurations ---
angles_alt = np.linspace(0, 30, 7)      # reduced angular range
N_real_alt = 8                           # fewer disorder realizations

Iridescence_angle_alt = []
Iridescence_Nreal_alt = []

# -------- Sensitivity 1: Angular range --------
for sigma in sigmas:
    xy_all = []
    for _ in range(N_real):
        xy_theta = []
        for th in angles_alt:
            R = []
            for wl in wavelengths:
                stack = build_stack(wl, sigma)
                R.append(reflectance(stack, wl, th))
            xy_theta.append(spectrum_to_xy(np.array(R), wavelengths))
        xy_theta = np.array(xy_theta)

        mean_xy = xy_theta.mean(axis=0)
        angular_spread = np.mean(np.linalg.norm(xy_theta - mean_xy, axis=1))

        # ensemble spread (fixed angle)
        xy_ensemble = []
        for _ in range(len(xy_theta)):
            R = []
            for wl in wavelengths:
                stack = build_stack(wl, sigma)
                R.append(reflectance(stack, wl, 0))
            xy_ensemble.append(spectrum_to_xy(np.array(R), wavelengths))
        xy_ensemble = np.array(xy_ensemble)

        ensemble_spread = np.mean(
            np.linalg.norm(xy_ensemble - xy_ensemble.mean(axis=0), axis=1)
        )
        ensemble_spread = max(ensemble_spread, 1e-6)

        xy_all.append(angular_spread / ensemble_spread)

    Iridescence_angle_alt.append(np.mean(xy_all))

Iridescence_angle_alt = np.array(Iridescence_angle_alt)

# -------- Sensitivity 2: Ensemble size --------
for sigma in sigmas:
    xy_all = []
    for _ in range(N_real_alt):
        xy_theta = []
        for th in angles:
            R = []
            for wl in wavelengths:
                stack = build_stack(wl, sigma)
                R.append(reflectance(stack, wl, th))
            xy_theta.append(spectrum_to_xy(np.array(R), wavelengths))
        xy_theta = np.array(xy_theta)

        mean_xy = xy_theta.mean(axis=0)
        angular_spread = np.mean(np.linalg.norm(xy_theta - mean_xy, axis=1))

        xy_ensemble = []
        for _ in range(len(xy_theta)):
            R = []
            for wl in wavelengths:
                stack = build_stack(wl, sigma)
                R.append(reflectance(stack, wl, 0))
            xy_ensemble.append(spectrum_to_xy(np.array(R), wavelengths))
        xy_ensemble = np.array(xy_ensemble)

        ensemble_spread = np.mean(
            np.linalg.norm(xy_ensemble - xy_ensemble.mean(axis=0), axis=1)
        )
        ensemble_spread = max(ensemble_spread, 1e-6)

        xy_all.append(angular_spread / ensemble_spread)

    Iridescence_Nreal_alt.append(np.mean(xy_all))

Iridescence_Nreal_alt = np.array(Iridescence_Nreal_alt)

# ============================================================
# 9. PUBLICATION-READY SENSITIVITY FIGURE  
# ============================================================

plt.figure(figsize=(4.8, 3.6))

plt.plot(sigmas, Iridescence, '-o', lw=2.8, ms=7,
         label='Main result', color='#1f77b4')

plt.plot(sigmas, Iridescence_angle_alt, '--s', lw=2.2, ms=6,
         label='Reduced angular range (0–30°)', color='#ff7f0e')

plt.plot(sigmas, Iridescence_Nreal_alt, ':^', lw=2.2, ms=6,
         label='Reduced ensemble size (N=8)', color='#2ca02c')

plt.xlabel(r"Normalized disorder strength, $\sigma_D/D$", fontsize=12)
plt.ylabel(r"Relative perceptual iridescence, $\eta$", fontsize=12)

plt.legend(frameon=True, fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("disorder-iridescence-sensitivity.png", dpi=300, bbox_inches='tight')
plt.show()

print("Sensitivity analysis completed.")
print("Figure saved as 'disorder-iridescence-sensitivity.png'")
