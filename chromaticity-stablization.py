import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# ============================================================
# Helper: spectrum to chromaticity
# ============================================================

def spectrum_to_xy(R, wavelengths, xbar, ybar, zbar):
    X = np.trapezoid(R * xbar, wavelengths)
    Y = np.trapezoid(R * ybar, wavelengths)
    Z = np.trapezoid(R * zbar, wavelengths)
    return X / (X + Y + Z), Y / (X + Y + Z)

# ============================================================
# CIE 1931 CMFs (as used in Methods)
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

# ============================================================
# Wavelength grid (Same as Steps 1-4)
# ============================================================

wavelengths = np.linspace(400, 750, 400)

xbar_i = np.interp(wavelengths, wl_cmf, xbar)
ybar_i = np.interp(wavelengths, wl_cmf, ybar)
zbar_i = np.interp(wavelengths, wl_cmf, zbar)

# ============================================================
# COPY MATERIAL MODELS FROM STEP 1 (for completeness)
# ============================================================

def n_keratin(wl_nm):
    return 1.532 + 5890.0 / wl_nm**2

def n_melanin_complex(wl_nm):
    n = 1.648 + 23700.0 / wl_nm**2
    k = 0.56 * np.exp(-wl_nm / 270.0)
    return n + 1j * k

def effective_refractive_index_TE(fk, fm, fa, wl_nm):
    w = 2.5  # Freyer-adjusted TE
    eps_k = n_keratin(wl_nm)**2
    eps_m = n_melanin_complex(wl_nm)**2
    eps_a = 1.0
    eps_eff = (
        fk * eps_k**(-w) +
        fm * eps_m**(-w) +
        fa * eps_a**(-w)
    ) ** (-1.0 / w)
    return np.sqrt(eps_eff)

def effective_refractive_index_TM(fk, fm, fa, wl_nm):
    w = -1.5  # Freyer-adjusted TM
    eps_k = n_keratin(wl_nm)**2
    eps_m = n_melanin_complex(wl_nm)**2
    eps_a = 1.0
    eps_eff = (
        fk * eps_k**(-w) +
        fm * eps_m**(-w) +
        fa * eps_a**(-w)
    ) ** (-1.0 / w)
    return np.sqrt(eps_eff)

# ============================================================
# REGION 4 PARAMETERS (Freyer Table 1)
# ============================================================

params = {
    "a": 195,      # nm, period
    "b": 150,      # nm
    "c": 130,      # nm, cortex thickness
    "Dm": 120,     # nm, melanosome diameter
    "Da": 55,      # nm, air channel diameter
    "Nm": 5        # number of periods
}

# ============================================================
# STEP 1: Ordered multilayer calculation
# ============================================================

def compute_ordered_multilayer_spectra(angles):
    """Calculate reflectance spectra for ordered multilayer at different angles"""
    N_angles = len(angles)
    R_ordered = np.zeros((N_angles, len(wavelengths)))
    
    for angle_idx, theta_deg in enumerate(angles):
        print(f"  Computing ordered spectrum at {theta_deg:.1f}° ({angle_idx+1}/{N_angles})")
        
        for wl_idx, wl in enumerate(wavelengths):
            # Build stack (simplified for demonstration)
            stack = []
            
            # Cortex layer
            n_k = n_keratin(wl)
            stack.append(((n_k, n_k), params["c"] * 1e-9))
            
            # Calculate effective refractive index for one period
            A_cell = params["a"] * params["b"]
            f_m = np.pi * (params["Dm"]/2)**2 / A_cell
            f_a = np.pi * (params["Da"]/2)**2 / A_cell
            f_k = 1.0 - f_m - f_a
            
            n_TE = effective_refractive_index_TE(f_k, f_m, f_a, wl)
            n_TM = effective_refractive_index_TM(f_k, f_m, f_a, wl)
            
            # Add Nm periods
            for _ in range(params["Nm"]):
                stack.append(((n_TE, n_TM), params["a"] * 1e-9))
            
            # Calculate reflectance at this angle (simplified TMM)
            theta = np.radians(theta_deg)
            k0 = 2 * np.pi / (wl * 1e-9)
            
            # TE calculation
            M_TE = np.eye(2, dtype=complex)
            for (n_TE_layer, n_TM_layer), d in stack:
                cos_t = np.sqrt(1 - (np.sin(theta)/np.real(n_TE_layer))**2 + 0j)
                delta = k0 * n_TE_layer * d * cos_t
                eta = n_TE_layer * cos_t
                M_layer = np.array([
                    [np.cos(delta), 1j*np.sin(delta)/eta],
                    [1j*eta*np.sin(delta), np.cos(delta)]
                ])
                M_TE = M_layer @ M_TE
            
            # TM calculation  
            M_TM = np.eye(2, dtype=complex)
            for (n_TE_layer, n_TM_layer), d in stack:
                cos_t = np.sqrt(1 - (np.sin(theta)/np.real(n_TM_layer))**2 + 0j)
                delta = k0 * n_TM_layer * d * cos_t
                eta = n_TM_layer / cos_t  # Different for TM
                M_layer = np.array([
                    [np.cos(delta), 1j*np.sin(delta)/eta],
                    [1j*eta*np.sin(delta), np.cos(delta)]
                ])
                M_TM = M_layer @ M_TM
            
            # Calculate reflectance and average TE/TM
            Y_TE = (M_TE[1,0] + M_TE[1,1]) / (M_TE[0,0] + M_TE[0,1])
            r_TE = (1 - Y_TE) / (1 + Y_TE)
            R_TE = np.abs(r_TE)**2
            
            Y_TM = (M_TM[1,0] + M_TM[1,1]) / (M_TM[0,0] + M_TM[0,1])
            r_TM = (1 - Y_TM) / (1 + Y_TM)
            R_TM = np.abs(r_TM)**2
            
            R_ordered[angle_idx, wl_idx] = (R_TE + R_TM) / 2
    
    return R_ordered

# ============================================================
# STEP 2: Disordered ensemble calculation
# ============================================================

def compute_disordered_ensemble_spectra(N_realizations, sigma_Dm=0.05, sigma_Da=0.05):
    """Calculate reflectance spectra for disordered ensemble"""
    R_ensemble = np.zeros((N_realizations, len(wavelengths)))
    
    for realization in range(N_realizations):
        print(f"  Computing disordered realization {realization+1}/{N_realizations}")
        
        # Apply disorder to parameters
        Dm_real = params["Dm"] * (1 + sigma_Dm * np.random.randn())
        Da_real = params["Da"] * (1 + sigma_Da * np.random.randn())
        
        for wl_idx, wl in enumerate(wavelengths):
            # Build stack with disordered parameters
            stack = []
            
            # Cortex layer
            n_k = n_keratin(wl)
            stack.append(((n_k, n_k), params["c"] * 1e-9))
            
            # Calculate effective refractive index with disordered parameters
            A_cell = params["a"] * params["b"]
            f_m = np.pi * (Dm_real/2)**2 / A_cell
            f_a = np.pi * (Da_real/2)**2 / A_cell
            f_k = 1.0 - f_m - f_a
            
            n_TE = effective_refractive_index_TE(f_k, f_m, f_a, wl)
            n_TM = effective_refractive_index_TM(f_k, f_m, f_a, wl)
            
            # Add Nm periods
            for _ in range(params["Nm"]):
                stack.append(((n_TE, n_TM), params["a"] * 1e-9))
            
            # Calculate reflectance at normal incidence
            k0 = 2 * np.pi / (wl * 1e-9)
            
            # TE calculation
            M_TE = np.eye(2, dtype=complex)
            for (n_TE_layer, n_TM_layer), d in stack:
                delta = k0 * n_TE_layer * d
                eta = n_TE_layer
                M_layer = np.array([
                    [np.cos(delta), 1j*np.sin(delta)/eta],
                    [1j*eta*np.sin(delta), np.cos(delta)]
                ])
                M_TE = M_layer @ M_TE
            
            # TM calculation
            M_TM = np.eye(2, dtype=complex)
            for (n_TE_layer, n_TM_layer), d in stack:
                delta = k0 * n_TM_layer * d
                eta = n_TM_layer
                M_layer = np.array([
                    [np.cos(delta), 1j*np.sin(delta)/eta],
                    [1j*eta*np.sin(delta), np.cos(delta)]
                ])
                M_TM = M_layer @ M_TM
            
            # Calculate reflectance and average TE/TM
            Y_TE = (M_TE[1,0] + M_TE[1,1]) / (M_TE[0,0] + M_TE[0,1])
            r_TE = (1 - Y_TE) / (1 + Y_TE)
            R_TE = np.abs(r_TE)**2
            
            Y_TM = (M_TM[1,0] + M_TM[1,1]) / (M_TM[0,0] + M_TM[0,1])
            r_TM = (1 - Y_TM) / (1 + Y_TM)
            R_TM = np.abs(r_TM)**2
            
            R_ensemble[realization, wl_idx] = (R_TE + R_TM) / 2
    
    return R_ensemble

# ============================================================
# STEP 3: EMT continuum calculation
# ============================================================

def compute_emt_continuum_spectrum():
    """Calculate EMT continuum spectrum (angle-averaged)"""
    print("  Computing EMT continuum spectrum")
    
    R_emt = np.zeros(len(wavelengths))
    
    # Average over angles for EMT
    angles_emt = np.linspace(0, 40, 7)
    
    for wl_idx, wl in enumerate(wavelengths):
        # Build stack
        stack = []
        
        # Cortex layer
        n_k = n_keratin(wl)
        stack.append(((n_k, n_k), params["c"] * 1e-9))
        
        # Use average fractions for EMT
        A_cell = params["a"] * params["b"]
        f_m = np.pi * (params["Dm"]/2)**2 / A_cell
        f_a = np.pi * (params["Da"]/2)**2 / A_cell
        f_k = 1.0 - f_m - f_a
        
        n_TE = effective_refractive_index_TE(f_k, f_m, f_a, wl)
        n_TM = effective_refractive_index_TM(f_k, f_m, f_a, wl)
        
        # Add Nm periods
        for _ in range(params["Nm"]):
            stack.append(((n_TE, n_TM), params["a"] * 1e-9))
        
        # Average over angles
        R_angles = []
        for theta_deg in angles_emt:
            theta = np.radians(theta_deg)
            k0 = 2 * np.pi / (wl * 1e-9)
            
            # TE calculation
            M_TE = np.eye(2, dtype=complex)
            for (n_TE_layer, n_TM_layer), d in stack:
                cos_t = np.sqrt(1 - (np.sin(theta)/np.real(n_TE_layer))**2 + 0j)
                delta = k0 * n_TE_layer * d * cos_t
                eta = n_TE_layer * cos_t
                M_layer = np.array([
                    [np.cos(delta), 1j*np.sin(delta)/eta],
                    [1j*eta*np.sin(delta), np.cos(delta)]
                ])
                M_TE = M_layer @ M_TE
            
            # TM calculation
            M_TM = np.eye(2, dtype=complex)
            for (n_TE_layer, n_TM_layer), d in stack:
                cos_t = np.sqrt(1 - (np.sin(theta)/np.real(n_TM_layer))**2 + 0j)
                delta = k0 * n_TM_layer * d * cos_t
                eta = n_TM_layer / cos_t  # Different for TM
                M_layer = np.array([
                    [np.cos(delta), 1j*np.sin(delta)/eta],
                    [1j*eta*np.sin(delta), np.cos(delta)]
                ])
                M_TM = M_layer @ M_TM
            
            # Calculate reflectance and average TE/TM
            Y_TE = (M_TE[1,0] + M_TE[1,1]) / (M_TE[0,0] + M_TE[0,1])
            r_TE = (1 - Y_TE) / (1 + Y_TE)
            R_TE = np.abs(r_TE)**2
            
            Y_TM = (M_TM[1,0] + M_TM[1,1]) / (M_TM[0,0] + M_TM[0,1])
            r_TM = (1 - Y_TM) / (1 + Y_TM)
            R_TM = np.abs(r_TM)**2
            
            R_angles.append((R_TE + R_TM) / 2)
        
        R_emt[wl_idx] = np.mean(R_angles)
    
    return R_emt

# ============================================================
# MAIN COMPUTATION
# ============================================================

print("=" * 60)
print("STEP 5: Chromaticity Stabilization Analysis")
print("=" * 60)
print(f"Wavelength grid: {len(wavelengths)} points, {wavelengths[0]:.0f}-{wavelengths[-1]:.0f} nm")
print()

# Configuration
N_angles = 15
N_realizations = 25
angles = np.linspace(0, 60, N_angles)

print("1. Computing ordered multilayer spectra...")
R_ordered = compute_ordered_multilayer_spectra(angles)

print("\n2. Computing disordered ensemble spectra...")
R_ensemble = compute_disordered_ensemble_spectra(N_realizations)

print("\n3. Computing EMT continuum spectrum...")
R_emt = compute_emt_continuum_spectrum()

print("\n4. Calculating chromaticities...")

# Calculate chromaticities
xy_angle = []
for i in range(N_angles):
    x, y = spectrum_to_xy(R_ordered[i], wavelengths, xbar_i, ybar_i, zbar_i)
    xy_angle.append((x, y))
xy_angle = np.array(xy_angle)

xy_ens = []
for i in range(N_realizations):
    x, y = spectrum_to_xy(R_ensemble[i], wavelengths, xbar_i, ybar_i, zbar_i)
    xy_ens.append((x, y))
xy_ens = np.array(xy_ens)

x_emt, y_emt = spectrum_to_xy(R_emt, wavelengths, xbar_i, ybar_i, zbar_i)

print("All computations completed successfully")
print()

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(6, 6))

# Plot angle variation with color gradient
colors = plt.cm.viridis(np.linspace(0, 1, N_angles))
for i in range(N_angles-1):
    plt.plot(
        [xy_angle[i,0], xy_angle[i+1,0]],
        [xy_angle[i,1], xy_angle[i+1,1]],
        color=colors[i], lw=2, alpha=0.7
    )

# Mark angle points
scatter_angles = plt.scatter(xy_angle[:,0], xy_angle[:,1], c=angles, cmap='viridis', 
                            s=80, edgecolors='k', linewidth=1.5, zorder=5,
                            label=f"Ordered multilayer (θ=0°–{angles[-1]:.0f}°)")

# Ensemble as cloud
plt.scatter(
    xy_ens[:, 0], xy_ens[:, 1],
    s=40, alpha=0.6, c='orange',
    edgecolors='k', linewidth=1,
    label=f"Disordered ensemble (n={N_realizations})"
)

# EMT as stable reference
plt.scatter(
    x_emt, y_emt,
    s=350, marker="*", zorder=10,
    c='red', edgecolors='k', linewidth=2.5,
    label="EMT continuum (angle-averaged)"
)

# Add angle labels
for i, theta in enumerate([0, 20, 40, 60]):
    if i < len(angles):
        plt.annotate(f"{theta:.0f}°", 
                    xy=(xy_angle[i,0], xy_angle[i,1]),
                    xytext=(5,5), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

plt.xlabel("CIE 1931 x", fontsize=18)
plt.ylabel("CIE 1931 y", fontsize=18)
#plt.title("Chromaticity Stabilization in Peacock Feathers\nRegion 4 (Freyer & Stavenga 2020)", fontsize=13, pad=15)

# Calculate chromaticity variation metrics
angle_variation = np.std(xy_angle, axis=0)
ensemble_variation = np.std(xy_ens, axis=0)

plt.text(0.02, 0.98, 
         f"Angle variation: $\\sigma_x$={angle_variation[0]:.4f}, $\\sigma_y$={angle_variation[1]:.4f}\n"
         f"Ensemble variation: $\\sigma_x$={ensemble_variation[0]:.4f}, $\\sigma_y$={ensemble_variation[1]:.4f}",
         transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.grid(alpha=0.2, linestyle='--')
plt.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=12)
plt.xlim(0.15, 0.30)
plt.ylim(0.15, 0.30)
plt.tight_layout()

# Save figure
plt.savefig("chromaticity-stabilization.png", dpi=300, bbox_inches='tight')
print("\nFigure saved as 'chromaticity-stabilization.png'")

plt.show()
######################################
######################################
######################################

# Spectral peak spread (ordered case)
peak_wavelengths = []
for i in range(N_angles):
    peak_idx = np.argmax(R_ordered[i])
    peak_wavelengths.append(wavelengths[peak_idx])

spectral_spread = np.std(peak_wavelengths)  # nm

# Chromaticity spread (already computed)
chromaticity_spread = np.mean(angle_variation)

PSE = chromaticity_spread / spectral_spread


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"EMT continuum chromaticity: x = {x_emt:.4f}, y = {y_emt:.4f}")
print(f"Angle variation range: x = [{xy_angle[:,0].min():.4f}, {xy_angle[:,0].max():.4f}]")
print(f"                     y = [{xy_angle[:,1].min():.4f}, {xy_angle[:,1].max():.4f}]")
print(f"Ensemble variation range: x = [{xy_ens[:,0].min():.4f}, {xy_ens[:,0].max():.4f}]")
print(f"                       y = [{xy_ens[:,1].min():.4f}, {xy_ens[:,1].max():.4f}]")

# Calculate stabilization factor
angle_span = np.max(np.linalg.norm(xy_angle - xy_angle.mean(axis=0), axis=1))
ensemble_span = np.max(np.linalg.norm(xy_ens - xy_ens.mean(axis=0), axis=1))
stabilization_factor = angle_span / ensemble_span if ensemble_span > 0 else 0

print(f"\nChromaticity stabilization factor: {stabilization_factor:.2f}")
print("(Higher values indicate greater stabilization via disorder)")
print("=" * 60)
print(f"Perceptual Stabilization Efficiency η = {PSE:.4e}")

