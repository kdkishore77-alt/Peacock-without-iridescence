import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Region 4 parameters (Freyer)
# -----------------------------
a0 = 195      # nm
cortex = 130  # nm
Nm = 5

params = {
    "a": a0,        # nm
    "b": 150,       # nm  
    "c": cortex,    # nm
    "Dm": 120,      # nm
    "Da": 55,       # nm
    "Nm": Nm
}

# -----------------------------
# Material functions (same as before)
# -----------------------------
def n_keratin(wl_nm):
    return 1.532 + 5890.0 / wl_nm**2

def n_melanin_complex(wl_nm):
    n = 1.648 + 23700.0 / wl_nm**2
    k = 0.56 * np.exp(-wl_nm / 270.0)
    return n + 1j * k

def effective_refractive_index_TE(fk, fm, fa, wl_nm):
    w = 2.5
    eps_k = n_keratin(wl_nm)**2
    eps_m = n_melanin_complex(wl_nm)**2
    eps_a = 1.0
    eps_eff = (fk * eps_k**(-w) + fm * eps_m**(-w) + fa * eps_a**(-w))**(-1.0/w)
    return np.sqrt(eps_eff)

def effective_refractive_index_TM(fk, fm, fa, wl_nm):
    w = -1.5
    eps_k = n_keratin(wl_nm)**2
    eps_m = n_melanin_complex(wl_nm)**2
    eps_a = 1.0
    eps_eff = (fk * eps_k**(-w) + fm * eps_m**(-w) + fa * eps_a**(-w))**(-1.0/w)
    return np.sqrt(eps_eff)

# -----------------------------
# TMM function (keep as is)
# -----------------------------
def reflectance_TE_TM(stack, wavelengths_nm):
    R_TE = []
    R_TM = []
    
    for wl_nm in wavelengths_nm:
        k0 = 2 * np.pi / (wl_nm * 1e-9)
        
        # TE calculation
        M_TE = np.eye(2, dtype=complex)
        for (n_TE, n_TM), d in stack:
            delta = k0 * n_TE * d
            eta = n_TE
            M_layer = np.array([
                [np.cos(delta), 1j*np.sin(delta)/eta],
                [1j*eta*np.sin(delta), np.cos(delta)]
            ])
            M_TE = M_layer @ M_TE
        
        eta_sub = 1.0 / n_keratin(wl_nm)

        Y_TE = (M_TE[1,0] + M_TE[1,1]*eta_sub) / (M_TE[0,0] + M_TE[0,1]*eta_sub)
        r_TE = (1 - Y_TE) / (1 + Y_TE)
        R_TE.append(np.abs(r_TE)**2)
        
        # TM calculation
        M_TM = np.eye(2, dtype=complex)
        for (n_TE, n_TM), d in stack:
            delta = k0 * n_TM * d
            eta = 1.0 / n_TM
            M_layer = np.array([
                [np.cos(delta), 1j*np.sin(delta)/eta],
                [1j*eta*np.sin(delta), np.cos(delta)]
            ])
            M_TM = M_layer @ M_TM
        
        Y_TM = (M_TM[1,0] + M_TM[1,1]*eta_sub) / (M_TM[0,0] + M_TM[0,1]*eta_sub)
        r_TM = (1 - Y_TM) / (1 + Y_TM)
        R_TM.append(np.abs(r_TM)**2)
    
    return np.array(R_TE)

# -----------------------------
#  Build disordered stack 
# -----------------------------
def build_disordered_stack_fixed(params, wl_nm, a_perturbed):
    stack = []

    # Cortex
    n_k = n_keratin(wl_nm)
    stack.append(((n_k, n_k), 2 * params["c"] * 1e-9))

    # Period-averaged filling fractions (same every period)
    a0 = params["a"]
    b = params["b"]
    Dm = params["Dm"]
    Da = params["Da"]

    fm = np.pi * (Dm/2)**2 / (a0 * b)
    fa = np.pi * (Da/2)**2 / (a0 * b)
    fk = 1 - fm - fa

    n_TE = effective_refractive_index_TE(fk, fm, fa, wl_nm)
    n_TM = effective_refractive_index_TM(fk, fm, fa, wl_nm)

    # Disordered multilayer: thickness disorder ONLY
    for _ in range(params["Nm"]):
        stack.append(((n_TE, n_TM), a_perturbed * 1e-9))

    return stack

# -----------------------------
#  Ensemble averaging workflow
# -----------------------------
wavelengths = np.linspace(400, 750, 551)  
n_realizations = 100  
sigma_Dm = 0.05
sigma_Da = 0.05

R_ensemble = np.zeros_like(wavelengths, dtype=float)

# Store individual realizations for debugging
all_realizations = []

for realization in range(n_realizations):
    print(f"Processing realization {realization+1}/{n_realizations}")

    # Freyer-style disorder: perturb period thickness ONLY
    a_pert = params["a"] * (1 + sigma_Da * np.random.randn())

   
    # Calculate FULL spectrum for this realization
    R_this_realization = np.zeros_like(wavelengths, dtype=float)
    
    # OPTIMIZATION: Build stack once at reference wavelength, 
    # then adjust for dispersion at each wavelength
    for i, wl in enumerate(wavelengths):
        stack = build_disordered_stack_fixed(params, wl, a_pert)
        R_wl = reflectance_TE_TM(stack, [wl])
        R_this_realization[i] = R_wl[0]
    
    all_realizations.append(R_this_realization)
    R_ensemble += R_this_realization

# -----------------------------
# Calculate and plot results
# -----------------------------
R_avg = R_ensemble / n_realizations

# For comparison, calculate ordered spectrum
print("\nCalculating ordered reference spectrum...")
R_ordered = np.zeros_like(wavelengths)
for i, wl in enumerate(wavelengths):
    stack_ordered = build_disordered_stack_fixed(params, wl, params["a"])
    R_wl = reflectance_TE_TM(stack_ordered, [wl])
    R_ordered[i] = R_wl[0]

# -----------------------------
# Plot with proper comparison
# -----------------------------
plt.figure(figsize=(10, 6))

# Plot individual realizations (transparent)
for i in range(min(10, n_realizations)):
    plt.plot(wavelengths, 100 * all_realizations[i], 'gray', alpha=0.15, lw=0.5)

# Plot ensemble average
plt.plot(wavelengths, 100 * R_avg, 'b-', lw=3, label=f'Ensemble avg (N={n_realizations})')

# Plot ordered reference
plt.plot(wavelengths, 100 * R_ordered, 'r--', lw=2.5, label='Ordered reference')

plt.xlabel("Wavelength (nm)", fontsize=20)
plt.ylabel("Reflectance (%)", fontsize=20)
#plt.title(rf"Disorder Effect on Structural Color $\sigma_{{Dm}}$={sigma_Dm}, $\sigma_{{Da}}$={sigma_Da}",fontsize=16)
plt.legend(loc='upper left', fontsize=16)
plt.grid(alpha=0.3)
plt.ylim(0, 70)

# Add inset showing variability
ax_inset = plt.axes([0.65, 0.65, 0.25, 0.25])
for i in range(min(5, n_realizations)):
    ax_inset.plot(wavelengths, all_realizations[i], alpha=0.5, lw=2.0)
ax_inset.set_xlim(500, 600)
ax_inset.set_ylim(0, 0.6)
ax_inset.set_xlabel("λ (nm)", fontsize=14)
ax_inset.set_ylabel("R", fontsize=14)
ax_inset.set_title("Individual realizations", fontsize=14)
ax_inset.grid(alpha=0.2)

plt.gcf().set_constrained_layout(True)
plt.savefig('disorder.png', dpi=600, bbox_inches='tight')
plt.show()

# -----------------------------
# Quantitative analysis
# -----------------------------
print("\n" + "="*50)
print("QUANTITATIVE ANALYSIS")
print("="*50)

# Find peaks
peak_idx_ordered = np.argmax(R_ordered)
peak_wl_ordered = wavelengths[peak_idx_ordered]
peak_R_ordered = R_ordered[peak_idx_ordered]

peak_idx_disordered = np.argmax(R_avg)
peak_wl_disordered = wavelengths[peak_idx_disordered]
peak_R_disordered = R_avg[peak_idx_disordered]

print(f"Ordered spectrum:")
print(f"  Peak reflectance: {100*peak_R_ordered:.1f}% at {peak_wl_ordered:.0f} nm")

print(f"\nDisordered ensemble (avg of {n_realizations} realizations):")
print(f"  Peak reflectance: {100*peak_R_disordered:.1f}% at {peak_wl_disordered:.0f} nm")
print(f"  Peak shift: {peak_wl_disordered-peak_wl_ordered:.1f} nm")
print(f"  Peak change: {100*(peak_R_disordered-peak_R_ordered)/peak_R_ordered:.1f}%")

# Calculate standard deviation across ensemble
R_matrix = np.array(all_realizations)
std_across_wavelengths = np.std(R_matrix, axis=0)
avg_std = np.mean(std_across_wavelengths[200:400])  # Around peak region
print(f"\nEnsemble variability:")
print(f"  Average std dev across realizations: {100*avg_std:.2f}%")
