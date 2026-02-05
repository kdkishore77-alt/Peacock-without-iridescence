import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# SAME material models
# -----------------------------
def n_keratin(wl_nm):
    return 1.532 + 5890.0 / wl_nm**2

def n_melanin_complex(wl_nm):
    n = 1.648 + 23700.0 / wl_nm**2
    k = 0.56 * np.exp(-wl_nm / 270.0)
    return n + 1j * k

def effective_refractive_index_TE(fk, fm, fa, wl_nm):
    w = 2.5  # Freyer-adjusted
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
    w = -1.5  # Freyer-adjusted
    eps_k = n_keratin(wl_nm)**2
    eps_m = n_melanin_complex(wl_nm)**2
    eps_a = 1.0
    eps_eff = (
        fk * eps_k**(-w) +
        fm * eps_m**(-w) +
        fa * eps_a**(-w)
    ) ** (-1.0 / w)
    return np.sqrt(eps_eff)

# -----------------------------
# Geometry-derived continuum fractions
# -----------------------------
params = {
    "a": 195,
    "b": 150,
    "Dm": 120,
    "Da": 55,
    "Nm": 5,
    "cortex": 130
}

dz = 1.0
depth = params["Nm"] * params["a"]

f_m0 = np.pi * (params["Dm"]/2)**2 / (params["a"]*params["b"])
f_a0 = np.pi * (params["Da"]/2)**2 / (params["a"]*params["b"])
decay = 180.0

def averaged_fractions(z):
    fm = f_m0 * np.exp(-z / decay)
    fa = f_a0
    fk = 1.0 - fm - fa
    return fk, fm, fa

# -----------------------------
# Build continuum EMT stack
# -----------------------------
def build_stack(wl_nm):
    n_k = n_keratin(wl_nm)
    stack = [((n_k, n_k), params["cortex"] * 1e-9)]  # Tuple for TE/TM
    for z in np.arange(0, depth, dz):
        fk, fm, fa = averaged_fractions(z)
        n_TE = effective_refractive_index_TE(fk, fm, fa, wl_nm)
        n_TM = effective_refractive_index_TM(fk, fm, fa, wl_nm)
        stack.append(((n_TE, n_TM), dz * 1e-9))
    return stack
# -----------------------------
# Oblique-incidence TMM (TE)
# -----------------------------
def reflectance_TE_TM(stack, wl_nm, theta_deg):
    """
    Calculate reflectance for TE and TM polarizations at oblique incidence.
    Stack format: [((n_TE, n_TM), thickness), ...]
    Returns average for unpolarized light.
    """
    theta = np.radians(theta_deg)
    k0 = 2*np.pi / (wl_nm * 1e-9)
    
    # TE calculation
    M_TE = np.eye(2, dtype=complex)
    for (n_TE, n_TM), d in stack:
        cos_t = np.sqrt(1 - (np.sin(theta)/np.real(n_TE))**2)
        delta = k0 * n_TE * d * cos_t
        eta = n_TE * cos_t
        M_layer = np.array([
            [np.cos(delta), 1j*np.sin(delta)/eta],
            [1j*eta*np.sin(delta), np.cos(delta)]
        ])
        M_TE = M_layer @ M_TE
    
    Y_TE = (M_TE[1,0] + M_TE[1,1]) / (M_TE[0,0] + M_TE[0,1])
    r_TE = (1 - Y_TE) / (1 + Y_TE)
    R_TE = np.abs(r_TE)**2
    
    # TM calculation
    M_TM = np.eye(2, dtype=complex)
    for (n_TE, n_TM), d in stack:

        cos_t = np.sqrt(1 - (np.sin(theta)/np.real(n_TM))**2)
        delta = k0 * n_TM * d * cos_t
        eta = n_TM / cos_t   
        M_layer = np.array([
            [np.cos(delta), 1j*np.sin(delta)/eta],
            [1j*eta*np.sin(delta), np.cos(delta)]
        ])
        M_TM = M_layer @ M_TM
    
    Y_TM = (M_TM[1,0] + M_TM[1,1]) / (M_TM[0,0] + M_TM[0,1])
    r_TM = (1 - Y_TM) / (1 + Y_TM)
    R_TM = np.abs(r_TM)**2
    
    # Average for unpolarized light (Freyer 2020)
    return (R_TE + R_TM) / 2

# -----------------------------
# Angle averaging
# -----------------------------
wavelengths = np.linspace(400, 750, 350)
angles = np.linspace(0, 40, 7)

R = np.zeros_like(wavelengths)

for i, wl in enumerate(wavelengths):
    stack = build_stack(wl)
    # Use new TE/TM function
    R[i] = np.mean([reflectance_TE_TM(stack, wl, th) for th in angles])
# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(wavelengths, R, lw=2.5)
plt.xlabel("Wavelength (nm)", fontsize=18)
plt.ylabel("Reflectance", fontsize=18)
#plt.title("Region 4: angle-robust structural colour (EMT continuum)", fontsize=18)
plt.ylim(0, 0.2)
plt.tight_layout()
plt.savefig('angle-averaged.png', dpi=300)
plt.show()

