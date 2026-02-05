import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- 
# SAME material models (Steps 1–2)
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
    eps_eff = (
        fk * eps_k**(-w) +
        fm * eps_m**(-w) +
        fa * eps_a**(-w)
    ) ** (-1.0 / w)
    return np.sqrt(eps_eff)

def effective_refractive_index_TM(fk, fm, fa, wl_nm):
    w = -1.5  # Freyer adjusted TM
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
# PARAMETERS FROM STEPS 1–2
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
total_depth = params["Nm"] * params["a"]

'''
# -----------------------------
# CONTINUUM FRACTIONS (derived)
# -----------------------------
def averaged_fractions(z):
    # Periodic structure as in Freyer
    period = params["a"]
    z_mod = z % period
    
    # Simplified: f_m peaks at center of melanosome layer
    if 0 <= z_mod < params["Dm"]:
        f_m = np.pi * (params["Dm"]/2)**2 / (params["a"]*params["b"])
    else:
        f_m = 0
    
    # Air channels in their own layer
    if params["Dm"] <= z_mod < params["Dm"] + params["Da"]:
        f_a = np.pi * (params["Da"]/2)**2 / (params["a"]*params["b"])
    else:
        f_a = 0
    
    f_k = 1.0 - f_m - f_a
    return f_k, f_m, f_a
'''

# -----------------------------
# CONTINUUM FRACTIONS (ensemble / continuum limit)
# -----------------------------

# Precompute period-averaged volume fractions
A_cell = params["a"] * params["b"]
f_m_bar = np.pi * (params["Dm"] / 2)**2 / A_cell
f_a_bar = np.pi * (params["Da"] / 2)**2 / A_cell
f_k_bar = 1.0 - f_m_bar - f_a_bar

def averaged_fractions(z):
    """
    Continuum (ensemble-averaged) volume fractions.
    Independent of depth z: fixed-point limit of disorder averaging.
    """
    return f_k_bar, f_m_bar, f_a_bar

# -----------------------------
# BUILD CONTINUUM STACK
# -----------------------------
def build_continuum_stack(wl_nm):
    stack = []
    n_k = n_keratin(wl_nm)
    stack.append(((n_k, n_k), params["cortex"] * 1e-9))  # TE/TM tuple
    
    # Define z_vals here
    z_vals = np.arange(0, total_depth, dz)
    
    for z in z_vals:
        fk, fm, fa = averaged_fractions(z)
        n_TE = effective_refractive_index_TE(fk, fm, fa, wl_nm)
        n_TM = effective_refractive_index_TM(fk, fm, fa, wl_nm)
        stack.append(((n_TE, n_TM), dz * 1e-9))
    
    return stack

# -----------------------------
# SAME TMM
# -----------------------------

def reflectance_TE_TM(stack, wavelengths_nm):
    """
    Calculate reflectance for TE and TM polarizations and average them.
    Stack format: [((n_TE, n_TM), thickness), ...]
    """
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
        
        n_in = 1.0
        n_out = 1.0

        Y_TE = (M_TE[1,0] + M_TE[1,1]*n_out) / (M_TE[0,0] + M_TE[0,1]*n_out)
        r_TE = (n_in - Y_TE) / (n_in + Y_TE)

        R_TE.append(np.abs(r_TE)**2)
        
        # TM calculation
        M_TM = np.eye(2, dtype=complex)
        for (n_TE, n_TM), d in stack:
            delta = k0 * n_TM * d
            eta = n_TM
            M_layer = np.array([
                [np.cos(delta), 1j*np.sin(delta)/eta],
                [1j*eta*np.sin(delta), np.cos(delta)]
            ])
            M_TM = M_layer @ M_TM

        n_in = 1.0
        n_out = 1.0

        Y_TM = (M_TM[1,0] + M_TM[1,1]*n_out) / (M_TM[0,0] + M_TM[0,1]*n_out)
        r_TM = (n_in - Y_TM) / (n_in + Y_TM)
        
        R_TM.append(np.abs(r_TM)**2)
    
    # Average TE and TM for unpolarized light (Freyer 2020)
    return (np.array(R_TE) + np.array(R_TM)) / 2

# -----------------------------
# SPECTRUM
# -----------------------------
wavelengths = np.linspace(400, 750, 400)
R = []
for wl in wavelengths:
    stack = build_continuum_stack(wl)
    R_wl = reflectance_TE_TM(stack, [wl])  
    R.append(R_wl[0])

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(7,4))
plt.plot(wavelengths, R, lw=2)
plt.xlabel("Wavelength (nm)", fontsize=18)
plt.ylabel("Reflectance", fontsize=18)
#plt.title("Region 4: continuum limit (derived from disorder)", fontsize=18)
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig('continum-EMT.png', dpi=300)
plt.show()
