import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# -----------------------------
# Freyer 2020 – Region 4 parameters (Table 1)
# -----------------------------
params = {
    "a": 195e-9,      # m, lattice period (longitudinal)
    "b": 150e-9,      # m, lateral period
    "c": 130e-9,      # m, cortex thickness
    "Dm": 120e-9,     # m, melanosome diameter
    "Da": 55e-9,      # m, air-channel diameter
    "Nm": 5           # number of periods
}

# -----------------------------
# Material dispersion (Freyer 2020)
# -----------------------------
def n_keratin(wl_nm):
    """Keratin refractive index (real)"""
    return 1.532 + 5890.0 / wl_nm**2

def n_melanin_complex(wl_nm):
    """Melanin refractive index (complex)"""
    wl = wl_nm
    n_real = 1.648 + 23700.0 / wl**2
    k = 0.56 * np.exp(-wl / 270.0)  # extinction coefficient
    return n_real + 1j * k

def epsilon_material(wl_nm, material):
    """Return complex permittivity for given material"""
    if material == 'keratin':
        n = n_keratin(wl_nm)
        return n**2 + 0j
    elif material == 'melanin':
        nk = n_melanin_complex(wl_nm)
        return nk**2
    elif material == 'air':
        return 1.0 + 0j
    else:
        raise ValueError(f"Unknown material: {material}")

# -----------------------------
# Effective-medium mixing rule (Freyer 2020, Eq. 1)
# -----------------------------
def effective_permittivity(fk, fm, fa, wl_nm, w=2):
    """
    Power-law mixing rule for effective permittivity.
    w = 2 for polarization-averaged normal incidence (Freyer 2020)
    """
    eps_k = epsilon_material(wl_nm, 'keratin')
    eps_m = epsilon_material(wl_nm, 'melanin')
    eps_a = epsilon_material(wl_nm, 'air')
    
    # Handle complex numbers carefully
    eps_eff = (
        fk * eps_k**(-w) +
        fm * eps_m**(-w) +
        fa * eps_a**(-w)
    )**(-1.0 / w)
    
    return eps_eff

def effective_refractive_index(fk, fm, fa, wl_nm, w=2):
    """Complex refractive index from effective permittivity"""
    eps_eff = effective_permittivity(fk, fm, fa, wl_nm, w)
    return np.sqrt(eps_eff)

# -----------------------------
# Geometry: Area fractions at depth z 
# -----------------------------
def area_fraction_circle_at_z(D, z_offset, z, b):
    """
    Calculate area fraction of a cylinder (diameter D) intersected
    by plane at depth z, within unit cell width b.
    
    z_offset: center position of cylinder in z-direction
    """
    R = D / 2.0
    dz = z - z_offset
    
    if abs(dz) > R:
        return 0.0
    
    # Intersection width at this z-slice
    width = 2.0 * np.sqrt(R**2 - dz**2)
    
    # Area fraction in xy-plane
    # The cylinder extends infinitely in y-direction, width in x-direction
    # Unit cell area in xy-plane: a*b, but we're at fixed z
    # For area fraction at depth z: (width * b) / (a*b) = width/a
    return width / params["a"]

def filling_fractions_at_depth(z, params):
    """
    Calculate filling fractions fk, fm, fa at depth z.
    
    Geometry from Freyer 2020 Fig. 1:
    - Melanosomes centered at z = 0, a, 2a, ...
    - Air channels centered at z = Dm + Da/2, a + Dm + Da/2, ...
    """
    a = params["a"]
    Dm = params["Dm"]
    Da = params["Da"]
    b = params["b"]
    
    # Initialize fractions
    fm_total = 0.0
    fa_total = 0.0
    
    # Sum contributions from all unit cells (periodic)
    for n in range(params["Nm"]):
        # Melanosome in this unit cell
        z_m_offset = n * a
        fm_total += area_fraction_circle_at_z(Dm, z_m_offset, z, b)
        
        # Air channel in this unit cell (offset by Dm + Da/2)
        z_a_offset = n * a + Dm + Da/2.0
        fa_total += area_fraction_circle_at_z(Da, z_a_offset, z, b)
    
    fk = 1.0 - fm_total - fa_total
    
    # Ensure non-negative (numerical stability)
    fk = max(0.0, min(1.0, fk))
    fm_total = max(0.0, min(1.0, fm_total))
    fa_total = max(0.0, min(1.0, fa_total))
    
    # Renormalize if needed
    total = fk + fm_total + fa_total
    if total > 0:
        fk /= total
        fm_total /= total
        fa_total /= total
    
    return fk, fm_total, fa_total

# -----------------------------
# Build complete depth profile
# -----------------------------
def build_depth_profile(params, wl_nm=550.0, dz_nm=1.0):
    """
    Build depth profile of complex refractive index.
    
    Returns: (z_positions, n_complex)
    """
    dz = dz_nm * 1e-9  # convert to meters
    total_depth = params["c"] + params["Nm"] * params["a"]
    
    z_vals = np.arange(0, total_depth, dz)
    n_vals = []
    
    for z in z_vals:
        if z < params["c"]:
            # Cortex: pure keratin (or could use effective medium with fm=fa=0)
            n = n_keratin(wl_nm) + 0j
        else:
            # Barbule region: effective medium
            z_barb = z - params["c"]
            fk, fm, fa = filling_fractions_at_depth(z_barb, params)
            n = effective_refractive_index(fk, fm, fa, wl_nm, w=2)
        
        n_vals.append(n)
    
    return np.array(z_vals), np.array(n_vals)

# -----------------------------
# TMM reflectance calculation 
# -----------------------------
def transfer_matrix_2x2(n, d, k0, polarization='avg'):
    """
    Create 2x2 transfer matrix for a single layer.
    
    For normal incidence, TE and TM are identical.
    For polarization average, we use w=2 in mixing rule.
    """
    if polarization == 'TE':
        eta = n
    elif polarization == 'TM':
        eta = n  # For normal incidence, same as TE
    elif polarization == 'avg':
        eta = n  # Already averaged in effective medium
    else:
        raise ValueError("polarization must be 'TE', 'TM', or 'avg'")
    
    delta = k0 * n * d
    
    M = np.array([
        [np.cos(delta), 1j * np.sin(delta) / eta],
        [1j * eta * np.sin(delta), np.cos(delta)]
    ], dtype=complex)
    
    return M

def calculate_reflectance(z_profile, n_profile, wavelengths_nm):
    """
    Calculate reflectance spectrum using TMM.
    
    z_profile: depth positions (m)
    n_profile: complex refractive indices at each position
    wavelengths_nm: wavelengths in nm
    """
    reflectance = []
    
    for wl_nm in wavelengths_nm:
        # Convert to meters
        wl = wl_nm * 1e-9
        k0 = 2.0 * np.pi / wl
        
        # Initialize transfer matrix
        M_total = np.eye(2, dtype=complex)
        
        # Calculate for each layer
        for i in range(len(z_profile) - 1):
            # Layer thickness
            d = z_profile[i+1] - z_profile[i]
            
            # Average refractive index in this layer
            n_avg = (n_profile[i] + n_profile[i+1]) / 2.0
            
            # Create layer matrix
            M_layer = transfer_matrix_2x2(n_avg, d, k0, polarization='avg')
            M_total = M_layer @ M_total
        
        # Incident medium (air)
        n_inc = 1.0 + 0j
        eta_inc = n_inc
        
        # Exit medium (air)
        n_exit = 1.0 + 0j
        eta_exit = n_exit
        
        # Calculate reflectance
        Y = (M_total[1,0] + M_total[1,1] * eta_exit) / \
            (M_total[0,0] + M_total[0,1] * eta_exit)
        
        r = (eta_inc - Y) / (eta_inc + Y)
        R = np.abs(r)**2
        
        reflectance.append(R)
    
    return np.array(reflectance)

# -----------------------------
# Main analysis
# -----------------------------
def main():
    # Create wavelength array
    wavelengths_nm = np.linspace(400, 750, 351)
    
    # Build profiles at a reference wavelength
    z_profile, n_profile = build_depth_profile(params, wl_nm=550.0, dz_nm=1.0)
    
    # Calculate reflectance at each wavelength
    # Note: For accurate results, we should rebuild profile for each wavelength
    # due to material dispersion. Here's an efficient approach:
    reflectance_spectrum = []
    
    for wl_nm in wavelengths_nm:
        z_profile, n_profile = build_depth_profile(params, wl_nm=wl_nm, dz_nm=1.0)
        R = calculate_reflectance(z_profile, n_profile, [wl_nm])[0]
        reflectance_spectrum.append(R)
    
    reflectance_spectrum = np.array(reflectance_spectrum)
    
    # Plot results
    plt.figure(figsize=(14, 5))
    
    # 1. Refractive index profile at 550 nm
    plt.subplot(1, 3, 1)
    z_profile_550, n_profile_550 = build_depth_profile(params, wl_nm=550.0, dz_nm=1.0)
    plt.plot(z_profile_550 * 1e9, np.real(n_profile_550), 'b-', linewidth=2)
    plt.xlabel('Depth (nm)', fontsize=12)
    plt.ylabel('Refractive index (real part)', fontsize=12)
    plt.title('Refractive index profile\n(λ = 550 nm)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 2. Imaginary part (absorption)
    plt.subplot(1, 3, 2)
    plt.plot(z_profile_550 * 1e9, np.imag(n_profile_550), 'r-', linewidth=2)
    plt.xlabel('Depth (nm)', fontsize=12)
    plt.ylabel('Extinction coefficient', fontsize=12)
    plt.title('Absorption profile\n(λ = 550 nm)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 3. Reflectance spectrum
    plt.subplot(1, 3, 3)
    plt.plot(wavelengths_nm, reflectance_spectrum, 'k-', linewidth=2)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance', fontsize=12)
    plt.title('Reflectance spectrum\n(Region 4)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.7)
    
    plt.tight_layout()
    plt.savefig('photonic-crystal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print key metrics
    peak_idx = np.argmax(reflectance_spectrum)
    peak_wl = wavelengths_nm[peak_idx]
    peak_R = reflectance_spectrum[peak_idx]
    
    print(f"Peak reflectance: {peak_R:.3f} at {peak_wl:.0f} nm")
    print(f"FWHM: To be calculated from spectrum")
    
    return z_profile_550, n_profile_550, wavelengths_nm, reflectance_spectrum

if __name__ == "__main__":
    z_prof, n_prof, wls, R_spectrum = main()
