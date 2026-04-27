import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from pysindy.optimizers import STLSQ, FROLS

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)


PLOTTING   = True
PLOT_MODES = True
PRINT_XI   = True
n_modes    = 16
TRAIN_FRAC = 0.99

# ── Optimizer switch ───────────────────────────────────────────────────────────
# "STLSQ" : Sequentially Thresholded Least Squares — iterative, good default
# "FROLS"  : Forward Regression Orthogonal Least Squares — greedy, term-budget style
OPTIMIZER = "FROLS"

# STLSQ settings
STLSQ_THRESHOLD = 1e-3   # sparsity knob: raise → fewer terms, lower → more terms
STLSQ_ALPHA     = 1e-6   # L2 ridge regularisation
STLSQ_MAX_ITER  = 50

# FROLS settings
FROLS_MAX_ITER  = 2      # max active terms per output (>=2 needed for a harmonic oscillator)
FROLS_ALPHA     = 0
FROLS_KAPPA     = 0

# ── Savitzky-Golay smoothed derivative ────────────────────────────────────────
# Why: raw FFT coefficients carry numerical noise; differentiating twice without
# smoothing explodes that noise and poisons Xi.
SAVGOL_WINDOW = 21   # must be odd; increase for noisier signals
SAVGOL_ORDER  = 5    # polynomial order; 3-5 is typical

def smooth(a):
    """Smooth a 1-D or 2-D (time × features) array."""
    if a.ndim == 1:
        return savgol_filter(a, SAVGOL_WINDOW, SAVGOL_ORDER)
    return np.column_stack(
        [savgol_filter(a[:, i], SAVGOL_WINDOW, SAVGOL_ORDER) for i in range(a.shape[1])]
    )

def derivative(a, dt):
    """Savitzky-Golay derivative (1-D or 2-D, time axis=0)."""
    if a.ndim == 1:
        return savgol_filter(a, SAVGOL_WINDOW, SAVGOL_ORDER, deriv=1, delta=dt)
    return np.column_stack(
        [savgol_filter(a[:, i], SAVGOL_WINDOW, SAVGOL_ORDER, deriv=1, delta=dt)
         for i in range(a.shape[1])]
    )


# ── 2-D wave equation solver ───────────────────────────────────────────────────
def solve_wave_equation_2d(
    c=1.0, L=2.0, Nx=100, Ny=100, n_steps=500, sigma=0.15,
):
    dx = L / Nx
    dt = 0.4 * dx / c
    r  = c * dt / dx

    x = np.linspace(-L/2, L/2, Nx+1)
    y = np.linspace(-L/2, L/2, Ny+1)
    t = np.arange(n_steps+1) * dt

    X, Y = np.meshgrid(x, y, indexing='ij')

    u    = np.zeros((n_steps+1, Nx+1, Ny+1))
    u[0] = np.exp(-((X+0.2)**2 + (Y+0.2)**2) / (2*sigma**2))

    u[1,1:-1,1:-1] = (
        u[0,1:-1,1:-1]
        + 0.5*r**2 * (
            u[0,2:,1:-1] + u[0,:-2,1:-1]
            + u[0,1:-1,2:] + u[0,1:-1,:-2]
            - 4*u[0,1:-1,1:-1]
        )
    )
    u[:, 0, :] = u[:, -2, :]
    u[:, -1, :] = u[:, 1, :]
    u[:, :, 0] = u[:, :, -2]
    u[:, :, -1] = u[:, 1, :]

    for n in range(1, n_steps):
        u[n+1,1:-1,1:-1] = (
            2*u[n,1:-1,1:-1]
            - u[n-1,1:-1,1:-1]
            + r**2 * (
                u[n,2:,1:-1] + u[n,:-2,1:-1]
                + u[n,1:-1,2:] + u[n,1:-1,:-2]
                - 4*u[n,1:-1,1:-1]
            )
        )

    u_fft = np.fft.rfft2(u, axes=(1, 2))
    return x, y, t, u, u_fft, dx, dt


x, y, t, u, modes, dx, dt = solve_wave_equation_2d()
Nt = len(t)

# ── Mode selection by energy ───────────────────────────────────────────────────
energy          = np.mean(np.abs(modes)**2, axis=0)
Nx_fft, Ny_fft = energy.shape
flat_energy     = energy.flatten()
top_indices     = np.argsort(flat_energy)[-n_modes:]
kx_idx, ky_idx  = np.unravel_index(top_indices, (Nx_fft, Ny_fft))
selected_modes  = list(zip(kx_idx, ky_idx))

# Modal-truncated reconstruction (baseline)
U_hat_filtered = np.zeros_like(modes, dtype=complex)
for i, j in selected_modes:
    U_hat_filtered[:, i, j] = modes[:, i, j]
U_reconstructed = np.fft.irfft2(
    U_hat_filtered, s=(u.shape[1], u.shape[2]), axes=(1, 2)
).real

N_train = int(Nt * TRAIN_FRAC)
t_train = t[:N_train]


# ══════════════════════════════════════════════════════════════════════════════
#  PER-MODE SINDy
# ══════════════════════════════════════════════════════════════════════════════
#
#  Each Fourier mode satisfies an independent harmonic oscillator:
#       ü_k = -ω_k² u_k     (wave equation in spectral space)
#
#  We therefore fit each mode with its own 4-dim state
#       x_k = [a_real, a_imag, v_real, v_imag]
#  and a library that contains ONLY those four columns.
#
#  This shrinks the search space from 64×64 (global) to 4×4 (per-mode),
#  and lets STLSQ reliably identify the two non-zero terms per equation.
# ══════════════════════════════════════════════════════════════════════════════

all_Xi  = []   # Xi_k  (4×4) for each mode
all_sim = []   # sol.y.T  (Nt×4) for each mode

print(f"Fitting per-mode SINDy [{OPTIMIZER}] …")
for idx, (kx, ky) in enumerate(selected_modes):

    # --- raw amplitudes ---------------------------------------------------
    a_r_raw = modes[:, kx, ky].real
    a_i_raw = modes[:, kx, ky].imag

    # --- smooth then differentiate ----------------------------------------
    a_r = smooth(a_r_raw)
    a_i = smooth(a_i_raw)
    v_r = derivative(a_r, dt)
    v_i = derivative(a_i, dt)
    v_r_dot = derivative(v_r, dt)   # second derivative via two SG passes
    v_i_dot = derivative(v_i, dt)

    # --- build per-mode state (Nt × 4) ------------------------------------
    X_k      = np.column_stack([a_r,    a_i,    v_r,    v_i   ])
    X_dot_k  = np.column_stack([v_r,    v_i,    v_r_dot, v_i_dot])

    # training slice
    Theta_k       = X_k     [:N_train]   # linear library = state itself
    X_dot_k_train = X_dot_k [:N_train]

    # --- optimizer --------------------------------------------------------
    if OPTIMIZER == "STLSQ":
        opt = STLSQ(threshold=STLSQ_THRESHOLD, alpha=STLSQ_ALPHA,
                    max_iter=STLSQ_MAX_ITER, normalize_columns=True)
    elif OPTIMIZER == "FROLS":
        opt = FROLS(max_iter=FROLS_MAX_ITER, alpha=FROLS_ALPHA, kappa=FROLS_KAPPA)
    else:
        raise ValueError(f"Unknown OPTIMIZER '{OPTIMIZER}'. Choose 'STLSQ' or 'FROLS'.")
    opt.fit(Theta_k, X_dot_k_train)
    Xi_k = opt.coef_.T          # shape (4, 4): Xi_k[i,j] = coeff of lib col i in eq j

    all_Xi.append(Xi_k)

    if PRINT_XI:
        print(f"  mode ({kx},{ky})  Xi =\n{np.round(Xi_k, 4)}\n")

    # --- simulate forward from t[0] to t[-1] ------------------------------
    X0_k = np.array([a_r[0], a_i[0], v_r[0], v_i[0]])

    def rhs(t_val, state, Xi=Xi_k):
        return (state.reshape(1, -1) @ Xi).flatten()

    sol = solve_ivp(
        rhs,
        t_span=(t[0], t[-1]),
        y0=X0_k,
        t_eval=t,
        method='RK45',
        rtol=1e-9,
        atol=1e-11,
    )
    all_sim.append(sol.y.T)          # shape (Nt, 4)

# ── Reconstruct field from per-mode simulations ───────────────────────────────
a_sim_real    = np.column_stack([sim[:, 0] for sim in all_sim])   # (Nt, n_modes)
a_sim_imag    = np.column_stack([sim[:, 1] for sim in all_sim])
a_sim_complex = a_sim_real + 1j * a_sim_imag

U_hat_sindy = np.zeros_like(modes, dtype=complex)
for idx, (i, j) in enumerate(selected_modes):
    U_hat_sindy[:, i, j] = a_sim_complex[:, idx]

U_sindy = np.fft.irfft2(
    U_hat_sindy, s=(u.shape[1], u.shape[2]), axes=(1, 2)
).real

# ── Reference: true modal amplitudes (real part) ──────────────────────────────
a_true_real = np.column_stack([modes[:, kx, ky].real for kx, ky in selected_modes])

# ── Plotting ──────────────────────────────────────────────────────────────────
if PLOTTING:
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    for i in range(Nt):
        if i % 150 == 0 or i == (Nt - 1):
            fig, axes = plt.subplots(3, 1, figsize=(8, 14))
            for ax, field, title in zip(
                axes,
                [u[i], U_reconstructed[i], U_sindy[i]],
                [f"True (step {i})",
                 f"Modal reconstruct (step {i})",
                 f"SINDy forecast (step {i})"],
            ):
                im = ax.imshow(
                    field.T, origin="lower",
                    extent=[x[0], x[-1], y[0], y[-1]],
                    cmap="Blues_r",
                )
                ax.set(xlabel="x", ylabel="y", title=title)
                fig.colorbar(im, ax=ax, label="Amplitude")
            plt.tight_layout()
            plt.show()

if PLOT_MODES:
    for idx, (kx, ky) in enumerate(selected_modes):
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(t,           a_true_real[:, idx],          color='black',    lw=1.5, label="True")
        ax.plot(t[:N_train], a_sim_real[:N_train, idx], '--', color='steelblue', lw=1.5, label="SINDy (train)")
        ax.plot(t[N_train:], a_sim_real[N_train:, idx], '--', color='tomato',    lw=1.5, label="SINDy (forecast)")
        ax.axvline(t[N_train], color='gray', linestyle=':', lw=1)
        ax.set(xlabel="Time", ylabel="Amplitude", title=f"Mode (kx={kx}, ky={ky})")
        ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()

print("\nSINDy-Modal MSE :", np.mean((U_reconstructed - U_sindy)**2))
print("Modal-True  MSE :", np.mean((u - U_reconstructed)**2))
print("SINDy-True  MSE :", np.mean((u - U_sindy)**2))