import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.optimizers import FROLS


import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)


PLOTTING   = True
PLOT_MODES = False
PRINT_XI   = True
n_modes    = 15
TRAIN_FRAC = 0.3
L = 2
c = 1

FROLS_MAX_ITER  = 1
FROLS_ALPHA     = 0
FROLS_KAPPA     = 0


def compute_time_derivative(a, dt):
    da = np.zeros_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
    da[0] = (a[1]  - a[0])   / dt
    da[-1] = (a[-1] - a[-2])  / dt
    return da


def solve_wave_equation_2d(
    c=1.0, L=2.0, Nx=100, Ny=100, n_steps=1500, sigma=0.15,
):
    dx = L / Nx
    dt = 0.4 * dx / c
    r  = c * dt / dx

    x = np.linspace(-L/2, L/2, Nx+1)
    y = np.linspace(-L/2, L/2, Ny+1)
    t = np.arange(n_steps+1) * dt

    X, Y = np.meshgrid(x, y, indexing='ij')

    """Establish initial amplitude and velocity"""
    x0_disp, y0_disp = +0.2, +0.2
    u0 = np.exp(-((X - x0_disp)**2 + (Y - y0_disp)**2) / (2*sigma**2))

    x0_vel, y0_vel = -0.1, -0.3
    v0 = np.exp(-((X - x0_vel)**2 + (Y - y0_vel)**2) / (2*sigma**2))

    u = np.zeros((n_steps+1, Nx+1, Ny+1))
    u[0] = u0

    """Finite difference laplacian approximation
    for u_xx + u_yy"""
    lap_u0 = (
        u0[2:, 1:-1] + u0[:-2, 1:-1] +
        u0[1:-1, 2:] + u0[1:-1, :-2] -
        4.0 * u0[1:-1, 1:-1]
    ) / dx**2

    """Taylor expansion to approximate first step"""
    u[1, 1:-1, 1:-1] = (
        u0[1:-1, 1:-1]
        + dt * v0[1:-1, 1:-1]
        + 0.5 * dt**2 * c**2 * lap_u0
    )

    """Apply periodic boundary conditions"""
    u[1, 0, :] = u[1, -2, :]
    u[1, -1, :] = u[1, 1, :]
    u[1, :, 0] = u[1, :, -2]
    u[1, :, -1] = u[1, :, 1]

    for n in range(1, n_steps):
        """Use second order/laplacian finite difference
        approximations for main stepping"""
        u[n+1, 1:-1, 1:-1] = (
            2.0 * u[n, 1:-1, 1:-1]
            - u[n-1, 1:-1, 1:-1]
            + r**2 * (
                u[n, 2:, 1:-1] + u[n, :-2, 1:-1] +
                u[n, 1:-1, 2:] + u[n, 1:-1, :-2] -
                4.0 * u[n, 1:-1, 1:-1]
            )
        )
        """Apply periodic boundary conditions"""
        u[n+1, 0, :] = u[n+1, -2, :]
        u[n+1, -1, :] = u[n+1, 1, :]
        u[n+1, :, 0] = u[n+1, :, -2]
        u[n+1, :, -1] = u[n+1, :, 1]

    u_fft = np.fft.rfft2(u, axes=(1, 2))
    return x, y, t, u, u_fft, dx, dt


x, y, t, u, modes, dx, dt = solve_wave_equation_2d()
Nt = len(t)

energy          = np.mean(np.abs(modes)**2, axis=0)
Nx_fft, Ny_fft = energy.shape
flat_energy     = energy.flatten()
top_indices     = np.argsort(flat_energy)[-n_modes:]
kx_idx, ky_idx  = np.unravel_index(top_indices, (Nx_fft, Ny_fft))
selected_modes = list(zip(kx_idx, ky_idx))

selected_modes = [(kx, ky) for kx, ky in selected_modes if not (kx == 0 and ky == 0)]

Nx_grid = u.shape[1]
Ny_grid = u.shape[2]


def physical_k_sq(kx, ky, Nx_grid, Ny_grid, L):
    kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
    ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
    return ((2 * np.pi / L) ** 2) * (kx_w ** 2 + ky_w ** 2)


U_hat_filtered = np.zeros_like(modes, dtype=complex)
for i, j in selected_modes:
    U_hat_filtered[:, i, j] = modes[:, i, j]
U_reconstructed = np.fft.irfft2(
    U_hat_filtered, s=(u.shape[1], u.shape[2]), axes=(1, 2)
).real

N_train = int(Nt * TRAIN_FRAC)
t_train = t[:N_train]

all_Xi  = []
all_sim = []

print(f"Fitting SINDy")
for idx, (kx, ky) in enumerate(selected_modes):

    a_r_raw = modes[:, kx, ky].real
    a_i_raw = modes[:, kx, ky].imag

    a_r = a_r_raw
    a_i = a_i_raw
    v_r = compute_time_derivative(a_r, dt)
    v_i = compute_time_derivative(a_i, dt)
    v_r_dot = compute_time_derivative(v_r, dt)
    v_i_dot = compute_time_derivative(v_i, dt)

    X_k = np.column_stack([a_r, a_i, v_r, v_i])
    X_dot_k = np.column_stack([v_r, v_i, v_r_dot, v_i_dot])

    Theta_k = X_k [:N_train]
    X_dot_k_train = X_dot_k [:N_train]

    opt = FROLS(max_iter=FROLS_MAX_ITER, alpha=FROLS_ALPHA, kappa=FROLS_KAPPA)

    opt.fit(Theta_k, X_dot_k_train)
    Xi_k = opt.coef_.T

    all_Xi.append(Xi_k)

    if PRINT_XI:
        kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
        ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
        kx_phys = (2 * np.pi / L) * kx_w
        ky_phys = (2 * np.pi / L) * ky_w
        omega_sq_expected  = c ** 2 * (kx_phys ** 2 + ky_phys ** 2)
        omega_sq_recovered_r = -Xi_k[0, 2]
        omega_sq_recovered_i = -Xi_k[1, 3]
        print(
            f"mode (kx_idx={kx}, ky_idx={ky})  "
            f"Xi =\n{np.round(Xi_k, 2)}\n"
            f"Expected ω²: {omega_sq_expected:.4f}\n"
            f"Recovered real:{omega_sq_recovered_r:.4f} imag:{omega_sq_recovered_i:.4f}\n"
        )

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
    all_sim.append(sol.y.T)

a_sim_real    = np.column_stack([sim[:, 0] for sim in all_sim])
a_sim_imag    = np.column_stack([sim[:, 1] for sim in all_sim])
a_sim_complex = a_sim_real + 1j * a_sim_imag

U_hat_sindy = np.zeros_like(modes, dtype=complex)
for idx, (i, j) in enumerate(selected_modes):
    U_hat_sindy[:, i, j] = a_sim_complex[:, idx]

U_sindy = np.fft.irfft2(
    U_hat_sindy, s=(u.shape[1], u.shape[2]), axes=(1, 2)
).real

a_true_real = np.column_stack([modes[:, kx, ky].real for kx, ky in selected_modes])

if PLOTTING:
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    for i in range(Nt):
        if i % 200 == 0 or i == (Nt - 1):
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
        kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
        ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
        ax.set(xlabel="Time", ylabel="Amplitude",
               title=f"Mode (kx_phys={kx_w}, ky_phys={ky_w})  [idx=({kx},{ky})]")
        ax.legend(); ax.grid(True)
        plt.tight_layout(); plt.show()

print("\nSINDy-Modal MSE :", np.mean((U_reconstructed - U_sindy)**2))
print("Modal-True  MSE :", np.mean((u - U_reconstructed)**2))
print("SINDy-True  MSE :", np.mean((u - U_sindy)**2))